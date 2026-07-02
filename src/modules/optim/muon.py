"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz).

Muon optimizes the *hidden weight matrices* of a network by orthogonalizing the
momentum update via a Newton-Schulz iteration. Every other parameter -- input
embeddings, the classifier head, and all 1D gains/biases -- should keep being
optimized with a standard AdamW step, which is why the recommended entry point
here is :class:`MuonWithAuxAdam` / :class:`SingleDeviceMuonWithAuxAdam`.

The implementation follows the reference from Keller Jordan
(https://github.com/KellerJordan/Muon), with a small generalization so that the
orthogonalization also works for convolutional filters of arbitrary rank (e.g.
the 5D group-convolution filters used by the equivariant net): any parameter
with more than two dimensions is flattened to a 2D matrix before the
Newton-Schulz iteration and reshaped back afterwards.
"""

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn

__all__ = [
    "zeropower_via_newtonschulz5",
    "muon_update",
    "adam_update",
    "Muon",
    "SingleDeviceMuon",
    "MuonWithAuxAdam",
    "SingleDeviceMuonWithAuxAdam",
    "split_parameters_for_muon",
    "build_muon_optimizer",
]


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Orthogonalize ``G`` via a quintic Newton-Schulz iteration.

    Computes an approximation of the orthogonal factor ``U V^T`` of ``G`` (from
    its SVD ``G = U S V^T``) using a fixed number of matmul-only iterations. The
    quintic coefficients are tuned so the iteration converges very fast while
    only requiring that the singular values end up in roughly ``[0.7, 1.3]``
    (exact orthogonality is not needed for the optimizer to work well).

    :param G: A 2D (or batched-2D) tensor to orthogonalize.
    :param steps: Number of Newton-Schulz iterations.
    :return: The orthogonalized tensor, same shape and dtype as ``G``.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1, so the iteration is contractive.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute the (orthogonalized) Muon update for a single parameter.

    :param grad: The parameter gradient.
    :param momentum: The momentum buffer (updated in place).
    :param beta: Momentum coefficient.
    :param ns_steps: Number of Newton-Schulz iterations.
    :param nesterov: Whether to use a Nesterov-style momentum lookahead.
    :return: The update, reshaped to match ``grad``.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    # Flatten conv (and higher-rank group-conv) filters into 2D matrices.
    if update.ndim > 2:
        update = update.view(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # Scale so the update RMS roughly matches that of an Adam step regardless of
    # the matrix aspect ratio.
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: Sequence[float],
    eps: float,
) -> torch.Tensor:
    """Compute a bias-corrected Adam update for a single parameter."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz (distributed variant).

    Muon internally runs standard SGD-momentum and then orthogonalizes each
    update using a Newton-Schulz iteration. This optimizer is intended **only**
    for the 2D+ hidden weight matrices of a network; embeddings, the classifier
    head and all scalar/vector (``ndim < 2``) parameters must be optimized with
    a separate AdamW optimizer (see :class:`MuonWithAuxAdam`).

    This variant shards the orthogonalization work across ranks and therefore
    requires an initialized ``torch.distributed`` process group. For single-GPU
    / non-distributed runs use :class:`SingleDeviceMuon` instead.

    :param params: The hidden weight matrices to optimize.
    :param lr: Learning rate, in units of spectral norm per update.
    :param weight_decay: AdamW-style decoupled weight decay.
    :param momentum: Momentum coefficient.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        assert len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        # Sort by size so ranks receive balanced work in the distributed loop.
        params = sorted(params, key=lambda x: x.numel(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params: List[nn.Parameter] = group["params"]
            pad = [torch.empty_like(params[-1])] * (
                (world_size - len(params) % world_size) % world_size
            )
            params_pad = params + pad
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + world_size],
                    params_pad[base_i + rank],
                )

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """Non-distributed :class:`Muon` for single-device training."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad, state["momentum_buffer"], beta=group["momentum"]
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def _init_muon_group(group: dict) -> None:
    group["lr"] = group.get("lr", 0.02)
    group["momentum"] = group.get("momentum", 0.95)
    group["weight_decay"] = group.get("weight_decay", 0.0)
    assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}


def _init_adam_group(group: dict) -> None:
    group["lr"] = group.get("lr", 3e-4)
    group["betas"] = group.get("betas", (0.9, 0.95))
    group["eps"] = group.get("eps", 1e-10)
    group["weight_decay"] = group.get("weight_decay", 0.0)
    assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for the hidden matrices, AdamW for everything else (distributed).

    Wraps both optimizers behind a single ``.step()``. The caller specifies, per
    parameter group, whether Muon should be used via the ``use_muon`` flag::

        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
            dict(params=other_params, use_muon=False, lr=3e-4,
                 betas=(0.9, 0.95), weight_decay=0.01),
        ]
        optimizer = MuonWithAuxAdam(param_groups)

    Muon groups accept ``lr``, ``momentum`` and ``weight_decay``; Adam groups
    accept ``lr``, ``betas``, ``eps`` and ``weight_decay``.

    This variant requires an initialized ``torch.distributed`` process group;
    use :class:`SingleDeviceMuonWithAuxAdam` otherwise.
    """

    def __init__(self, param_groups: List[dict]) -> None:
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.numel(), reverse=True
                )
                _init_muon_group(group)
            else:
                _init_adam_group(group)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            if group["use_muon"]:
                params: List[nn.Parameter] = group["params"]
                if not params:
                    continue
                pad = [torch.empty_like(params[-1])] * (
                    (world_size - len(params) % world_size) % world_size
                )
                params_pad = params + pad
                for base_i in range(0, len(params), world_size):
                    if base_i + rank < len(params):
                        p = params[base_i + rank]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad, state["momentum_buffer"], beta=group["momentum"]
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + world_size],
                        params_pad[base_i + rank],
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed :class:`MuonWithAuxAdam` for single-device training."""

    def __init__(self, param_groups: List[dict]) -> None:
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                _init_muon_group(group)
            else:
                _init_adam_group(group)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# Substrings identifying parameters that must NOT be optimized by Muon: input
# embeddings, the first ("stem"/patch) conv, and the classifier head. These are
# matched case-insensitively against fully-qualified parameter names.
_ADAMW_NAME_KEYWORDS: Tuple[str, ...] = (
    "embed",  # pos_embedding / positional & patch embeddings
    "class_token",  # ViT class token
    "conv_proj",  # ViT patch-embedding conv (the "first" conv)
    "stem",  # first conv of the (equivariant) conv nets
    "head",  # ViT classifier head (`heads.*`)
    "classifier",  # EquivariantNet classifier head
)


def split_parameters_for_muon(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split ``model`` parameters into (Muon, AdamW) groups.

    Muon optimizes the hidden weight *matrices* (``ndim >= 2``) of the body of
    the network. Everything else -- all 1D gains/biases, the input embeddings,
    the first/patch conv and the classifier head -- goes to AdamW.

    :param model: The model whose parameters should be partitioned.
    :return: A ``(muon_params, adamw_params)`` tuple.
    """
    muon_params: List[nn.Parameter] = []
    adamw_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        use_adamw = p.ndim < 2 or any(k in lname for k in _ADAMW_NAME_KEYWORDS)
        (adamw_params if use_adamw else muon_params).append(p)
    return muon_params, adamw_params


def build_muon_optimizer(
    model: nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    adamw_lr: float = 3e-4,
    adamw_betas: Sequence[float] = (0.9, 0.95),
    adamw_eps: float = 1e-10,
    adamw_weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Build a :class:`MuonWithAuxAdam` optimizer for ``model``.

    Automatically splits the model parameters into Muon (hidden matrices) and
    AdamW (embeddings, head, gains/biases) groups, and picks the distributed or
    single-device implementation depending on the current process group.

    :param model: The model to optimize.
    :param muon_lr: Learning rate for the Muon (hidden-matrix) group.
    :param muon_momentum: Momentum for the Muon group.
    :param muon_weight_decay: Weight decay for the Muon group.
    :param adamw_lr: Learning rate for the auxiliary AdamW group.
    :param adamw_betas: ``(beta1, beta2)`` for the AdamW group.
    :param adamw_eps: Epsilon for the AdamW group.
    :param adamw_weight_decay: Weight decay for the AdamW group.
    :return: A configured Muon-with-AdamW optimizer.
    """
    muon_params, adamw_params = split_parameters_for_muon(model)
    param_groups = [
        dict(
            params=muon_params,
            use_muon=True,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=muon_weight_decay,
        ),
        dict(
            params=adamw_params,
            use_muon=False,
            lr=adamw_lr,
            betas=tuple(adamw_betas),
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
        ),
    ]

    distributed = (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    if distributed:
        return MuonWithAuxAdam(param_groups)
    return SingleDeviceMuonWithAuxAdam(param_groups)
