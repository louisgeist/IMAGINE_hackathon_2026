from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np
import nltk
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from src.modules.imagenet_module import ImageNetModule

MatrixMetric = Literal["path", "wup"]
Distribution = Literal["normalized", "softmax"]

MATRIX_FILENAME = "semantic_label_smoothing_matrix.pt"


def ensure_wordnet() -> None:
    """Download WordNet corpora if they are not already available."""
    for package in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package)


def load_wnids_from_imagefolder(train_dir: str | Path) -> list[str]:
    """Return ImageNet WNIDs in the same order as `ImageFolder` class indices.

    `torchvision.datasets.ImageFolder` sorts class folder names alphabetically,
    so the returned list is sorted to match model output indices.
    """
    train_root = Path(train_dir)
    if not train_root.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_root}")

    wnids = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if not wnids:
        raise ValueError(f"No class folders found under {train_root}")

    return wnids


def load_wnids_from_class_index(class_index_json_path: str | Path) -> list[str]:
    """Return ImageNet WNIDs ordered by the standard class-index JSON mapping."""
    with open(class_index_json_path, "r", encoding="utf-8") as file:
        class_idx = json.load(file)

    sorted_indices = sorted(class_idx.keys(), key=int)
    return [class_idx[index][0] for index in sorted_indices]


def load_imagenet_synsets(wnids: Sequence[str]) -> list[Any]:
    """Convert ImageNet WNIDs into NLTK synsets in the same order."""
    synsets = []
    for wnid in wnids:
        pos = wnid[0]
        offset = int(wnid[1:])
        try:
            synsets.append(wn.synset_from_pos_and_offset(pos, offset))
        except Exception:
            print(f"Warning: Could not parse synset for {wnid}. Using None fallback.")
            synsets.append(None)
    return synsets


def compute_similarity_matrix(
    synsets: Sequence[Any],
    metric: MatrixMetric = "path",
) -> np.ndarray:
    """Compute a symmetric WordNet similarity matrix for ImageNet classes."""
    num_classes = len(synsets)
    similarity = np.zeros((num_classes, num_classes), dtype=np.float32)

    print(f"Computing WordNet {metric} similarities for {num_classes} classes...")
    for i in tqdm(range(num_classes)):
        syn_i = synsets[i]
        if syn_i is None:
            continue

        similarity[i, i] = 0.0
        for j in range(i + 1, num_classes):
            syn_j = synsets[j]
            if syn_j is None:
                continue

            if metric == "path":
                value = syn_i.path_similarity(syn_j) or 0.0
            elif metric == "wup":
                value = syn_i.wup_similarity(syn_j) or 0.0
            else:
                raise ValueError(f"Unknown similarity metric: {metric}")

            similarity[i, j] = value
            similarity[j, i] = value

    return similarity


def build_smoothing_matrix(
    similarity: np.ndarray,
    epsilon: float,
    distribution: Distribution = "normalized",
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert a similarity matrix into per-class smoothed target distributions."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

    num_classes = similarity.shape[0]
    smoothing = np.zeros((num_classes, num_classes), dtype=np.float32)

    for class_idx in range(num_classes):
        row = similarity[class_idx].astype(np.float64, copy=True)
        row[class_idx] = 0.0

        if distribution == "normalized":
            total = row.sum()
            if total > 0.0:
                off_diagonal = row / total * epsilon
            else:
                off_diagonal = np.full(num_classes, epsilon / max(num_classes - 1, 1))
                off_diagonal[class_idx] = 0.0
        elif distribution == "softmax":
            scaled = row / temperature
            finite = np.isfinite(scaled)
            if finite.any():
                row_max = scaled[finite].max()
                exp_row = np.where(finite, np.exp(scaled - row_max), 0.0)
                exp_row /= exp_row.sum()
                off_diagonal = exp_row * epsilon
            else:
                off_diagonal = np.full(num_classes, epsilon / max(num_classes - 1, 1))
                off_diagonal[class_idx] = 0.0
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        smoothing[class_idx] = off_diagonal.astype(np.float32)
        smoothing[class_idx, class_idx] = 1.0 - epsilon

    return smoothing


def get_or_compute_similarity_matrix(
    matrix_path: str | Path,
    train_dir: Optional[str | Path] = None,
    class_index_json_path: Optional[str | Path] = None,
    metric: MatrixMetric = "path",
) -> tuple[torch.Tensor, list[str]]:
    """Load a cached similarity matrix or compute and save it if missing."""
    matrix_path = Path(matrix_path)
    train_dir = Path(train_dir) if train_dir is not None else None

    if matrix_path.is_file():
        payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
        wnids = list(payload["wnids"])
        cached_metric = payload.get("metric", "path")
        if cached_metric != metric:
            raise ValueError(
                f"Cached matrix at {matrix_path} uses metric={cached_metric!r}, "
                f"but metric={metric!r} was requested."
            )

        if train_dir is not None:
            current_wnids = load_wnids_from_imagefolder(train_dir)
            if current_wnids != wnids:
                raise ValueError(
                    "Cached semantic similarity matrix does not match the current "
                    f"ImageFolder classes in {train_dir}."
                )

        return payload["similarity"].to(dtype=torch.float32), wnids

    if train_dir is None and class_index_json_path is None:
        raise ValueError(
            "Matrix cache not found. Provide train_dir or class_index_json_path "
            "so the similarity matrix can be computed."
        )

    ensure_wordnet()
    if train_dir is not None:
        wnids = load_wnids_from_imagefolder(train_dir)
    else:
        wnids = load_wnids_from_class_index(class_index_json_path)

    synsets = load_imagenet_synsets(wnids)
    similarity = compute_similarity_matrix(synsets, metric=metric)
    similarity_tensor = torch.from_numpy(similarity)

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "similarity": similarity_tensor,
            "wnids": wnids,
            "metric": metric,
        },
        matrix_path,
    )
    print(f"Semantic similarity matrix saved to {matrix_path}")
    return similarity_tensor, wnids


class SemanticLabelSmoothing:
    r"""Turns hard (or soft) targets into semantically smoothed targets.

    Unlike uniform label smoothing, probability mass on incorrect classes is
    spread according to WordNet similarity to the true class.

    Smoothing is applied only within the epoch window ``[start_epoch, end_epoch)``,
    using the same absolute/fractional epoch semantics as `LabelSmoothing`.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        num_classes: int = 1000,
        start_epoch: float = 0.0,
        end_epoch: float = 1.0,
        matrix_path: str | Path = "data/semantic_label_smoothing_matrix.pt",
        train_dir: Optional[str | Path] = None,
        class_index_json_path: Optional[str | Path] = None,
        metric: MatrixMetric = "path",
        distribution: Distribution = "normalized",
        temperature: float = 1.0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha
        self.num_classes = num_classes
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.matrix_path = Path(matrix_path)
        self.train_dir = Path(train_dir) if train_dir is not None else None
        self.class_index_json_path = (
            Path(class_index_json_path) if class_index_json_path is not None else None
        )
        self.metric = metric
        self.distribution = distribution
        self.temperature = temperature

        similarity, self.wnids = get_or_compute_similarity_matrix(
            matrix_path=self.matrix_path,
            train_dir=self.train_dir,
            class_index_json_path=self.class_index_json_path,
            metric=self.metric,
        )
        if similarity.shape[0] != num_classes:
            raise ValueError(
                f"Expected {num_classes} classes, got {similarity.shape[0]} "
                f"from {self.matrix_path}"
            )

        self._similarity = similarity

    @staticmethod
    def _resolve_epoch(value: float, max_epochs: Optional[int]) -> float:
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            if max_epochs is None:
                raise ValueError(
                    "Fractional epoch bounds require max_epochs to be provided."
                )
            return value * max_epochs
        return value

    def is_active(self, epoch: int, max_epochs: Optional[int] = None) -> bool:
        if self.alpha == 0.0:
            return False

        start = self._resolve_epoch(self.start_epoch, max_epochs)
        end = self._resolve_epoch(self.end_epoch, max_epochs)
        return start <= epoch < end

    def current_alpha(self, epoch: int, max_epochs: Optional[int] = None) -> float:
        return self.alpha if self.is_active(epoch, max_epochs) else 0.0

    def _smoothing_matrix(self, alpha: float, device: torch.device) -> torch.Tensor:
        smoothing = build_smoothing_matrix(
            self._similarity.cpu().numpy(),
            epsilon=alpha,
            distribution=self.distribution,
            temperature=self.temperature,
        )
        return torch.as_tensor(smoothing, dtype=torch.get_default_dtype(), device=device)

    def __call__(
        self,
        targets: torch.Tensor,
        epoch: int = 0,
        max_epochs: Optional[int] = None,
    ) -> torch.Tensor:
        alpha = self.current_alpha(epoch, max_epochs)
        if alpha == 0.0:
            return targets

        smoothing_matrix = self._smoothing_matrix(alpha, targets.device)

        if targets.dim() == 1:
            return smoothing_matrix[targets.long()]

        targets = targets.to(dtype=torch.get_default_dtype())
        return targets @ smoothing_matrix


class SemanticLabelSmoothingImageNetModule(ImageNetModule):
    """`ImageNetModule` variant that applies semantic label smoothing."""

    def __init__(
        self,
        *args: Any,
        semantic_label_smoothing: Optional[SemanticLabelSmoothing] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.semantic_label_smoothing = semantic_label_smoothing

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)

        target = y
        if self.training and self.semantic_label_smoothing is not None:
            max_epochs = self.trainer.max_epochs if self.trainer is not None else None
            target = self.semantic_label_smoothing(
                y, epoch=self.current_epoch, max_epochs=max_epochs
            )

        loss = self.criterion(logits, target)
        if y.dim() > 1:
            y = y.argmax(dim=1)
        return loss, logits, y.long()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute and cache the semantic label smoothing similarity matrix."
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=Path("data") / MATRIX_FILENAME,
        help="Path where the cached matrix will be stored.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="ImageFolder training directory used to infer class WNIDs.",
    )
    parser.add_argument(
        "--class-index-json",
        type=Path,
        default=None,
        help="Optional ImageNet class-index JSON used when train-dir is not set.",
    )
    parser.add_argument(
        "--metric",
        choices=("path", "wup"),
        default="path",
        help="WordNet similarity metric.",
    )
    args = parser.parse_args()

    similarity, wnids = get_or_compute_similarity_matrix(
        matrix_path=args.matrix_path,
        train_dir=args.train_dir,
        class_index_json_path=args.class_index_json,
        metric=args.metric,
    )
    print(f"Matrix shape: {tuple(similarity.shape)}")
    print(f"Number of classes: {len(wnids)}")
