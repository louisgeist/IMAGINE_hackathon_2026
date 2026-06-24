
<div align="center">

# IMAGINE Hackathon 2026
*How much energy does it really take to train a competent vision model?*
</div>

In this repo, we train a **ViT-S/16** for **image classification** on the **ImageNet-1k** dataset.

The goal is to achieve **85% top-5 accuracy** on our test set using as little energy as possible.

Our baseline trains for under 7 hours on a single A6000 GPU, consuming around 2630 watt-hours. The baseline is already partially optimized with mixed precision and model compilation, but we can for sure do better!

### Index
- [Before you Start](#before-you-start)
    - [Dataset](#dataset)
    - [uv](#uv)
    - [CodeCarbon](#codecarbon)
    - [Electricity Maps](#electricity-maps)
    - [Weights & Biases](#weights--biases)
- [Project Structure](#project-structure)
- [Defining Experiments](#defining-experiments)
    - [Experiment Tagging](#experiment-tagging)
    - [Hyperparameter Search](#hyperparameter-search)
- [Training](#training)
- [Evaluation](#evaluation)

## Before you Start
We need to set up some things before we start training. First clone the repo and `cd` into it, then do the following:

### Dataset
To download our split of ImageNet-1k, run `download_imagenet.sh`:
```bash
./download_imagenet.sh
```
The download should take around 20 minutes. After it finishes, you should find the `train` and `val` partitions under the `data/` folder.

### uv
We are going to use the uv package manager, ask your teammates for help with the setup if needed.

Before moving on to the next steps, it is a good idea to get the environment ready:
```bash
uv pip install -r pyproject.toml
```

### CodeCarbon
We are going to use CodeCarbon to measure the energy consumption and carbon emissions of our training runs.
CodeCarbon has an API that we can use to upload all the measurements to their dashboard, but to do this we all need to have an account and that CodeCarbon can authenticate to the API. Step by step:
1) Go to https://dashboard.codecarbon.io/ > `Sign in or create an account`
2) Click `Register` at the bottom, fill and submit the form with your ENPC/uni email
3) Send me (Marta López) your email address on Slack so that I can add you to our organization
4) On your machine, install and configure CodeCarbon
    > Note: CodeCarbon will open a browser window when you log in; make sure your terminal supports this (the VSCode terminal is a safe bet)
    ```bash
    uv run codecarbon login  # Will open a browser window to authenticate you
    uv run codecarbon config  # Select: organization -> IMAGINE; project -> IMAGINE Hackathon 2026; experiment -> Baseline
    ```
5) To get CPU power consumption measurements, run:
    ```bash
    sudo chmod -R a+r /sys/class/powercap/intel-rapl
    ```

### Electricity Maps
To get more accurate carbon emission measurements, CodeCarbon can use the ElectricityMaps API. For this to work, follow these steps:
1) Go to https://app.electricitymaps.com/auth/signup
2) Enter your ENPC/uni email and click `Sign up`, then follow the instruction to complete your registration
3) Scroll down to the `Resources` section > `API Key`
4) Create a new API key and copy it to your clipboard
5) Write the plain API key to `electricity_maps_key.txt` in the root directory of the repo:
    ```bash
    echo <your API key> >> ./electricity_maps_key.txt
    ```

### Weights & Biases
Make sure you are logged in to Weights & Biases before launching experiments!
```bash
uv run wandb login
```

## Project Structure
This repo is based on the [Lightning + Hydra template by ashleve](https://github.com/ashleve/lightning-hydra-template).

There are three main components that you can play around with:
- The [`datamodule`](./configs/datamodule/) reads the raw data, applies data augmentation, collates batches, and sends them to the GPU.
- The [`module`](./configs/module/) defines the network, schedulers, and optimizers and controls what happens in each training step.
- The [`trainer`](./configs/trainer/) configures some global training options, such as the total number of epochs, gradient clipping settings, or the use of mixed precision.

So you will be working mainly with those three configs, as well as [`experiment`](./configs/experiment/), and maybe [`hparams_search`](./configs/hparams_search) and [`callbacks`](./configs/callbacks/).

## Defining Experiments
To define a new experiment, create a YAML file under [`configs/experiment`](./configs/experiment).

If needed, override the `module`, `datamodule` or `trainer` configs as done in the [example](configs/experiment/example.yaml) with your own config files defined in the `module`, `datamodule` or `trainer` directories under `./configs`, respectively.

If needed, you may also define new versions of [`imagenet_datamodule.py`](./src/datamodules/imagenet_datamodule.py) and [`imagenet_module.py`](./src/modules/imagenet_module.py) in the corresponding directories under `src`.

### Experiment Tagging
To keep things tidy, please define the following in the [default train config](./configs/train.yaml).
- `team_name`: name of your team. Can be a single letter, the team leader's name, or a short name, as long as it is the same for all team members.
- `experiment_name`: name of the approach you are trying out. The logs are timestamped, so running the same experiment several times will not overwrite previous output.
- `tags`: a *project stage* and a *run type* tag. The available options are:
    - First tag options: `data`, `explore`, `baseline`, `ablate`, `hyperparam`, `final`, `<custom tag>`
    - Second tag options: `train`, `pre-train`, `post-train`, `debug`, `<custom tag>`
Adapt `experiment_name` and `tags` as necessary for each experiment.

### Hyperparameter Search
You can use the Hydra `--multirun` option to launch a simple, *sequential* grid search as shown in the [documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/). For example:

```bash
uv run src/train.py --multirun datamodule.batch_size=32,64,128 module.optimizer.lr="range(0.01,0.06,0.01)" tags="[hyperparam,train]"
```

You can also try Optuna for a smarter hyperparameter search. There is an example config in [optuna.yaml](./configs/hparams_search/optuna.yaml), you can use it by indicating `hparams_search: optuna` in your experiment config.

> Don't forget to use the `hyperparam` tag in the tags field of the config!

## Training
Run:
```bash
uv run src/train.py
```
with any Hydra command-line overrides that you need for your experiment.

## Evaluation
Once the test set has been released, complete the [eval config](./configs/eval.yaml) with your selected checkpoints and your team name, then run:
```bash
uv run src/eval.py
```
This will register the output of the model for each image in the test set and upload the results to the evaluation server. We will only reveal the test performance after everyone has submitted their results.
