<div align="center">
    <h1><code>
        2022-Fall SNU NLP Final Project
    </h1></code>
</div>

## Introduction

- Finetuning Google's multilingual BERT to Korean.
- [wandb](https://wandb.ai/), [poetry](https://github.com/python-poetry/poetry), [hydra](https://hydra.cc/), [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) is used with [PyTorch](https://pytorch.org/)

## Code Structure

- `config` dir: directory for config files
- `dataset` dir: dataloader and dataset codes are here.
- `model` dir: `model.py` is for wrapping network architecture. `model_arch.py` is for coding network architecture.
- `utils` dir:
    - `train_model.py` and `test_model.py` are for train and test model once.
    - `utils.py` is for utility. random seed setting, get BERT Model, soft argmax, etc.
    - `writer.py` is for writing logs in tensorboard / wandb.
- `trainer.py` file: this is for setting up and iterating epoch.

## Setup

### Install requirements

- python3
- Using poetry for package manager
- `poety install`

### Config

- Config is written in yaml file using hydra.
- `name` is train name you run.
- `num_epoch` is for end iteration step of training.
- `scheduler` in `train`
    - learning rate decay starts from `max_lr` to `min_lr` through epochs
- `loss` field
    - `divisor`: divider value of some of loss values
    - `custom_smoothing`: list of decreasing values which is used for smoohted target values. This value is applied forward for 'start' target value and backward for 'end' target value
    - `order_const`: coefficient of ordering loss
        - order_const * mean(log(max(start_pred - end_pred, 0) + 1))

### Code lint

1. `poetry install` for install develop dependencies

1. `pre-commit install` for adding pre-commit to git hook

## Train

- `python trainer.py working_dir=$(pwd)`

## Reference

- https://github.com/ryul99/pytorch-project-template
    - Code Template / One of our team member is author