<div align="center">
    <h1><code>
        2022-Fall SNU NLP Final Project
    </h1></code>
</div>

## Introduction

- Fine-tuning Google's multilingual BERT for Korean question answering task.
- [wandb](https://wandb.ai/), [poetry](https://github.com/python-poetry/poetry), [hydra](https://hydra.cc/), [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) is used with [PyTorch](https://pytorch.org/)

## Code Structure

- `config` dir: directory for config files.
- `dataset` dir: dataloader and dataset codes are here.
  - `meta` dir: actual dataset JSON file (Downloaded from https://korquad.github.io/KorQuad%201.0/)
- `model` dir
  - `model.py` is for wrapping network architecture.
  - `model_arch.py` is for coding network architecture.
- `utils` dir:
    - `train_model.py` and `test_model.py` are for train and test model once.
    - `utils.py` is for utility. random seed setting, get BERT Model, soft argmax, etc.
    - `writer.py` is for writing logs in tensorboard / wandb.
    - `evaluate.py` is the official evaluation script from KorQuAD 1.0 dataset authors. (Downloaded from https://korquad.github.io/KorQuad%201.0/)
- `trainer.py` file: this is for setting up and iterating epoch.

## Setup

### Install requirements

- Python 3 (We tested on Python 3.10)
- Install `poetry` for Python package manager
- Run `poetry install`

### Config

- Config is written in yaml file using hydra.
- `name` is train session name you run.
- `num_epoch` is for end iteration step of training.
- `scheduler` in `train`
    - learning rate decay starts from `max_lr` to `min_lr` linearly through epochs
- `loss` field
    - `divisor`: divisor of sum of loss values
    - `custom_smoothing`: list of decreasing values which is used for smoohted target values. This value is applied before 'start' target value and after 'end' target value
    - `order_const`: scale constant of ordering constraint loss
        - order_const * mean(log(max(start_pred - end_pred, 0) + 1))

### Code lint

1. `poetry install` for install develop dependencies

1. `pre-commit install` for adding pre-commit to git hook

## Train

- `python trainer.py working_dir=$(pwd)`

## Reference

- https://github.com/ryul99/pytorch-project-template
    - Code Template / Authored by team member (Changmin Choi)
