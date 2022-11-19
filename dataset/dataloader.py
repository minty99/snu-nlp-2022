import glob
import os
from enum import Enum, auto

import hydra
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.utils import extract_data


class DataloaderMode(Enum):
    train = auto()
    test = auto()
    inference = auto()


def create_dataloader(cfg, mode, rank):
    dataset = Dataset_(cfg, mode)
    train_use_shuffle = True
    sampler = None
    if cfg.dist.gpus > 0 and cfg.data.divide_dataset_per_gpu:
        sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
        train_use_shuffle = False
    if mode is DataloaderMode.train:
        return DataLoader(
            dataset=dataset,
            batch_size=cfg.train.batch_size,
            shuffle=train_use_shuffle,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif mode is DataloaderMode.test:
        return DataLoader(
            dataset=dataset,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.test.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise ValueError(f"invalid dataloader mode {mode}")


class Dataset_(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        if mode is DataloaderMode.train:
            self.data_file = self.cfg.data.train_file
        elif mode is DataloaderMode.test:
            self.data_file = self.cfg.data.test_file
        else:
            raise ValueError(f"invalid dataloader mode {mode}")

        self.x, self.y = extract_data(os.path.join(self.cfg.working_dir, self.data_file))
        self.x["input_ids"] = torch.tensor(self.x["input_ids"], dtype=torch.int)
        self.x["attention_mask"] = torch.tensor(self.x["attention_mask"], dtype=torch.int)
        self.x["token_type_ids"] = torch.tensor(self.x["token_type_ids"], dtype=torch.int)

    def __len__(self):
        return len(self.x["input_ids"])

    def __getitem__(self, idx):
        return (
            self.x["input_ids"][idx],
            self.x["attention_mask"][idx],
            self.x["token_type_ids"][idx],
            self.y["start"][idx],
            self.y["end"][idx],
            self.y["qa_id"][idx],
        )
