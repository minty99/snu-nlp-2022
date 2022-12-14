import argparse
import datetime
import itertools
import os
import random
import time
import traceback

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from dataset.dataloader import DataloaderMode, create_dataloader
from model.model import Model
from model.model_arch import Net_arch
from utils.test_model import test_model
from utils.train_model import train_model
from utils.utils import get_logger, is_logging_process, set_random_seed, soft_argmax
from utils.writer import Writer


def setup(cfg, rank):
    os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.dist.master_port
    timeout_sec = 1800
    if cfg.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    # initialize the process group
    dist.init_process_group(
        cfg.dist.mode,
        rank=rank,
        world_size=cfg.dist.gpus,
        timeout=timeout,
    )


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, cfg):
    mp.spawn(fn, args=(cfg,), nprocs=cfg.dist.gpus, join=True)


def train_loop(rank, cfg):
    logger = get_logger(cfg, os.path.basename(__file__))
    if cfg.device == "cuda" and cfg.dist.gpus != 0:
        cfg.device = rank
        # turn off background generator when distributed run is on
        cfg.data.use_background_generator = False
        setup(cfg, rank)
        torch.cuda.set_device(cfg.device)

    # setup writer
    if is_logging_process():
        # set log/checkpoint dir
        os.makedirs(cfg.log.chkpt_dir, exist_ok=True)
        # set writer (tensorboard / wandb)
        writer = Writer(cfg, "tensorboard")
        cfg_str = OmegaConf.to_yaml(cfg)
        logger.info("Config:\n" + cfg_str)
        if cfg.data.train_file == "" or cfg.data.test_file == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")
        logger.info("BackgroundGenerator is turned off when Distributed running is on")

    # Sync dist processes (because of download MNIST Dataset)
    if cfg.dist.gpus != 0:
        dist.barrier()

    # make dataloader
    if is_logging_process():
        logger.info("Making train dataloader...")
    train_loader = create_dataloader(cfg, DataloaderMode.train, rank)
    if is_logging_process():
        logger.info("Making test dataloader...")
    test_loader = create_dataloader(cfg, DataloaderMode.test, rank)

    # init Model
    net_arch = Net_arch(cfg)

    def qa_loss(output, target):
        ce = torch.nn.CrossEntropyLoss()
        # [..., 0]: prob for start position prediction
        # [..., 1]: prob for end position prediction
        start = soft_argmax(output[..., 0])
        end = soft_argmax(output[..., 1])
        zero = torch.zeros(start.shape, device=start.device)
        order_const = cfg.loss.order_const

        def smooth(target, num_classes=output.shape[1]):
            t_0 = target[..., 0]  # B
            t_1 = target[..., 1]  # B
            t_0_onehot = F.one_hot(t_0, num_classes).float()  # B, num_classes
            t_1_onehot = F.one_hot(t_1, num_classes).float()  # B, num_classes
            smooth_ratio = cfg.loss.custom_smoothing
            for i, r in enumerate(smooth_ratio):
                for b in range(t_0_onehot.shape[0]):
                    if t_0[b] - i >= 0:
                        t_0_onehot[b, t_0[b] - i] = r
                    if t_1[b] + i < num_classes:
                        t_1_onehot[b, t_1[b] + i] = r
            # normalize
            t_0_onehot = t_0_onehot / torch.sum(t_0_onehot, dim=-1, keepdim=True)
            t_1_onehot = t_1_onehot / torch.sum(t_1_onehot, dim=-1, keepdim=True)
            return torch.stack([t_0_onehot, t_1_onehot], dim=-1)

        if cfg.loss.custom_smoothing:
            target = smooth(target)

        return (
            ce(output[..., 0], target[..., 0])
            + ce(output[..., 1], target[..., 1])
            + order_const * torch.mean(torch.log(torch.maximum(start - end, zero) + 1))
        ) / cfg.loss.divisor

    model = Model(cfg, net_arch, qa_loss, rank)

    # load training state / network checkpoint
    if cfg.load.resume_state_path is not None:
        model.load_training_state()
    elif cfg.load.network_chkpt_path is not None:
        model.load_network()
    else:
        if is_logging_process():
            logger.info("Starting new training run.")

    try:
        if cfg.dist.gpus == 0 or cfg.data.divide_dataset_per_gpu:
            epoch_step = 1
        else:
            epoch_step = cfg.dist.gpus
        for model.epoch in itertools.count(model.epoch + 1, epoch_step):
            if model.epoch > cfg.num_epoch:
                break
            train_model(cfg, model, train_loader, writer)
            if model.epoch % cfg.log.chkpt_interval == 0:
                model.save_network()
                model.save_training_state()
            test_model(cfg, model, test_loader, writer)
        if is_logging_process():
            logger.info("End of Train")
    except Exception as e:
        if is_logging_process():
            logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.gpus != 0:
            cleanup()


@hydra.main(version_base="1.1", config_path="config", config_name="default")
def main(hydra_cfg):
    hydra_cfg.device = hydra_cfg.device.lower()
    with open_dict(hydra_cfg):
        hydra_cfg.job_logging_cfg = HydraConfig.get().job_logging
    # random seed
    if hydra_cfg.random_seed is None:
        hydra_cfg.random_seed = random.randint(1, 10000)
    set_random_seed(hydra_cfg.random_seed)

    hydra_cfg.start_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    if hydra_cfg.dist.gpus < 0:
        hydra_cfg.dist.gpus = torch.cuda.device_count()
    if hydra_cfg.device == "cpu" or hydra_cfg.dist.gpus == 0:
        hydra_cfg.dist.gpus = 0
        train_loop(0, hydra_cfg)
    else:
        distributed_run(train_loop, hydra_cfg)


if __name__ == "__main__":
    main()
