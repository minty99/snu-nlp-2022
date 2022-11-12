import logging
import os
import random
import subprocess
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import sentencepiece as sp
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import BertModel

from utils.kobert_utils import download, get_tokenizer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_logging_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_logger(cfg, name=None):
    # log_file_path is used when unit testing
    if is_logging_process():
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg, resolve=True))
        return logging.getLogger(name)


def get_pytorch_kobert_model(ctx="cpu", cachedir=".cache"):
    def get_kobert_model(model_path, vocab_file, ctx="cpu"):
        bertmodel = BertModel.from_pretrained(model_path, return_dict=False)
        device = torch.device(ctx)
        bertmodel.to(device)
        bertmodel.eval()
        vocab = sp.SentencePieceProcessor()
        vocab.load(vocab_file)
        return bertmodel, vocab

    pytorch_kobert = {
        "url": "s3://skt-lsl-nlp-model/KoBERT/models/kobert_v1.zip",
        "chksum": "411b242919",  # 411b2429199bc04558576acdcac6d498
    }

    # download model
    model_info = pytorch_kobert
    model_path, is_cached = download(model_info["url"], model_info["chksum"], cachedir=cachedir)
    cachedir_full = os.path.expanduser(cachedir)
    zipf = ZipFile(os.path.expanduser(model_path))
    zipf.extractall(path=cachedir_full)
    model_path = os.path.join(os.path.expanduser(cachedir), "kobert_from_pretrained")
    # download vocab
    vocab_path = get_tokenizer()
    return get_kobert_model(model_path, vocab_path, ctx)
