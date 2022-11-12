import json
import logging
import os
import random
import subprocess
from collections import defaultdict
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


def extract_data(filename):
    x = defaultdict(list)
    y = defaultdict(list)
    _, vocab = get_pytorch_kobert_model()
    pad = vocab.piece_to_id("[PAD]")
    with open(filename, "r") as f:
        data = json.load(f)
        data = data["data"]

    exceed_count = 0
    total_count = 0
    for d in data:
        for pair in d["paragraphs"]:
            if total_count > 100:
                break  # TODO: for debug
            context = pair["context"]
            for qa in pair["qas"]:
                qa_id = qa["id"]
                question = qa["question"]
                answer_txt = qa["answers"][0]["text"]
                answer_start = qa["answers"][0]["answer_start"]

                # get encoded input txt
                input_str = f"[CLS] {question} [SEP] {context}"
                input_encoded = vocab.encode(input_str, out_type=int)
                attention_mask = [1] * len(input_encoded)
                total_count += 1
                if len(input_encoded) > 512:
                    # print("WARNING: input string token is over 512")
                    # print(f"\t{input_str}")
                    input_encoded = input_encoded[:512]
                    attention_mask = attention_mask[:512]
                    exceed_count += 1
                else:
                    input_encoded += [pad] * (512 - len(input_encoded))
                    attention_mask += [0] * (512 - len(attention_mask))

                first_sep_idx = input_encoded.index(vocab.piece_to_id("[SEP]"))
                token_type_ids = [0] * (first_sep_idx + 1) + [1] * (512 - (first_sep_idx + 1))

                # get target start end pos
                proto = vocab.encode_as_immutable_proto(input_str)
                answer_start += 5
                answer_end = answer_start + len(answer_txt) - 1
                ret_answer_start = -1
                ret_answer_end = -1
                for idx, p in enumerate(proto.pieces):
                    if p.begin <= answer_start < p.end:
                        ret_answer_start = idx
                    if p.begin <= answer_end < p.end:
                        ret_answer_end = idx

                # input: [input_ids, attention_mask, token_type_ids]
                # target: [start_pos, end_pos, qa_id]
                x["input_ids"].append(input_encoded)
                x["attention_mask"].append(attention_mask)
                x["token_type_ids"].append(token_type_ids)
                y["start"].append(ret_answer_start)
                y["end"].append(ret_answer_end)
                y["qa_id"].append(qa_id)

    print(f"WARNING: {exceed_count}/{total_count} elements exceed 512 ({exceed_count / total_count * 100:.4f} %)")
    return x, y
