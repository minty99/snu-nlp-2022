import json
import logging
import os
import pickle as pk
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import BertTokenizerFast, BertModel

from utils.evaluate import evaluate


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


def get_multilingual_bert_model(ctx="cpu"):
    model = BertModel.from_pretrained("bert-base-multilingual-cased").to(ctx)
    return model


def get_multilingual_bert_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    return tokenizer


def get_multilingual_bert(ctx="cpu"):
    return get_multilingual_bert_model(ctx), get_multilingual_bert_tokenizer()


def soft_argmax(x: torch.Tensor, beta=1e10):
    x_range = torch.arange(0, x.shape[-1]).to(x.device)
    return torch.sum(torch.softmax(x * beta, dim=-1) * x_range, dim=-1)


def extract_data(filename, drop_long=True):
    if os.path.exists(f".cache/{os.path.basename(filename)}.pk"):
        with open(f".cache/{os.path.basename(filename)}.pk", "rb") as cache_file:
            x, y = pk.load(cache_file)
            return x, y

    x = defaultdict(list)
    y = defaultdict(list)
    tokenizer = get_multilingual_bert_tokenizer()
    pad = tokenizer.convert_tokens_to_ids("[PAD]")
    with open(filename, "r") as f:
        data = json.load(f)
        data = data["data"]

    long_count = 0
    dropped_count = 0
    label_changed_count = 0
    total_count = 0

    sep_token = tokenizer.convert_tokens_to_ids("[SEP]")
    for d in data:
        for pair in d["paragraphs"]:
            context = pair["context"]
            for qa in pair["qas"]:
                qa_id = qa["id"]
                question = qa["question"]
                answer_txt = qa["answers"][0]["text"]
                answer_start = qa["answers"][0]["answer_start"]

                # get encoded input txt
                input_str = f"[CLS] {question} [SEP] {context}"
                input_encoded = tokenizer.encode(input_str, add_special_tokens=False)
                attention_mask = [1] * len(input_encoded)
                total_count += 1
                if len(input_encoded) > 511:
                    # print("WARNING: input string token is over 512")
                    # print(f"\t{input_str}")
                    input_encoded = input_encoded[:511]
                    attention_mask = attention_mask[:511]
                    input_encoded.append(sep_token)  # add [SEP] at the end
                    attention_mask.append(1)
                    long_count += 1
                else:
                    input_encoded.append(sep_token)  # add [SEP] at the end
                    attention_mask.append(1)
                    input_encoded += [pad] * (512 - len(input_encoded))
                    attention_mask += [0] * (512 - len(attention_mask))

                first_sep_idx = input_encoded.index(sep_token)
                token_type_ids = [0] * (first_sep_idx + 1) + [1] * (512 - (first_sep_idx + 1))

                # get target start end pos
                offsets = tokenizer(input_str, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                answer_start += input_str.find("[SEP]") + 6
                answer_end = answer_start + len(answer_txt) - 1
                ret_answer_start = -1
                ret_answer_end = -1
                for idx, p in enumerate(offsets):
                    begin, end = p
                    if ret_answer_start == -1 and begin <= answer_start < end:
                        ret_answer_start = idx
                    if ret_answer_end == -1 and begin <= answer_end < end:
                        ret_answer_end = idx
                    elif ret_answer_end == -1 and begin > answer_end:
                        ret_answer_end = idx - 1

                if ret_answer_end >= 512 and drop_long:
                    # Do not add long texts which have answers after 512 into dataset
                    dropped_count += 1
                    continue

                # Add long texts, but change label into (511, 511) which means no answer in given paragraph
                # This can make the test loss errorneous
                if ret_answer_start >= 512:
                    ret_answer_start = 511
                if ret_answer_end >= 512:
                    label_changed_count += 1
                    ret_answer_end = 511

                # input: [input_ids, attention_mask, token_type_ids]
                # target: [start_pos, end_pos, qa_id]
                x["input_ids"].append(input_encoded)
                x["attention_mask"].append(attention_mask)
                x["token_type_ids"].append(token_type_ids)
                y["start"].append(ret_answer_start)
                y["end"].append(ret_answer_end)
                y["qa_id"].append(qa_id)

    print(f"WARNING: {long_count}/{total_count} elements exceed 512 ({long_count / total_count * 100:.4f} %)")
    if drop_long:
        print(f"WARNING: {dropped_count}/{total_count} elements dropped ({dropped_count / total_count * 100:.4f} %)")
    if not drop_long:
        print(
            f"WARNING: {label_changed_count}/{total_count} elements' label changed ({label_changed_count / total_count * 100:.4f} %)"
        )

    os.makedirs(".cache", exist_ok=True)
    with open(f".cache/{os.path.basename(filename)}.pk", "wb") as cache_file:
        pk.dump((x, y), cache_file)
    return x, y


def calc_metric(cfg, predictions):
    filename = cfg.data.test_file
    expected_version = "KorQuAD_v1.0"
    with open(filename) as dataset_file:
        dataset_json = json.load(dataset_file)
        read_version = "_".join(dataset_json["version"].split("_")[:-1])
        if read_version != expected_version:
            print("Evaluation expects " + expected_version + ", but got dataset with " + read_version, file=sys.stderr)
        dataset = dataset_json["data"]
    return evaluate(dataset, predictions)
