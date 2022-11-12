# coding=utf-8
# Copyright 2019 SK T-Brain Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from zipfile import ZipFile

import sentencepiece as sp
import torch
from transformers import BertModel

from utils.utils import get_pytorch_kobert_model


def main():
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    model, vocab = get_pytorch_kobert_model()
    sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)
    print(pooled_output.shape)
    print(vocab)
    print(sequence_output[0])


if __name__ == "__main__":
    main()
