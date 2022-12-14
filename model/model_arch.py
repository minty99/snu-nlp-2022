import torch.nn as nn
import torch.nn.functional as F

from utils.utils import get_multilingual_bert_model


class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, cfg):
        super(Net_arch, self).__init__()
        self.cfg = cfg
        self.bert = get_multilingual_bert_model(ctx=cfg.device)
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        # n_words = 512 (max)
        x = self.bert(*x).last_hidden_state  # [N, n_words, 768]
        x = self.fc(x)  # [N, n_words, 2]

        return x  # [N, n_words, 2]
