import torch.nn as nn
import torch.nn.functional as F

from kobert import get_pytorch_kobert_model


class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, cfg):
        super(Net_arch, self).__init__()
        self.cfg = cfg
        self.bert, _ = get_pytorch_kobert_model(ctx=cfg.device)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        # for param in self.parameters():
        #     param.requires_grad = False
        # for param in self.fc1.parameters():
        #     param.requires_grad = True
        # for param in self.fc2.parameters():
        #     param.requires_grad = True
        # for param in self.fc3.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        # n_words = 512 (max)
        x, _ = self.bert(*x)  # [N, n_words, 768]
        # x = F.dropout(x, 0.5)
        x = self.fc1(x)  # [N, n_words, 768] -> [N, n_words, 256]
        x = F.relu(x)
        # x = F.dropout(x, 0.5)
        x = self.fc2(x)  # [N, n_words, 256] -> [N, n_words, 64]
        x = F.relu(x)
        # x = F.dropout(x, 0.5)
        x = self.fc3(x)  # [N, n_words, 64] -> [N, n_words, 2]
        return x  # [N, n_words, 2]
