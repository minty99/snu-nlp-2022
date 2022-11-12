import torch.nn as nn
import torch.nn.functional as F

from kobert import get_pytorch_kobert_model


class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, cfg):
        super(Net_arch, self).__init__()
        self.cfg = cfg
        self.bert, _ = get_pytorch_kobert_model(ctx=cfg.device)
        self.fc = nn.Linear(768, 2)
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        # n_words = 512 (max)
        x, _ = self.bert(*x)  # [N, n_words, 768]
        F.dropout(x, 0.5, training=self.training, inplace=True)
        x = self.fc(x)  # [N, n_words, 768] -> [N, n_words, 2]

        return x  # [N, n_words, 2]
