import torch.nn as nn
import torch.nn.functional as F

from kobert import get_pytorch_kobert_model


class Net_arch(nn.Module):
    # Network architecture
    def __init__(self, cfg):
        super(Net_arch, self).__init__()
        self.cfg = cfg
        self.bert, _ = get_pytorch_kobert_model(ctx=cfg.device)
        self.fc1 = nn.Linear(768, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True
        for param in self.fc3.parameters():
            param.requires_grad = True

    def forward(self, x):
        # n_words = 512 (max)
        x, _ = self.bert(*x)
        # x = F.dropout(x, 0.5)
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, 0.5)
        x = self.fc2(x)
        x = F.relu(x)
        # x = F.dropout(x, 0.5)
        x = self.fc3(x)
        x = F.relu(x)
        # x = F.dropout(x, 0.5)
        x = self.fc4(x)
        x = F.relu(x)
        # x = F.dropout(x, 0.5)
        x = self.fc5(x)
        return x  # [N, n_words, 2]
