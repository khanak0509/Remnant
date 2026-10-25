import torch
from torch import nn

from config import WINDOW


class CNN(nn.Module):
    def __init__(self, n_features, out=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * WINDOW, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, out))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)))


class LSTM(nn.Module):
    def __init__(self, n_features, out=1, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, out))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1])


class Transformer(nn.Module):
    def __init__(self, n_features, out=1, d=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d)
        self.pos = nn.Parameter(torch.randn(1, WINDOW, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, out))

    def forward(self, x):
        h = self.encoder(self.proj(x) + self.pos)
        return self.head(h.mean(dim=1))


MODELS = {"cnn": CNN, "lstm": LSTM, "transformer": Transformer}
