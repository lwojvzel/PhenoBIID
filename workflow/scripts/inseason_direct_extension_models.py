"""Additional fixed-family controls on the existing seasonal input interface."""
import torch
from torch import nn


class SeasonalCNNHistoryLSTM(nn.Module):
    def __init__(self, sequence_features, static_features):
        super().__init__()
        self.convolutions = nn.ModuleList([
            nn.Conv1d(sequence_features, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1)])
        self.dropout = nn.Dropout(.1)
        self.history = nn.LSTM(2, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128+64+static_features, 128),
            nn.GELU(), nn.Dropout(.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, sequence, static, valid):
        mask = valid[:, None, :]
        values = sequence.transpose(1, 2)*mask
        for layer in self.convolutions:
            values = self.dropout(torch.relu(layer(values)))*mask
        seasonal = values.sum(2)/mask.sum(2).clamp_min(1)
        # The shared history encoder stores the most recent lag first.
        lags = torch.stack((static[:, :5].flip(1), static[:, 5:10].flip(1)), -1)
        _, (hidden, _) = self.history(lags)
        return self.head(torch.cat((seasonal, hidden[-1], static), 1)).squeeze(1)
