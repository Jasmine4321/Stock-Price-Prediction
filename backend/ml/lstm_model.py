"""
LSTM with two output heads sharing one encoder:
  - regression head  -> predicted future return (used to derive mean/high/low)
  - classification head -> P(down), P(flat), P(up)

Sharing the encoder means the model learns one representation of "what's
happening in this window" and both heads read off it -- fewer parameters
than training two separate models, which matters when each ticker only has
a few years of daily data to train on.
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Regression head: predicts mean future return, plus log-variance so we
        # can derive a high/low band from a Gaussian around the mean instead of
        # guessing a fixed +/- % range.
        self.return_mean = nn.Linear(hidden_size, 1)
        self.return_logvar = nn.Linear(hidden_size, 1)

        # Classification head: 3 classes = down, flat, up
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        # x: (batch, lookback_days, n_features)
        _, (h_n, _) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])  # (batch, hidden_size)

        mean = self.return_mean(last_hidden).squeeze(-1)
        logvar = self.return_logvar(last_hidden).squeeze(-1)
        direction_logits = self.direction_head(last_hidden)

        return mean, logvar, direction_logits
