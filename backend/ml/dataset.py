"""
Converts a feature DataFrame into (X, y) sequences for the LSTM.

X: sliding windows of LOOKBACK_DAYS worth of features
y: three targets computed HORIZON_DAYS ahead of the end of each window:
   - future_return   (regression target -> derives mean/high/low)
   - direction_class (0=down, 1=flat, 2=up -> classification target)

Critical detail for correctness: targets are computed using ONLY information
that would have been available at prediction time (no lookahead), and
train/val/test are split chronologically, never shuffled randomly -- shuffling
time series leaks future information into training.
"""

import numpy as np
import pandas as pd

from config import LOOKBACK_DAYS, HORIZON_DAYS, FLAT_THRESHOLD_PCT
from ml.features import FEATURE_COLUMNS


def make_sequences(feat_df: pd.DataFrame):
    """
    feat_df must already have FEATURE_COLUMNS + 'close' (output of build_features).
    Returns X (n, LOOKBACK_DAYS, n_features), y_return (n,), y_direction (n,)
    """
    values = feat_df[FEATURE_COLUMNS].values
    close = feat_df["close"].values

    X, y_return, y_direction = [], [], []

    n = len(feat_df)
    last_start = n - LOOKBACK_DAYS - HORIZON_DAYS
    for start in range(last_start):
        end = start + LOOKBACK_DAYS
        target_idx = end + HORIZON_DAYS - 1

        window = values[start:end]
        current_price = close[end - 1]
        future_price = close[target_idx]

        future_return = (future_price - current_price) / current_price

        if future_return * 100 > FLAT_THRESHOLD_PCT:
            direction = 2  # up
        elif future_return * 100 < -FLAT_THRESHOLD_PCT:
            direction = 0  # down
        else:
            direction = 1  # flat

        X.append(window)
        y_return.append(future_return)
        y_direction.append(direction)

    return np.array(X, dtype=np.float32), np.array(y_return, dtype=np.float32), np.array(y_direction, dtype=np.int64)


def chronological_split(X, y_return, y_direction, train_frac=0.7, val_frac=0.15):
    """No shuffling -- earliest data trains, latest data tests."""
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    splits = {
        "train": (X[:train_end], y_return[:train_end], y_direction[:train_end]),
        "val": (X[train_end:val_end], y_return[train_end:val_end], y_direction[train_end:val_end]),
        "test": (X[val_end:], y_return[val_end:], y_direction[val_end:]),
    }
    return splits
