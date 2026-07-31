"""
Trains one LSTM per supported ticker.

Run:
    python -m ml.train              # trains all 15
    python -m ml.train --ticker TSLA   # trains just one

Saves per ticker into ml/models/{ticker}/:
    model.pt        - trained weights
    scaler.json      - feature mean/std used to normalize inputs
    metrics.json      - backtested accuracy on the held-out test split,
                        including a naive baseline for comparison

The metrics.json file is what the forecast API reports as "accuracy" --
never a training-time number, always the chronological, held-out test result.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import SUPPORTED_STOCKS, MODEL_DIR, HORIZON_DAYS
from data.fetcher import get_history
from ml.features import build_features, FEATURE_COLUMNS
from ml.dataset import make_sequences, chronological_split
from ml.lstm_model import StockLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_nll_loss(mean, logvar, target):
    """Negative log-likelihood of target under N(mean, exp(logvar)).
    Lets the model express uncertainty -- wider logvar on choppier stocks --
    instead of forcing every prediction into the same fixed +/- band."""
    var = torch.exp(logvar)
    return (0.5 * torch.log(var) + 0.5 * (target - mean) ** 2 / var).mean()


def train_one_ticker(ticker: str, epochs: int = 40, batch_size: int = 32, lr: float = 1e-3):
    print(f"\n=== Training {ticker} ===")
    df = get_history(ticker)
    feat = build_features(df)
    X, y_return, y_direction = make_sequences(feat)

    if len(X) < 200:
        print(f"Not enough data for {ticker} ({len(X)} sequences) -- skipping")
        return

    splits = chronological_split(X, y_return, y_direction)
    X_train, yr_train, yd_train = splits["train"]
    X_val, yr_val, yd_val = splits["val"]
    X_test, yr_test, yd_test = splits["test"]

    # Normalize features using TRAIN stats only -- using val/test stats here
    # would leak future information into the normalization.
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

    def normalize(X):
        return (X - mean) / std

    train_ds = TensorDataset(
        torch.tensor(normalize(X_train)), torch.tensor(yr_train), torch.tensor(yd_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(normalize(X_val)).to(DEVICE)
    yr_val_t = torch.tensor(yr_val).to(DEVICE)

    model = StockLSTM(n_features=len(FEATURE_COLUMNS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yrb, ydb in train_loader:
            xb, yrb, ydb = xb.to(DEVICE), yrb.to(DEVICE), ydb.to(DEVICE)
            optimizer.zero_grad()
            mean_pred, logvar_pred, dir_logits = model(xb)
            loss = gaussian_nll_loss(mean_pred, logvar_pred, yrb) + ce_loss(dir_logits, ydb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mean_pred, logvar_pred, _ = model(X_val_t)
            val_loss = gaussian_nll_loss(mean_pred, logvar_pred, yr_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:>3} | val_loss {val_loss:.5f}")

    model.load_state_dict(best_state)
    metrics = evaluate(model, X_test, yr_test, yd_test, mean, std)
    save_model(ticker, model, mean, std, metrics)
    print(f"{ticker}: directional_accuracy={metrics['directional_accuracy']:.3f} "
          f"(baseline={metrics['baseline_directional_accuracy']:.3f}) mae_pct={metrics['mae_pct']:.2f}")


def evaluate(model, X_test, yr_test, yd_test, mean, std):
    model.eval()
    X_norm = (X_test - mean) / std
    with torch.no_grad():
        mean_pred, logvar_pred, dir_logits = model(torch.tensor(X_norm).to(DEVICE))
        mean_pred = mean_pred.cpu().numpy()
        dir_pred = dir_logits.argmax(dim=1).cpu().numpy()

    directional_accuracy = float((dir_pred == yd_test).mean())
    mae_pct = float(np.mean(np.abs(mean_pred - yr_test)) * 100)

    # Naive baseline: "next week's direction = flat" is the naive persistence
    # call. Reporting this alongside the model's number keeps the accuracy claim honest.
    baseline_pred = np.ones_like(yd_test)  # always predict "flat" (class 1)
    baseline_directional_accuracy = float((baseline_pred == yd_test).mean())

    return {
        "directional_accuracy": directional_accuracy,
        "baseline_directional_accuracy": baseline_directional_accuracy,
        "mae_pct": mae_pct,
        "horizon_days": HORIZON_DAYS,
        "test_samples": int(len(yd_test)),
    }


def save_model(ticker, model, mean, std, metrics):
    out_dir = os.path.join(MODEL_DIR, ticker)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="Train a single ticker, or omit to train all 15")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else list(SUPPORTED_STOCKS.keys())
    for t in tickers:
        try:
            train_one_ticker(t)
            print(f"{t} Trained")
        except Exception as e:
            print(f"Failed on {t}: {e}")
