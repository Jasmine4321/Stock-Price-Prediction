"""
Trains one LSTM per (ticker, horizon) combination -- 15 stocks x 4 horizons
= 60 models by default. Each horizon gets its own model because the
relationship between a 60-day window and "tomorrow" is very different from
the relationship between that same window and "one year from now".

Run:
    python -m ml.train                          # trains all tickers, all horizons
    python -m ml.train --ticker TSLA             # one ticker, all horizons
    python -m ml.train --ticker TSLA --horizon 7d  # one ticker, one horizon

Saves per (ticker, horizon) into ml/models/{ticker}/{horizon}/:
    model.pt, scaler.json, metrics.json

metrics.json is what the API reports as "accuracy" -- always the
chronological, held-out test result, never a training-time number.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import SUPPORTED_STOCKS, MODEL_DIR, HORIZONS
from data.fetcher import get_history
from ml.features import build_features, FEATURE_COLUMNS
from ml.dataset import make_sequences, chronological_split
from ml.lstm_model import StockLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_nll_loss(mean, logvar, target):
    var = torch.exp(logvar)
    return (0.5 * torch.log(var) + 0.5 * (target - mean) ** 2 / var).mean()


def train_one(ticker: str, horizon_key: str, epochs: int = 40, batch_size: int = 32, lr: float = 1e-3):
    horizon_days = HORIZONS[horizon_key]["days"]
    flat_pct = HORIZONS[horizon_key]["flat_threshold_pct"]

    print(f"\n=== Training {ticker} [{horizon_key}] ===")
    df = get_history(ticker)
    feat = build_features(df)
    X, y_return, y_direction = make_sequences(feat, horizon_days, flat_pct)

    min_required = 200 if horizon_days < 100 else 80  # 1y horizon inherently yields fewer usable windows
    if len(X) < min_required:
        print(f"Not enough data for {ticker} [{horizon_key}] ({len(X)} sequences) -- skipping")
        return

    splits = chronological_split(X, y_return, y_direction)
    X_train, yr_train, yd_train = splits["train"]
    X_val, yr_val, yd_val = splits["val"]
    X_test, yr_test, yd_test = splits["test"]

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
    metrics = evaluate(model, X_test, yr_test, yd_test, mean, std, horizon_days)
    save_model(ticker, horizon_key, model, mean, std, metrics)
    print(f"{ticker} [{horizon_key}]: directional_accuracy={metrics['directional_accuracy']:.3f} "
          f"(baseline={metrics['baseline_directional_accuracy']:.3f}) mae_pct={metrics['mae_pct']:.2f}")


def evaluate(model, X_test, yr_test, yd_test, mean, std, horizon_days):
    model.eval()
    X_norm = (X_test - mean) / std
    with torch.no_grad():
        mean_pred, logvar_pred, dir_logits = model(torch.tensor(X_norm).to(DEVICE))
        mean_pred = mean_pred.cpu().numpy()
        dir_pred = dir_logits.argmax(dim=1).cpu().numpy()

    directional_accuracy = float((dir_pred == yd_test).mean())
    mae_pct = float(np.mean(np.abs(mean_pred - yr_test)) * 100)

    baseline_pred = np.ones_like(yd_test)  # naive: always predict "flat"
    baseline_directional_accuracy = float((baseline_pred == yd_test).mean())

    return {
        "directional_accuracy": directional_accuracy,
        "baseline_directional_accuracy": baseline_directional_accuracy,
        "mae_pct": mae_pct,
        "horizon_days": horizon_days,
        "test_samples": int(len(yd_test)),
    }


def save_model(ticker, horizon_key, model, mean, std, metrics):
    out_dir = os.path.join(MODEL_DIR, ticker, horizon_key)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="Train a single ticker, or omit to train all 15")
    parser.add_argument("--horizon", default=None, choices=list(HORIZONS.keys()),
                         help="Train a single horizon (1d/7d/30d/1y), or omit to train all 4")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else list(SUPPORTED_STOCKS.keys())
    horizon_keys = [args.horizon] if args.horizon else list(HORIZONS.keys())

    for t in tickers:
        for h in horizon_keys:
            try:
                train_one(t, h)
            except Exception as e:
                print(f"Failed on {t} [{h}]: {e}")