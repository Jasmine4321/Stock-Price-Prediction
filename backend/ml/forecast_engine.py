"""
Loads a trained model for a given (ticker, horizon) and runs real inference.
No LLM involved anywhere in this file.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from config import MODEL_DIR, LOOKBACK_DAYS, HORIZONS, DEFAULT_HORIZON, SUPPORTED_STOCKS
from data.fetcher import get_history
from ml.features import build_features, FEATURE_COLUMNS
from ml.lstm_model import StockLSTM

_MODEL_CACHE = {}


def _load(ticker: str, horizon_key: str):
    cache_key = f"{ticker}:{horizon_key}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_dir = os.path.join(MODEL_DIR, ticker, horizon_key)
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model for {ticker} [{horizon_key}]. "
            f"Run `python -m ml.train --ticker {ticker} --horizon {horizon_key}` first."
        )

    model = StockLSTM(n_features=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with open(os.path.join(model_dir, "scaler.json")) as f:
        scaler = json.load(f)
    with open(os.path.join(model_dir, "metrics.json")) as f:
        metrics = json.load(f)

    _MODEL_CACHE[cache_key] = (model, scaler, metrics)
    return _MODEL_CACHE[cache_key]


def run_forecast(ticker: str, horizon_key: str = DEFAULT_HORIZON) -> dict:
    if ticker not in SUPPORTED_STOCKS:
        raise ValueError(f"{ticker} is not one of the supported stocks")
    if horizon_key not in HORIZONS:
        raise ValueError(f"{horizon_key} is not a supported horizon. Choose from {list(HORIZONS.keys())}")

    model, scaler, metrics = _load(ticker, horizon_key)

    df = get_history(ticker)
    feat = build_features(df)
    if len(feat) < LOOKBACK_DAYS:
        raise ValueError(f"Not enough recent history for {ticker} to forecast")

    window = feat[FEATURE_COLUMNS].values[-LOOKBACK_DAYS:]
    mean = np.array(scaler["mean"])
    std = np.array(scaler["std"])
    window_norm = (window - mean) / std

    x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_mean, pred_logvar, dir_logits = model(x)
        pred_return = pred_mean.item()
        pred_std = float(np.exp(0.5 * pred_logvar.item()))
        probs = F.softmax(dir_logits, dim=1).squeeze(0).tolist()

    current_price = float(feat["close"].iloc[-1])
    predicted_price = current_price * (1 + pred_return)
    predicted_high = current_price * (1 + pred_return + pred_std)
    predicted_low = current_price * (1 + pred_return - pred_std)

    return {
        "ticker": ticker,
        "horizon": horizon_key,
        "horizon_days": HORIZONS[horizon_key]["days"],
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_high": round(max(predicted_high, predicted_low), 2),
        "predicted_low": round(min(predicted_high, predicted_low), 2),
        "p_down": round(probs[0], 3),
        "p_flat": round(probs[1], 3),
        "p_up": round(probs[2], 3),
        "backtested_accuracy": {
            "directional_accuracy": metrics["directional_accuracy"],
            "baseline_directional_accuracy": metrics["baseline_directional_accuracy"],
            "mae_pct": metrics["mae_pct"],
            "test_samples": metrics["test_samples"],
        },
    }