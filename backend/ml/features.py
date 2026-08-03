"""
Turns raw OHLCV into model-ready features.

Kept deliberately simple and well-understood (returns, moving averages, RSI,
rolling volatility) rather than exotic indicators -- easier to explain in a
viva, and less prone to overfitting on 15 stocks worth of data.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "ma_5",
    "ma_10",
    "ma_20",
    "rsi_14",
    "volatility_10",
    "volume_change",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: raw OHLCV DataFrame (columns: open, high, low, close, volume)
    Output: same index, with FEATURE_COLUMNS + 'close' added, NaN rows dropped.
    """
    out = df.copy()

    out["return_1d"] = out["close"].pct_change(1)
    out["return_5d"] = out["close"].pct_change(5)

    out["ma_5"] = out["close"].rolling(5).mean() / out["close"] - 1
    out["ma_10"] = out["close"].rolling(10).mean() / out["close"] - 1
    out["ma_20"] = out["close"].rolling(20).mean() / out["close"] - 1

    out["rsi_14"] = _rsi(out["close"], period=14)

    out["volatility_10"] = out["return_1d"].rolling(10).std()

    out["volume_change"] = out["volume"].pct_change(1)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return (rsi / 100.0).fillna(0.5)  # normalize to 0-1, neutral fill
