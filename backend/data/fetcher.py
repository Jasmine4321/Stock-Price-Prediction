"""
Pulls historical OHLCV data for the 15 supported tickers.

Real source: yfinance (free, no API key).
Offline fallback: generate a synthetic random-walk price series so the rest
of the pipeline (features -> model -> forecast) can be built and tested
without needing a live internet connection. Swap USE_SYNTHETIC to False
once you're running this with internet access.
"""

import numpy as np
import pandas as pd

USE_SYNTHETIC = False  # flip to True only for offline dev without internet


def get_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns:
    open, high, low, close, volume
    """
    if USE_SYNTHETIC:
        return _synthetic_history(ticker)

    import yfinance as yf

    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index.name = "date"
    return df


def get_latest_price(ticker: str) -> dict:
    df = get_history(ticker, period="5d")
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    change = last["close"] - prev["close"]
    change_pct = (change / prev["close"]) * 100 if prev["close"] else 0.0
    return {
        "ticker": ticker,
        "price": round(float(last["close"]), 2),
        "change": round(float(change), 2),
        "change_pct": round(float(change_pct), 2),
        "day_high": round(float(last["high"]), 2),
        "day_low": round(float(last["low"]), 2),
        "volume": int(last["volume"]),
    }


def _synthetic_history(ticker: str, days: int = 1260) -> pd.DataFrame:
    """Deterministic-ish synthetic random walk, seeded per ticker, for offline testing."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    start_price = rng.uniform(50, 400)
    daily_returns = rng.normal(loc=0.0004, scale=0.02, size=days)
    close = start_price * np.cumprod(1 + daily_returns)

    high = close * (1 + rng.uniform(0.001, 0.02, size=days))
    low = close * (1 - rng.uniform(0.001, 0.02, size=days))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=days)
    volume = rng.integers(1_000_000, 20_000_000, size=days)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df
