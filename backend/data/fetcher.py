"""
Pulls historical OHLCV data for the 15 supported tickers.

Primary source: yfinance (free, no API key) -- can be intermittently
blocked/rate-limited by Yahoo. Falls back to Stooq automatically if
yfinance fails after retries, since Stooq has been more reliable in
practice. Both sources return the same shape, so the rest of the
pipeline (features -> model -> forecast) never needs to know which
one actually served the data.

Offline fallback: generate a synthetic random-walk price series so the
rest of the pipeline can be built and tested without internet. Swap
USE_SYNTHETIC to True only for that offline dev case.
"""

import time

import numpy as np
import pandas as pd
import yfinance as yf

USE_SYNTHETIC = False

_CACHE = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes -- avoids re-hitting the data source on every page load


def get_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns:
    open, high, low, close, volume
    """
    if USE_SYNTHETIC:
        return _synthetic_history(ticker)

    cache_key = f"{ticker}:{period}"
    cached = _CACHE.get(cache_key)
    if cached and (time.time() - cached["ts"]) < _CACHE_TTL_SECONDS:
        return cached["df"]

    df = None
    try:
        df = _fetch_yfinance(ticker, period)
    except Exception as e:
        print(f"[fetcher] yfinance failed for {ticker} ({e}); trying Stooq")

    if df is None or df.empty:
        try:
            df = _fetch_stooq(ticker, period)
        except Exception as e:
            raise ValueError(f"No data returned for {ticker} from yfinance or Stooq. Last error: {e}")

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} from either source")

    _CACHE[cache_key] = {"df": df, "ts": time.time()}
    return df


def _fetch_yfinance(ticker: str, period: str, retries: int = 2) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.index.name = "date"
                return df
        except Exception as e:
            last_error = e
        time.sleep(1.0 * (attempt + 1))
    raise ValueError(f"yfinance returned no data after {retries} attempts ({last_error})")


def _fetch_stooq(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    symbol = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

    last_error = None
    for attempt in range(retries):
        try:
            df = pd.read_csv(url)
            if not df.empty and "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.index.name = "date"
                if period == "5y":
                    df = df[df.index >= (df.index.max() - pd.DateOffset(years=5))]
                return df
        except Exception as e:
            last_error = e
        time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"Stooq returned no data after {retries} attempts ({last_error})")


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