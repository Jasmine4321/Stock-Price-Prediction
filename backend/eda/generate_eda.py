"""
Generates EDA (Exploratory Data Analysis) charts for the project documentation.

This script is intentionally standalone -- it is NOT imported by main.py or
any API route. It reuses your existing data/fetcher.py and ml/features.py so
the charts reflect exactly the same data and features the model is trained
on, but it plays no role in the running app itself.

Run:
    python -m eda.generate_eda --ticker TSLA
    python -m eda.generate_eda --ticker TSLA --ticker AAPL --ticker TSLA
    python -m eda.generate_eda            # runs for all 15 supported stocks

Output:
    eda/output/{ticker}_eda.png   -- one 6-panel figure per ticker
    eda/output/correlation_across_stocks.png  -- cross-stock return correlation
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SUPPORTED_STOCKS
from data.fetcher import get_history
from ml.features import build_features, FEATURE_COLUMNS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def generate_eda_for_ticker(ticker: str):
    print(f"Generating EDA for {ticker}...")
    df = get_history(ticker)
    feat = build_features(df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{ticker} — Exploratory Data Analysis ({SUPPORTED_STOCKS.get(ticker, '')})", fontsize=14)

    # 1. Price trend with moving averages
    ax = axes[0, 0]
    ax.plot(df.index, df["close"], label="Close", linewidth=1)
    ax.plot(df.index, df["close"].rolling(20).mean(), label="20-day MA", linewidth=1)
    ax.plot(df.index, df["close"].rolling(50).mean(), label="50-day MA", linewidth=1)
    ax.set_title("Price Trend")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    # 2. Daily returns distribution
    ax = axes[0, 1]
    returns = df["close"].pct_change().dropna() * 100
    ax.hist(returns, bins=60, density=True, alpha=0.7, color="#2B5E6B")
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
            color="#B23A2E", linewidth=1.5, label="Normal fit")
    ax.set_title(f"Daily Return Distribution (mean={mu:.2f}%, std={sigma:.2f}%)")
    ax.set_xlabel("Daily return (%)")
    ax.legend(fontsize=8)

    # 3. Rolling volatility
    ax = axes[0, 2]
    rolling_vol = df["close"].pct_change().rolling(20).std() * 100
    ax.plot(df.index, rolling_vol, color="#9C7F3F", linewidth=1)
    ax.set_title("20-day Rolling Volatility")
    ax.set_ylabel("Std. dev of daily return (%)")
    ax.tick_params(axis="x", rotation=30)

    # 4. Trading volume over time
    ax = axes[1, 0]
    ax.bar(df.index, df["volume"], width=1.0, color="#5B6560", alpha=0.6)
    ax.set_title("Trading Volume")
    ax.tick_params(axis="x", rotation=30)

    # 5. RSI-14 over time
    ax = axes[1, 1]
    rsi = feat["rsi_14"] * 100  # feature is normalized 0-1, scale back for readability
    ax.plot(feat.index, rsi, color="#2B5E6B", linewidth=1)
    ax.axhline(70, color="#B23A2E", linestyle="--", linewidth=0.8, label="overbought (70)")
    ax.axhline(30, color="#1E7A4C", linestyle="--", linewidth=0.8, label="oversold (30)")
    ax.set_title("RSI (14-day)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30)

    # 6. Feature correlation heatmap
    ax = axes[1, 2]
    corr = feat[FEATURE_COLUMNS].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=7)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=7)
    ax.set_title("Feature Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_eda.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def generate_cross_stock_correlation(tickers: list[str]):
    """How correlated are daily returns across the different stocks -- useful
    to show in the report as evidence for/against diversification, and to
    sanity-check that stocks aren't accidentally near-duplicates of each other."""
    print("Generating cross-stock return correlation...")
    returns = {}
    for t in tickers:
        try:
            df = get_history(t)
            returns[t] = df["close"].pct_change()
        except Exception as e:
            print(f"  skipping {t}: {e}")

    returns_df = pd.DataFrame(returns).dropna()
    corr = returns_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Daily Return Correlation Across Stocks")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "correlation_across_stocks.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append", default=None,
                         help="Repeatable. Omit to run for all 15 supported stocks.")
    args = parser.parse_args()

    tickers = args.ticker if args.ticker else list(SUPPORTED_STOCKS.keys())

    for t in tickers:
        try:
            generate_eda_for_ticker(t)
        except Exception as e:
            print(f"Failed on {t}: {e}")

    generate_cross_stock_correlation(tickers)