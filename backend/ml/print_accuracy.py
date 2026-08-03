"""
Prints a summary table of backtested accuracy for every trained (ticker,
horizon) model -- reads straight from the metrics.json files ml/train.py
already saved, doesn't touch the models or run any inference.

Run:
    python -m ml.print_accuracy                # every trained model
    python -m ml.print_accuracy --ticker TSLA   # just one ticker, all its horizons
    python -m ml.print_accuracy --sort acc      # sort by directional accuracy (default: ticker)
"""

import argparse
import json
import os

from config import SUPPORTED_STOCKS, HORIZONS, MODEL_DIR


def load_all_metrics(ticker_filter=None):
    rows = []
    tickers = [ticker_filter] if ticker_filter else list(SUPPORTED_STOCKS.keys())

    for ticker in tickers:
        for horizon_key in HORIZONS.keys():
            metrics_path = os.path.join(MODEL_DIR, ticker, horizon_key, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as f:
                m = json.load(f)
            rows.append({
                "ticker": ticker,
                "horizon": horizon_key,
                "directional_accuracy": m["directional_accuracy"],
                "baseline": m["baseline_directional_accuracy"],
                "beats_baseline": m["directional_accuracy"] > m["baseline_directional_accuracy"],
                "mae_pct": m["mae_pct"],
                "test_samples": m["test_samples"],
            })
    return rows


def print_table(rows, sort_by="ticker"):
    if not rows:
        print("No trained models found. Run `python -m ml.train` first.")
        return

    if sort_by == "acc":
        rows = sorted(rows, key=lambda r: r["directional_accuracy"], reverse=True)
    else:
        rows = sorted(rows, key=lambda r: (r["ticker"], list(HORIZONS.keys()).index(r["horizon"])))

    header = f"{'Ticker':<8}{'Horizon':<9}{'Accuracy':<11}{'Baseline':<11}{'Beats?':<8}{'MAE %':<9}{'Test N':<8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        beats = "YES" if r["beats_baseline"] else "no"
        print(
            f"{r['ticker']:<8}{r['horizon']:<9}{r['directional_accuracy']*100:>7.1f}%   "
            f"{r['baseline']*100:>7.1f}%   {beats:<8}{r['mae_pct']:>6.2f}   {r['test_samples']:>6}"
        )

    print("-" * len(header))
    n_beating = sum(1 for r in rows if r["beats_baseline"])
    print(f"{n_beating}/{len(rows)} models beat their naive baseline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="Show only one ticker (all its trained horizons)")
    parser.add_argument("--sort", default="ticker", choices=["ticker", "acc"])
    args = parser.parse_args()

    rows = load_all_metrics(args.ticker)
    print_table(rows, sort_by=args.sort)