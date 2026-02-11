"""Run baseline strategies (no API calls needed)."""
import os, sys, json
import pandas as pd
import numpy as np
import random

sys.path.insert(0, os.path.dirname(__file__))
from backtester import run_backtest, compute_metrics
from strategies import buy_and_hold_strategy, random_strategy, sma_crossover_strategy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "NFLX"]
FREQUENCIES = ["daily", "weekly", "monthly"]
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "stock_prices")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_metrics = []
    portfolio_series = {}

    for ticker in TICKERS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=["Date"], index_col="Date")
        prices = df["Close"].loc[TEST_START:TEST_END]
        print(f"{ticker}: {len(prices)} days, ${prices.iloc[0]:.2f} -> ${prices.iloc[-1]:.2f}")

        for freq in FREQUENCIES:
            # Buy-and-Hold
            r = run_backtest(prices, buy_and_hold_strategy, freq, ticker, "Buy-and-Hold")
            m = compute_metrics(r)
            all_metrics.append(m)
            portfolio_series[f"BH_{ticker}_{freq}"] = r.portfolio_values

            # SMA Crossover
            r = run_backtest(prices, sma_crossover_strategy, freq, ticker, "SMA-Crossover")
            m = compute_metrics(r)
            all_metrics.append(m)
            portfolio_series[f"SMA_{ticker}_{freq}"] = r.portfolio_values

            # Random
            r = run_backtest(prices, random_strategy, freq, ticker, "Random")
            m = compute_metrics(r)
            all_metrics.append(m)

            print(f"  {freq:8s} | BH: CR={all_metrics[-3]['cumulative_return_pct']:+.1f}% SR={all_metrics[-3]['sharpe_ratio']:.3f} | SMA: CR={all_metrics[-2]['cumulative_return_pct']:+.1f}% SR={all_metrics[-2]['sharpe_ratio']:.3f} | Rand: CR={all_metrics[-1]['cumulative_return_pct']:+.1f}%")

    pd.DataFrame(all_metrics).to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    print(f"\nSaved {len(all_metrics)} baseline results to results/data/baseline_results.csv")

if __name__ == "__main__":
    main()
