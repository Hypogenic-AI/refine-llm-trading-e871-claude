"""
Main experiment runner: tests LLM trading agent at daily/weekly/monthly horizons.
Compares against Buy-and-Hold, SMA Crossover, and Random baselines.
"""
import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from openai import OpenAI

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from backtester import run_backtest, compute_metrics, TradeResult
from strategies import (
    buy_and_hold_strategy,
    random_strategy,
    sma_crossover_strategy,
    create_llm_strategy,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuration
TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "NFLX"]
FREQUENCIES = ["daily", "weekly", "monthly"]
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"
LLM_RUNS = 3  # Number of runs per LLM config for variance estimation
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "stock_prices")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")

# LLM config
LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.3


def load_stock_data(ticker: str) -> pd.Series:
    """Load and return closing prices for a ticker in the test period."""
    filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    prices = df["Close"].loc[TEST_START:TEST_END]
    print(f"  {ticker}: {len(prices)} trading days ({prices.index[0].date()} to {prices.index[-1].date()})")
    return prices


def run_baselines(all_prices: dict) -> list:
    """Run all baseline strategies across all tickers and frequencies."""
    all_metrics = []

    for ticker, prices in all_prices.items():
        for freq in FREQUENCIES:
            # Buy-and-Hold
            result = run_backtest(prices, buy_and_hold_strategy, freq, ticker, "Buy-and-Hold")
            metrics = compute_metrics(result)
            all_metrics.append(metrics)
            print(f"  {ticker} | Buy-and-Hold | {freq}: CR={metrics['cumulative_return_pct']:.1f}%, SR={metrics['sharpe_ratio']:.3f}")

            # SMA Crossover
            result = run_backtest(prices, sma_crossover_strategy, freq, ticker, "SMA-Crossover")
            metrics = compute_metrics(result)
            all_metrics.append(metrics)
            print(f"  {ticker} | SMA-Crossover | {freq}: CR={metrics['cumulative_return_pct']:.1f}%, SR={metrics['sharpe_ratio']:.3f}")

            # Random (single run, deterministic via seed)
            result = run_backtest(prices, random_strategy, freq, ticker, "Random")
            metrics = compute_metrics(result)
            all_metrics.append(metrics)

    return all_metrics


def run_llm_experiments(all_prices: dict) -> list:
    """Run LLM agent at each frequency with multiple seeds."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    all_metrics = []
    all_results = []

    total_configs = len(TICKERS) * len(FREQUENCIES) * LLM_RUNS
    config_num = 0

    for ticker, prices in all_prices.items():
        for freq in FREQUENCIES:
            freq_metrics = []
            for run_idx in range(LLM_RUNS):
                config_num += 1
                run_seed = SEED + run_idx
                print(f"\n  [{config_num}/{total_configs}] {ticker} | LLM-{LLM_MODEL} | {freq} | run {run_idx + 1}/{LLM_RUNS}")

                strategy_fn = create_llm_strategy(client, model=LLM_MODEL, temperature=LLM_TEMPERATURE, run_seed=run_seed)
                result = run_backtest(prices, strategy_fn, freq, ticker, f"LLM-{LLM_MODEL}")
                metrics = compute_metrics(result)
                metrics["run"] = run_idx
                freq_metrics.append(metrics)
                all_results.append(result)

                print(f"    CR={metrics['cumulative_return_pct']:.1f}%, SR={metrics['sharpe_ratio']:.3f}, MDD={metrics['max_drawdown_pct']:.1f}%, Trades={metrics['num_trades']}")

                # Brief pause to respect rate limits
                time.sleep(0.5)

            # Average across runs for this ticker/freq
            avg_metrics = {
                "ticker": ticker,
                "strategy": f"LLM-{LLM_MODEL}",
                "frequency": freq,
                "cumulative_return_pct": np.mean([m["cumulative_return_pct"] for m in freq_metrics]),
                "annualized_return_pct": np.mean([m["annualized_return_pct"] for m in freq_metrics]),
                "sharpe_ratio": np.mean([m["sharpe_ratio"] for m in freq_metrics]),
                "sortino_ratio": np.mean([m["sortino_ratio"] for m in freq_metrics]),
                "max_drawdown_pct": np.mean([m["max_drawdown_pct"] for m in freq_metrics]),
                "volatility_pct": np.mean([m["volatility_pct"] for m in freq_metrics]),
                "num_trades": np.mean([m["num_trades"] for m in freq_metrics]),
                "win_rate_pct": np.mean([m["win_rate_pct"] for m in freq_metrics]),
                "cumulative_return_std": np.std([m["cumulative_return_pct"] for m in freq_metrics]),
                "sharpe_std": np.std([m["sharpe_ratio"] for m in freq_metrics]),
            }
            all_metrics.append(avg_metrics)

            # Also store individual run metrics
            for m in freq_metrics:
                m["is_individual_run"] = True
                all_metrics.append(m)

    return all_metrics, all_results


def classify_market_regime(prices: pd.Series, window: int = 50) -> pd.Series:
    """Classify each day as bull/bear/sideways based on price vs 50-day SMA."""
    sma = prices.rolling(window).mean()
    pct_diff = (prices - sma) / sma

    regime = pd.Series("sideways", index=prices.index)
    regime[pct_diff > 0.03] = "bull"
    regime[pct_diff < -0.03] = "bear"
    return regime


def main():
    print("=" * 70)
    print("LLM Trading Horizon Experiment")
    print(f"Model: {LLM_MODEL} | Test Period: {TEST_START} to {TEST_END}")
    print(f"Tickers: {TICKERS} | Frequencies: {FREQUENCIES}")
    print(f"LLM runs per config: {LLM_RUNS}")
    print("=" * 70)

    # Load data
    print("\n--- Loading stock data ---")
    all_prices = {}
    for ticker in TICKERS:
        all_prices[ticker] = load_stock_data(ticker)

    # Run baselines
    print("\n--- Running baseline strategies ---")
    baseline_metrics = run_baselines(all_prices)

    # Save baseline results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(baseline_metrics).to_csv(
        os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False
    )
    print(f"\n  Saved {len(baseline_metrics)} baseline results")

    # Run LLM experiments
    print("\n--- Running LLM experiments ---")
    start_time = time.time()
    llm_metrics, llm_results = run_llm_experiments(all_prices)
    elapsed = time.time() - start_time
    print(f"\n  LLM experiments completed in {elapsed:.0f}s")

    # Save LLM results
    # Separate averaged and individual
    llm_avg = [m for m in llm_metrics if not m.get("is_individual_run")]
    llm_individual = [m for m in llm_metrics if m.get("is_individual_run")]

    pd.DataFrame(llm_avg).to_csv(
        os.path.join(RESULTS_DIR, "llm_avg_results.csv"), index=False
    )
    pd.DataFrame(llm_individual).to_csv(
        os.path.join(RESULTS_DIR, "llm_individual_results.csv"), index=False
    )

    # Combine all averaged results
    all_avg = baseline_metrics + llm_avg
    pd.DataFrame(all_avg).to_csv(
        os.path.join(RESULTS_DIR, "all_results.csv"), index=False
    )

    # Market regime analysis
    print("\n--- Market regime classification ---")
    regime_data = []
    for ticker, prices in all_prices.items():
        regimes = classify_market_regime(prices)
        counts = regimes.value_counts()
        print(f"  {ticker}: {dict(counts)}")
        regime_data.append({"ticker": ticker, **dict(counts)})

    pd.DataFrame(regime_data).to_csv(
        os.path.join(RESULTS_DIR, "market_regimes.csv"), index=False
    )

    # Save portfolio value time series for LLM results (for plotting)
    for result in llm_results:
        fname = f"portfolio_{result.ticker}_{result.frequency}_run{result.decisions[:5]}.json"
        # Save just the first run per config for plotting
        pass  # We'll handle this in analysis

    # Save config
    config = {
        "seed": SEED,
        "tickers": TICKERS,
        "frequencies": FREQUENCIES,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_runs": LLM_RUNS,
        "transaction_cost_pct": 0.001,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("Experiment complete! Results saved to results/data/")
    print("=" * 70)

    # Print summary table
    print("\n--- Summary (LLM Averaged Results) ---")
    df_avg = pd.DataFrame(llm_avg)
    for freq in FREQUENCIES:
        subset = df_avg[df_avg["frequency"] == freq]
        mean_sr = subset["sharpe_ratio"].mean()
        mean_cr = subset["cumulative_return_pct"].mean()
        mean_mdd = subset["max_drawdown_pct"].mean()
        print(f"  {freq:8s}: Avg CR={mean_cr:+.1f}%, Avg SR={mean_sr:.3f}, Avg MDD={mean_mdd:.1f}%")


if __name__ == "__main__":
    main()
