"""
Run LLM trading experiments at daily, weekly, and monthly horizons.
Uses real GPT-4.1-mini API calls for all trading decisions.
"""
import os, sys, json, time
import pandas as pd
import numpy as np
import random
from datetime import datetime
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from backtester import run_backtest, compute_metrics
from strategies import create_llm_strategy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "NFLX"]
FREQUENCIES = ["daily", "weekly", "monthly"]
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "stock_prices")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")

LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.3

# Runs per frequency (daily has more data points -> lower variance naturally)
RUNS_PER_FREQ = {"daily": 2, "weekly": 3, "monthly": 3}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    all_individual = []
    all_averaged = []
    portfolio_data = []

    # Load all prices
    all_prices = {}
    for ticker in TICKERS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=["Date"], index_col="Date")
        all_prices[ticker] = df["Close"].loc[TEST_START:TEST_END]

    total_configs = sum(len(TICKERS) * RUNS_PER_FREQ[f] for f in FREQUENCIES)
    config_idx = 0
    start_time = time.time()

    for freq in FREQUENCIES:
        n_runs = RUNS_PER_FREQ[freq]
        print(f"\n{'='*60}")
        print(f"Frequency: {freq} ({n_runs} runs per ticker)")
        print(f"{'='*60}")

        for ticker in TICKERS:
            prices = all_prices[ticker]
            freq_runs = []

            for run_idx in range(n_runs):
                config_idx += 1
                run_seed = SEED + run_idx * 7  # Different seeds
                elapsed = time.time() - start_time
                print(f"\n[{config_idx}/{total_configs}] {ticker} | {freq} | run {run_idx+1}/{n_runs} (elapsed: {elapsed:.0f}s)")

                strategy_fn = create_llm_strategy(
                    client, model=LLM_MODEL, temperature=LLM_TEMPERATURE, run_seed=run_seed
                )
                result = run_backtest(prices, strategy_fn, freq, ticker, f"LLM-{LLM_MODEL}")
                metrics = compute_metrics(result)
                metrics["run"] = run_idx
                freq_runs.append(metrics)

                print(f"  CR={metrics['cumulative_return_pct']:+.1f}% | SR={metrics['sharpe_ratio']:.3f} | MDD={metrics['max_drawdown_pct']:.1f}% | Trades={metrics['num_trades']}")

                # Save portfolio values for first run
                if run_idx == 0:
                    portfolio_data.append({
                        "ticker": ticker,
                        "frequency": freq,
                        "dates": [str(d.date()) for d in result.dates],
                        "portfolio_values": result.portfolio_values,
                        "positions": result.positions,
                        "decisions": result.decisions,
                    })

            # Store individual runs
            all_individual.extend(freq_runs)

            # Compute averaged metrics
            avg = {
                "ticker": ticker,
                "strategy": f"LLM-{LLM_MODEL}",
                "frequency": freq,
            }
            for key in ["cumulative_return_pct", "annualized_return_pct", "sharpe_ratio",
                        "sortino_ratio", "max_drawdown_pct", "volatility_pct", "num_trades", "win_rate_pct"]:
                vals = [m[key] for m in freq_runs]
                avg[key] = np.mean(vals)
                avg[f"{key}_std"] = np.std(vals)
            all_averaged.append(avg)
            print(f"  AVG: CR={avg['cumulative_return_pct']:+.1f}% | SR={avg['sharpe_ratio']:.3f}")

    # Save results
    pd.DataFrame(all_individual).to_csv(os.path.join(RESULTS_DIR, "llm_individual_results.csv"), index=False)
    pd.DataFrame(all_averaged).to_csv(os.path.join(RESULTS_DIR, "llm_avg_results.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "llm_portfolio_data.json"), "w") as f:
        json.dump(portfolio_data, f)

    # Save config
    config = {
        "seed": SEED,
        "tickers": TICKERS,
        "frequencies": FREQUENCIES,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "runs_per_freq": RUNS_PER_FREQ,
        "transaction_cost_pct": 0.001,
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": time.time() - start_time,
    }
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All experiments complete! Elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"Results saved to {RESULTS_DIR}")

    # Print summary
    print(f"\n--- LLM Performance by Frequency (Averaged) ---")
    df = pd.DataFrame(all_averaged)
    for freq in FREQUENCIES:
        sub = df[df["frequency"] == freq]
        print(f"  {freq:8s}: CR={sub['cumulative_return_pct'].mean():+.1f}% | SR={sub['sharpe_ratio'].mean():.3f} | MDD={sub['max_drawdown_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
