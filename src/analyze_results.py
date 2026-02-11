"""
Analyze and visualize experiment results.
Statistical tests for hypothesis testing across trading horizons.
"""
import os, sys, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def load_results():
    """Load all experiment results."""
    baselines = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"))
    llm_avg = pd.read_csv(os.path.join(RESULTS_DIR, "llm_avg_results.csv"))
    llm_individual = pd.read_csv(os.path.join(RESULTS_DIR, "llm_individual_results.csv"))
    return baselines, llm_avg, llm_individual


def plot_sharpe_by_frequency(baselines, llm_avg):
    """Bar plot comparing Sharpe ratios across strategies and frequencies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Combine data
    freq_order = ["daily", "weekly", "monthly"]
    strategies = ["Buy-and-Hold", "SMA-Crossover", "LLM-gpt-4.1-mini"]
    colors = {"Buy-and-Hold": "#2196F3", "SMA-Crossover": "#FF9800", "LLM-gpt-4.1-mini": "#4CAF50"}

    x = np.arange(len(freq_order))
    width = 0.25

    for i, strat in enumerate(strategies):
        if strat.startswith("LLM"):
            df = llm_avg[llm_avg["strategy"] == strat]
        else:
            df = baselines[baselines["strategy"] == strat]

        means = [df[df["frequency"] == f]["sharpe_ratio"].mean() for f in freq_order]
        stds = [df[df["frequency"] == f]["sharpe_ratio"].std() for f in freq_order]

        bars = ax.bar(x + i * width, means, width, label=strat, color=colors[strat],
                      yerr=stds, capsize=4, alpha=0.85)

    ax.set_xlabel("Rebalancing Frequency")
    ax.set_ylabel("Sharpe Ratio (Annualized)")
    ax.set_title("Sharpe Ratio by Trading Frequency and Strategy")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f.capitalize() for f in freq_order])
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sharpe_by_frequency.png"), dpi=150)
    plt.close()
    print("  Saved sharpe_by_frequency.png")


def plot_cumulative_return_by_frequency(baselines, llm_avg):
    """Bar plot comparing cumulative returns across strategies and frequencies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    freq_order = ["daily", "weekly", "monthly"]
    strategies = ["Buy-and-Hold", "SMA-Crossover", "LLM-gpt-4.1-mini"]
    colors = {"Buy-and-Hold": "#2196F3", "SMA-Crossover": "#FF9800", "LLM-gpt-4.1-mini": "#4CAF50"}

    x = np.arange(len(freq_order))
    width = 0.25

    for i, strat in enumerate(strategies):
        if strat.startswith("LLM"):
            df = llm_avg[llm_avg["strategy"] == strat]
        else:
            df = baselines[baselines["strategy"] == strat]

        means = [df[df["frequency"] == f]["cumulative_return_pct"].mean() for f in freq_order]
        stds = [df[df["frequency"] == f]["cumulative_return_pct"].std() for f in freq_order]
        ax.bar(x + i * width, means, width, label=strat, color=colors[strat],
               yerr=stds, capsize=4, alpha=0.85)

    ax.set_xlabel("Rebalancing Frequency")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Cumulative Return by Trading Frequency and Strategy")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f.capitalize() for f in freq_order])
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cumulative_return_by_frequency.png"), dpi=150)
    plt.close()
    print("  Saved cumulative_return_by_frequency.png")


def plot_llm_per_stock(llm_avg, baselines):
    """Per-stock comparison of LLM at different frequencies vs Buy-and-Hold."""
    tickers = llm_avg["ticker"].unique()
    freq_order = ["daily", "weekly", "monthly"]

    fig, axes = plt.subplots(1, len(tickers), figsize=(4 * len(tickers), 5), sharey=True)
    if len(tickers) == 1:
        axes = [axes]

    for idx, ticker in enumerate(tickers):
        ax = axes[idx]
        llm_sub = llm_avg[(llm_avg["ticker"] == ticker)]
        bh_sub = baselines[(baselines["ticker"] == ticker) & (baselines["strategy"] == "Buy-and-Hold")]

        llm_sr = [llm_sub[llm_sub["frequency"] == f]["sharpe_ratio"].values[0] for f in freq_order]
        bh_sr = bh_sub["sharpe_ratio"].iloc[0]  # Same for all frequencies

        x = np.arange(len(freq_order))
        bars = ax.bar(x, llm_sr, color=["#ef5350", "#66BB6A", "#42A5F5"], alpha=0.85)
        ax.axhline(y=bh_sr, color="gray", linewidth=2, linestyle="--", label=f"Buy-Hold ({bh_sr:.2f})")
        ax.set_title(ticker)
        ax.set_xticks(x)
        ax.set_xticklabels([f.capitalize() for f in freq_order], rotation=45)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Sharpe Ratio")
    fig.suptitle("LLM Agent Sharpe Ratio by Stock and Frequency", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "llm_per_stock_sharpe.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved llm_per_stock_sharpe.png")


def plot_drawdown_comparison(baselines, llm_avg):
    """Compare max drawdown across strategies and frequencies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    freq_order = ["daily", "weekly", "monthly"]
    strategies = ["Buy-and-Hold", "SMA-Crossover", "LLM-gpt-4.1-mini"]
    colors = {"Buy-and-Hold": "#2196F3", "SMA-Crossover": "#FF9800", "LLM-gpt-4.1-mini": "#4CAF50"}

    x = np.arange(len(freq_order))
    width = 0.25

    for i, strat in enumerate(strategies):
        if strat.startswith("LLM"):
            df = llm_avg
        else:
            df = baselines[baselines["strategy"] == strat]

        means = [abs(df[df["frequency"] == f]["max_drawdown_pct"].mean()) for f in freq_order]
        ax.bar(x + i * width, means, width, label=strat, color=colors[strat], alpha=0.85)

    ax.set_xlabel("Rebalancing Frequency")
    ax.set_ylabel("Max Drawdown (%, lower is better)")
    ax.set_title("Maximum Drawdown by Trading Frequency and Strategy")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f.capitalize() for f in freq_order])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "drawdown_by_frequency.png"), dpi=150)
    plt.close()
    print("  Saved drawdown_by_frequency.png")


def plot_portfolio_timeseries(baselines, llm_portfolio_path):
    """Plot portfolio value over time for different strategies."""
    if not os.path.exists(llm_portfolio_path):
        print("  Skipping portfolio time series (no data)")
        return

    with open(llm_portfolio_path) as f:
        portfolio_data = json.load(f)

    # Plot for each ticker
    tickers = sorted(set(p["ticker"] for p in portfolio_data))

    for ticker in tickers[:3]:  # Just first 3 for brevity
        fig, ax = plt.subplots(figsize=(12, 5))

        for pdata in portfolio_data:
            if pdata["ticker"] == ticker:
                dates = pd.to_datetime(pdata["dates"])
                values = pdata["portfolio_values"]
                ax.plot(dates, values, label=f"LLM {pdata['frequency']}", alpha=0.8, linewidth=1.5)

        ax.set_title(f"{ticker} - Portfolio Value by Rebalancing Frequency")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"portfolio_{ticker}.png"), dpi=150)
        plt.close()
        print(f"  Saved portfolio_{ticker}.png")


def plot_trades_vs_return(llm_avg):
    """Scatter plot of number of trades vs return, colored by frequency."""
    fig, ax = plt.subplots(figsize=(8, 6))

    freq_colors = {"daily": "#ef5350", "weekly": "#66BB6A", "monthly": "#42A5F5"}

    for freq in ["daily", "weekly", "monthly"]:
        sub = llm_avg[llm_avg["frequency"] == freq]
        ax.scatter(sub["num_trades"], sub["cumulative_return_pct"],
                   c=freq_colors[freq], label=freq.capitalize(), s=100, alpha=0.8,
                   edgecolors="black", linewidth=0.5)
        for _, row in sub.iterrows():
            ax.annotate(row["ticker"], (row["num_trades"], row["cumulative_return_pct"]),
                       fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Number of Trades")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Trade Frequency vs. Return for LLM Agent")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "trades_vs_return.png"), dpi=150)
    plt.close()
    print("  Saved trades_vs_return.png")


def statistical_tests(llm_individual):
    """Run statistical tests comparing horizons."""
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    tickers = llm_individual["ticker"].unique()
    results = {}

    # For each pair of frequencies, compare Sharpe ratios
    freq_pairs = [("daily", "weekly"), ("daily", "monthly"), ("weekly", "monthly")]

    for metric in ["sharpe_ratio", "cumulative_return_pct", "max_drawdown_pct"]:
        print(f"\n--- {metric} ---")
        results[metric] = {}

        for f1, f2 in freq_pairs:
            # Get per-ticker averaged values for each frequency
            vals1 = []
            vals2 = []
            for ticker in tickers:
                v1 = llm_individual[(llm_individual["ticker"] == ticker) & (llm_individual["frequency"] == f1)][metric].mean()
                v2 = llm_individual[(llm_individual["ticker"] == ticker) & (llm_individual["frequency"] == f2)][metric].mean()
                vals1.append(v1)
                vals2.append(v2)

            vals1 = np.array(vals1)
            vals2 = np.array(vals2)
            diff = vals2 - vals1

            # Wilcoxon signed-rank test (non-parametric, paired)
            if len(vals1) >= 5:
                try:
                    stat, p_value = stats.wilcoxon(vals1, vals2)
                except ValueError:
                    stat, p_value = np.nan, np.nan
            else:
                # With 5 samples, use paired t-test as well
                stat_t, p_value_t = stats.ttest_rel(vals1, vals2)
                try:
                    stat, p_value = stats.wilcoxon(vals1, vals2, zero_method="wilcox")
                except (ValueError, Exception):
                    stat, p_value = np.nan, np.nan

            # Effect size (Cohen's d for paired samples)
            d_std = np.std(diff, ddof=1)
            cohens_d = np.mean(diff) / d_std if d_std > 0 else 0.0

            # Paired t-test
            t_stat, t_p = stats.ttest_rel(vals1, vals2)

            results[metric][f"{f1}_vs_{f2}"] = {
                "mean_diff": np.mean(diff),
                "wilcoxon_p": p_value,
                "ttest_p": t_p,
                "cohens_d": cohens_d,
                "f1_mean": np.mean(vals1),
                "f2_mean": np.mean(vals2),
            }

            sig = "**" if t_p < 0.05 else ""
            print(f"  {f1} vs {f2}: {metric} diff = {np.mean(diff):+.3f} (d={cohens_d:.2f}), "
                  f"t-test p={t_p:.4f}{sig}, Wilcoxon p={p_value:.4f}")
            print(f"    {f1} mean={np.mean(vals1):.3f}, {f2} mean={np.mean(vals2):.3f}")

    return results


def create_summary_table(baselines, llm_avg):
    """Create a summary table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Average Across All Tickers")
    print("=" * 80)
    header = f"{'Strategy':<25} {'Freq':>8} {'CR(%)':>8} {'SR':>8} {'MDD(%)':>8} {'Trades':>8}"
    print(header)
    print("-" * 80)

    for freq in ["daily", "weekly", "monthly"]:
        for strat in ["Buy-and-Hold", "SMA-Crossover", "Random"]:
            sub = baselines[(baselines["strategy"] == strat) & (baselines["frequency"] == freq)]
            if len(sub) > 0:
                print(f"{strat:<25} {freq:>8} {sub['cumulative_return_pct'].mean():>8.1f} "
                      f"{sub['sharpe_ratio'].mean():>8.3f} {sub['max_drawdown_pct'].mean():>8.1f} "
                      f"{sub['num_trades'].mean():>8.1f}")

        sub = llm_avg[llm_avg["frequency"] == freq]
        if len(sub) > 0:
            print(f"{'LLM-gpt-4.1-mini':<25} {freq:>8} {sub['cumulative_return_pct'].mean():>8.1f} "
                  f"{sub['sharpe_ratio'].mean():>8.3f} {sub['max_drawdown_pct'].mean():>8.1f} "
                  f"{sub['num_trades'].mean():>8.1f}")
        print()

    return None


def main():
    print("Loading results...")
    baselines, llm_avg, llm_individual = load_results()

    print(f"Baselines: {len(baselines)} rows")
    print(f"LLM averaged: {len(llm_avg)} rows")
    print(f"LLM individual: {len(llm_individual)} rows")

    # Summary table
    create_summary_table(baselines, llm_avg)

    # Plots
    print("\n--- Generating plots ---")
    plot_sharpe_by_frequency(baselines, llm_avg)
    plot_cumulative_return_by_frequency(baselines, llm_avg)
    plot_llm_per_stock(llm_avg, baselines)
    plot_drawdown_comparison(baselines, llm_avg)
    plot_trades_vs_return(llm_avg)
    plot_portfolio_timeseries(baselines, os.path.join(RESULTS_DIR, "llm_portfolio_data.json"))

    # Statistical tests
    stat_results = statistical_tests(llm_individual)

    # Save stats
    with open(os.path.join(RESULTS_DIR, "statistical_tests.json"), "w") as f:
        json.dump(stat_results, f, indent=2, default=str)

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
