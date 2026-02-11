# Refining LLM Trading: Do LLM Agents Perform Better at Longer Decision Horizons?

## Overview
This project investigates whether LLM trading agents achieve better performance when making longer-term (weekly/monthly) trading decisions compared to the standard daily approach. We test a GPT-4.1-mini agent at daily, weekly, and monthly rebalancing frequencies across 5 major stocks over 2023-2024.

## Key Findings
- **Weekly rebalancing outperforms daily** — Sharpe ratio 1.028 vs. 0.892 (+15%), with 75% fewer trades
- **Monthly rebalancing is too infrequent** — Sharpe drops to 0.421, with 2 out of 5 stocks showing negative returns
- **No LLM frequency beats Buy-and-Hold** (Sharpe 1.620) in this bull market period, consistent with FINSABER (KDD 2026)
- **Non-monotonic relationship**: there is a "Goldilocks zone" at weekly frequency where LLMs filter noise without losing signal
- **MSFT benefits most from weekly**: 175% Sharpe improvement (0.284 to 0.781) from daily to weekly

## Reproduce Results

```bash
# Setup
uv venv && source .venv/bin/activate
uv add pandas numpy matplotlib openai scipy seaborn

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run baselines (instant, no API calls)
python src/run_baselines.py

# Run LLM experiments (~45 minutes, ~9,375 API calls, ~$0.82)
python src/run_llm.py

# Analyze and visualize
python src/analyze_results.py
```

## File Structure
```
├── REPORT.md              # Full research report with results
├── README.md              # This file
├── planning.md            # Research plan and hypothesis decomposition
├── literature_review.md   # Pre-gathered literature review
├── resources.md           # Resource catalog
├── src/
│   ├── backtester.py      # Backtesting engine with configurable frequency
│   ├── strategies.py      # Trading strategies (baselines + LLM agent)
│   ├── run_baselines.py   # Run baseline experiments
│   ├── run_llm.py         # Run LLM experiments
│   └── analyze_results.py # Analysis and visualization
├── results/data/          # Raw experiment results (CSV, JSON)
├── figures/               # Generated visualizations
├── datasets/stock_prices/ # Yahoo Finance daily OHLCV data
├── papers/                # 19 downloaded research papers
└── code/                  # 5 cloned reference repositories
```

## Full Report
See [REPORT.md](REPORT.md) for the complete research report including methodology, statistical tests, per-stock results, and limitations.
