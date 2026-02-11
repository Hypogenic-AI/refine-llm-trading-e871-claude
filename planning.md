# Research Plan: Do LLM Trading Agents Perform Better at Longer Decision Horizons?

## Motivation & Novelty Assessment

### Why This Research Matters
Current LLM trading agents (FinMem, FinAgent, TradingAgents) overwhelmingly operate on daily buy/sell/hold decisions, yet multiple papers (FINSABER, StockBench) show they consistently fail to outperform simple buy-and-hold. This represents a fundamental mismatch: LLMs excel at semantic reasoning and synthesizing complex information over time, not at precise numerical day-to-day predictions. If longer horizons unlock better LLM trading performance, it would redirect how the entire field builds financial AI agents.

### Gap in Existing Work
Per the literature review, **no existing work directly compares LLM agent performance across multiple trading horizons using the same architecture and data** (Gap 1). FinPos (2025) showed multi-timescale rewards improve performance and that 30-day horizons are optimal, but tested only one architecture. FINSABER (KDD 2026) showed daily LLM trading fails over 20 years but didn't test whether weekly/monthly rebalancing would help.

### Our Novel Contribution
We conduct the first controlled experiment varying **only the decision frequency** (daily, weekly, monthly) of a real LLM trading agent across multiple stocks and market regimes, while holding the agent architecture, prompts, and data sources constant. This isolates the effect of trading horizon on LLM performance.

### Experiment Justification
- **Experiment 1 (Baseline strategies)**: Establish non-LLM performance floors (Buy-and-Hold, SMA Crossover) at each frequency to separate horizon effects from LLM-specific effects.
- **Experiment 2 (LLM agent at varying horizons)**: Core test — same GPT-4.1-mini agent making daily vs. weekly vs. monthly decisions on the same stocks and time periods. Tests our hypothesis directly.
- **Experiment 3 (Market regime analysis)**: Decompose results by bull/bear/sideways regimes to understand *when* longer horizons help most.

## Research Question
Does an LLM trading agent achieve better risk-adjusted returns (Sharpe ratio, max drawdown) when making weekly or monthly trading decisions compared to daily decisions?

## Hypothesis Decomposition
- **H1**: LLM agents at weekly rebalancing frequency will achieve higher Sharpe ratios than at daily frequency.
- **H2**: LLM agents at monthly rebalancing frequency will achieve higher Sharpe ratios than at daily frequency.
- **H3**: The advantage of longer horizons will be most pronounced during volatile (bear) markets, where daily noise is highest.
- **H4**: Longer-horizon LLM strategies will have lower maximum drawdowns than daily strategies.

## Proposed Methodology

### Approach
We use a single LLM agent architecture with configurable rebalancing frequency. The agent receives recent price data and makes a trading decision (Buy/Sell/Hold) for each stock. We backtest across a 2-year evaluation period (2023-01-01 to 2024-12-31) covering both bull and bear conditions, using 5 stocks (AAPL, MSFT, AMZN, TSLA, NFLX) to avoid survivorship bias concerns.

We use a real LLM (GPT-4.1-mini via OpenAI API) to make actual trading decisions — no simulation of LLM behavior.

### Experimental Steps
1. Load and validate stock price data from datasets/stock_prices/
2. Implement backtesting engine with configurable rebalancing frequency
3. Implement baseline strategies (Buy-and-Hold, SMA Crossover, Random)
4. Implement LLM agent that calls GPT-4.1-mini with price context
5. Run baselines at all frequencies for control
6. Run LLM agent at daily, weekly, and monthly frequencies (3 runs per config for variance)
7. Compute metrics: Cumulative Return, Sharpe Ratio, Max Drawdown, Sortino Ratio
8. Statistical comparison across horizons (paired t-tests, effect sizes)
9. Market regime decomposition (bull/bear/sideways based on 50-day SMA)

### Baselines
- **Buy-and-Hold**: Purchase on day 1, hold throughout (hardest to beat)
- **SMA Crossover (20/50)**: Classic momentum strategy
- **Random**: Random buy/sell/hold each period (lower bound)

### Evaluation Metrics
- **Cumulative Return (CR%)**: Total return over period
- **Sharpe Ratio (SR)**: Risk-adjusted return (annualized, rf=0)
- **Maximum Drawdown (MDD%)**: Worst peak-to-trough decline
- **Sortino Ratio**: Downside risk-adjusted return
- **Win Rate**: % of profitable trades
- **Number of trades**: Cost/activity comparison

### Statistical Analysis Plan
- Paired comparisons across stocks for each horizon pair (daily vs. weekly, daily vs. monthly)
- Wilcoxon signed-rank test (non-parametric, small sample)
- Effect sizes (Cohen's d)
- Confidence intervals via bootstrap
- Significance level: α = 0.05

## Expected Outcomes
- Weekly/monthly LLM agents should outperform daily on Sharpe ratio (per FinPos findings)
- Daily LLM agents may show higher turnover and worse drawdowns
- Buy-and-Hold may still beat all LLM strategies (per FINSABER), but the gap should narrow at longer horizons
- Bear market periods should show the largest horizon effect

## Timeline and Milestones
1. Environment setup + data validation: 10 min
2. Backtesting engine + baselines: 30 min
3. LLM agent implementation: 30 min
4. Run experiments: 60 min (API calls)
5. Analysis + visualization: 30 min
6. Documentation: 20 min

## Potential Challenges
- API rate limits: Use GPT-4.1-mini for cost efficiency; implement retry logic
- Variance in LLM outputs: Run 3 seeds per configuration
- Small sample size (5 stocks): Use non-parametric tests, report per-stock results
- Market regime classification: Use simple SMA-based approach to avoid overfitting

## Success Criteria
- Complete experiments across all 3 horizons × 5 stocks × 3 runs
- Statistical test results for H1-H4
- Clear visualization comparing horizons
- Honest assessment of whether hypothesis is supported
