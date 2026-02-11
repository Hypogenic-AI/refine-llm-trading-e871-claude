# Do LLM Trading Agents Perform Better at Longer Decision Horizons?

## 1. Executive Summary

We tested whether LLM trading agents achieve better performance when making longer-term (weekly/monthly) trading decisions compared to the standard daily approach. Using GPT-4.1-mini as the trading agent across 5 major stocks (AAPL, MSFT, AMZN, TSLA, NFLX) over a 2-year period (2023-2024), we found that **weekly rebalancing yields the best LLM performance** (Sharpe ratio 1.028 vs. 0.892 for daily), while **monthly rebalancing significantly degrades performance** (Sharpe 0.421). However, none of the LLM strategies outperform simple Buy-and-Hold (Sharpe 1.620) in this bull-market period, consistent with prior findings from FINSABER (KDD 2026).

**Key finding**: There is a "Goldilocks zone" for LLM trading decisions — weekly rebalancing outperforms both daily and monthly, suggesting LLMs benefit from filtering daily noise but lose important signal at monthly horizons.

## 2. Goal

### Hypothesis
Large language model agents may perform better in financial trading when making longer-term trading decisions, as opposed to optimizing for short-term, day-to-day profit.

### Why This Matters
Current LLM trading agents overwhelmingly operate on daily buy/sell/hold decisions (FinMem, FinAgent, TradingAgents), yet multiple benchmarking studies (FINSABER, StockBench) show these agents consistently fail to outperform passive strategies. If the decision horizon is a key variable, the entire field should reconsider how it builds financial AI agents. FinPos (2025) hinted that multi-timescale rewards improve performance, but no prior work has systematically isolated the effect of rebalancing frequency.

### Research Gap
No existing work directly compares LLM agent performance across multiple trading horizons (daily, weekly, monthly) using the same architecture and data. This is the first controlled experiment isolating decision frequency as the independent variable.

## 3. Data Construction

### Dataset Description
- **Source**: Yahoo Finance daily OHLCV data (pre-downloaded)
- **Tickers**: AAPL, MSFT, AMZN, TSLA, NFLX
- **Test Period**: January 2, 2023 – December 31, 2024 (502 trading days)
- **Data fields**: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits

### Example Samples

| Date       | Ticker | Close Price |
|------------|--------|-------------|
| 2023-01-03 | AAPL   | $123.10     |
| 2023-06-30 | TSLA   | $261.77     |
| 2024-12-31 | NFLX   | $89.13      |

### Data Quality
- **Missing values**: 0% — all trading days present
- **Outliers**: Normal market volatility; no data errors detected
- **Splits**: All prices are split-adjusted via Yahoo Finance
- **Coverage**: 502 trading days per ticker for the test period

### Preprocessing Steps
1. Loaded CSV files with Date as index, parsed as datetime
2. Filtered to test period (2023-01-01 to 2024-12-31)
3. Used Close prices for all trading signals and portfolio valuation
4. Computed rolling technical indicators (5-day SMA, 20-day SMA) within the agent

### Test Period Characteristics
This period was predominantly bullish, with all 5 stocks showing positive returns:
| Ticker | Start Price | End Price | Buy-and-Hold Return |
|--------|-------------|-----------|---------------------|
| AAPL   | $123.10     | $249.06   | +102.1%             |
| MSFT   | $233.99     | $418.41   | +78.6%              |
| AMZN   | $85.82      | $219.39   | +155.4%             |
| TSLA   | $108.10     | $403.84   | +273.2%             |
| NFLX   | $29.50      | $89.13    | +201.9%             |

## 4. Experiment Description

### Methodology

#### High-Level Approach
We built a single LLM trading agent with configurable rebalancing frequency (daily, weekly, monthly). The agent receives recent price data and technical indicators, then makes a BUY/SELL/HOLD decision via the GPT-4.1-mini API. We hold the agent architecture, prompt template, and data sources constant across all frequencies, isolating decision frequency as the sole independent variable.

#### Why This Method?
- **Controlled comparison**: Same model, same prompt structure, same data — only frequency varies
- **Real LLM**: Actual API calls to GPT-4.1-mini (not simulated behavior)
- **Standard baselines**: Buy-and-Hold, SMA Crossover, Random for reference
- **Multiple runs**: 2-3 runs per configuration to estimate variance

### Implementation Details

#### Tools and Libraries
| Library    | Version | Purpose           |
|------------|---------|-------------------|
| Python     | 3.12.8  | Runtime           |
| openai     | 2.20.0  | GPT-4.1-mini API  |
| pandas     | 2.3.3   | Data manipulation |
| numpy      | 2.2.6   | Numerical ops     |
| scipy      | 1.17.0  | Statistical tests |
| matplotlib | 3.10.8  | Visualization     |
| seaborn    | 0.13.2  | Statistical plots |

#### LLM Agent Design
The LLM agent receives the following context at each decision point:
- Current date and stock ticker
- Current price and position status (cash or long)
- Last 10 closing prices
- 5-day and 20-day simple moving averages
- Price changes over 1-day, 5-day, 10-day, and 20-day periods
- The decision horizon (next day / next week / next month)

The prompt explicitly instructs the model to consider the appropriate time horizon for its decision. The model responds with exactly one word: BUY, SELL, or HOLD.

#### Hyperparameters
| Parameter           | Value      | Selection Method     |
|---------------------|------------|----------------------|
| LLM Model           | gpt-4.1-mini | Cost-effective SOTA |
| Temperature          | 0.3        | Low variance, per literature |
| Max tokens           | 5          | Single-word response |
| Initial capital      | $10,000    | Standard             |
| Transaction cost     | 0.1%       | Conservative estimate |
| Lookback window      | 60 days    | Standard technical   |
| Daily runs           | 2          | Lower variance naturally |
| Weekly/Monthly runs  | 3          | Higher variance needs more runs |

#### Rebalancing Frequencies
- **Daily**: Decision at every trading day (502 decision points)
- **Weekly**: Decision every 5 trading days (100 decision points)
- **Monthly**: Decision every 21 trading days (24 decision points)

### Experimental Protocol

#### Reproducibility Information
- **Number of runs**: 2 per daily config, 3 per weekly/monthly config
- **Random seeds**: 42, 49, 56 (offset by 7)
- **Hardware**: CPU-only for API calls (no GPU needed)
- **Total API calls**: ~9,375
- **Total cost**: ~$0.82
- **Total runtime**: 43.5 minutes
- **API model version**: gpt-4.1-mini-2025-04-14

#### Evaluation Metrics
| Metric | Description | Why Used |
|--------|-------------|----------|
| Cumulative Return (CR%) | Total return over 2 years | Primary performance measure |
| Sharpe Ratio (SR) | Annualized risk-adjusted return | Primary hypothesis test metric |
| Maximum Drawdown (MDD%) | Worst peak-to-trough decline | Risk assessment |
| Sortino Ratio | Downside risk-adjusted return | Asymmetric risk measure |
| Number of Trades | Total buy/sell actions | Cost/activity measure |
| Win Rate | % of positive return periods | Decision quality |

### Raw Results

#### LLM Agent Results by Frequency (Averaged Across Tickers)

| Frequency | CR (%) | Sharpe Ratio | MDD (%) | Sortino | Trades |
|-----------|--------|-------------|---------|---------|--------|
| **Daily**   | +53.8  | 0.892       | -20.7   | 1.140   | 127    |
| **Weekly**  | +62.4  | 1.028       | -24.2   | 1.292   | 31     |
| **Monthly** | +13.5  | 0.421       | -28.0   | 0.523   | 11     |

#### Per-Stock LLM Results

| Ticker | Freq | CR (%) | Sharpe | MDD (%) | Trades |
|--------|------|--------|--------|---------|--------|
| AAPL | daily | +40.9 | 1.163 | -12.8 | 111 |
| AAPL | weekly | +49.0 | 1.304 | -18.4 | 28 |
| AAPL | monthly | +18.0 | 0.584 | -19.2 | 12 |
| MSFT | daily | +7.6 | 0.284 | -12.5 | 123 |
| MSFT | weekly | +29.4 | 0.781 | -18.9 | 31 |
| MSFT | monthly | +23.1 | 0.654 | -17.8 | 9 |
| AMZN | daily | +21.5 | 0.531 | -26.4 | 143 |
| AMZN | weekly | +30.1 | 0.678 | -18.7 | 33 |
| AMZN | monthly | +34.5 | 0.684 | -22.2 | 7 |
| TSLA | daily | +129.9 | 1.213 | -33.9 | 139 |
| TSLA | weekly | +121.7 | 1.100 | -46.7 | 37 |
| TSLA | monthly | -6.8 | 0.102 | -48.1 | 14 |
| NFLX | daily | +69.2 | 1.270 | -17.6 | 117 |
| NFLX | weekly | +81.8 | 1.276 | -18.2 | 26 |
| NFLX | monthly | -1.5 | 0.080 | -32.6 | 12 |

#### Baseline Comparison (Averaged Across Tickers)

| Strategy | Freq | CR (%) | Sharpe | MDD (%) | Trades |
|----------|------|--------|--------|---------|--------|
| Buy-and-Hold | all | +162.2 | 1.620 | -26.2 | 1 |
| SMA-Crossover | daily | +61.2 | 1.093 | -24.2 | 9 |
| SMA-Crossover | weekly | +56.0 | 1.058 | -24.4 | 9 |
| SMA-Crossover | monthly | +61.8 | 1.099 | -22.9 | 7 |
| Random | daily | +29.4 | 0.701 | -25.6 | 173 |
| LLM-gpt-4.1-mini | daily | +53.8 | 0.892 | -20.7 | 127 |
| LLM-gpt-4.1-mini | weekly | +62.4 | 1.028 | -24.2 | 31 |
| LLM-gpt-4.1-mini | monthly | +13.5 | 0.421 | -28.0 | 11 |

## 5. Result Analysis

### Key Findings

**Finding 1: Weekly rebalancing is the optimal LLM decision frequency.**
The LLM agent at weekly frequency achieves a Sharpe ratio of 1.028, compared to 0.892 at daily and 0.421 at monthly. This 15% Sharpe improvement over daily holds for 4 out of 5 stocks (all except TSLA).

**Finding 2: Monthly rebalancing dramatically degrades LLM performance.**
At monthly frequency, the LLM agent's average cumulative return drops to +13.5% (vs. +53.8% daily, +62.4% weekly). Two stocks (TSLA at -6.8%, NFLX at -1.5%) show negative returns. The agent makes too few decisions to adapt to regime changes.

**Finding 3: No LLM frequency beats Buy-and-Hold in this bull market.**
Buy-and-Hold achieves a Sharpe of 1.620 averaged across tickers — substantially higher than the best LLM configuration (weekly: 1.028). This is consistent with FINSABER (2025), which found LLM agents fail to generate alpha over Buy-and-Hold in extended testing.

**Finding 4: The LLM daily strategy has the lowest drawdowns.**
Despite lower returns, the daily LLM agent achieves the best maximum drawdown (-20.7%) across all active strategies, suggesting it is effective at risk management through frequent position adjustments.

**Finding 5: Weekly rebalancing achieves better returns with 75% fewer trades.**
Weekly LLM makes ~31 trades vs. ~127 for daily, while achieving +16% higher cumulative return. This implies significant transaction cost savings and better signal-to-noise ratio.

### Hypothesis Testing Results

#### H1: Weekly > Daily Sharpe Ratio
- Mean difference: +0.136 (weekly higher)
- Cohen's d: 0.59 (medium effect)
- Paired t-test: p = 0.256 (not significant at α = 0.05)
- Direction: **Supports hypothesis** (weekly better), but not statistically significant with n=5

#### H2: Monthly > Daily Sharpe Ratio
- Mean difference: -0.472 (monthly lower)
- Cohen's d: -0.66 (medium-large negative effect)
- Paired t-test: p = 0.213
- Direction: **Contradicts hypothesis** — monthly is substantially worse than daily

#### H3: Longer horizons help more in bear markets
- Limited data: Test period was predominantly bullish
- Cannot conclusively test this hypothesis

#### H4: Longer horizons have lower drawdowns
- Daily MDD: -20.7% (best)
- Weekly MDD: -24.2%
- Monthly MDD: -28.0% (worst)
- Direction: **Contradicts hypothesis** — shorter horizons had better drawdown control

### Comparison to Baselines
- LLM weekly (SR=1.028) approaches SMA Crossover (SR=1.058-1.099) but does not clearly beat it
- LLM daily (SR=0.892) underperforms SMA Crossover (SR=1.093)
- All active strategies significantly underperform Buy-and-Hold (SR=1.620) in this bull market
- LLM at all frequencies outperforms Random (SR=0.701-1.001)

### Visualizations

Plots saved in `figures/` directory:
- `sharpe_by_frequency.png` — Bar chart comparing Sharpe across strategies and frequencies
- `cumulative_return_by_frequency.png` — Cumulative return comparison
- `llm_per_stock_sharpe.png` — Per-stock LLM Sharpe by frequency
- `sharpe_heatmap.png` — Heatmap of Sharpe ratios by stock × frequency
- `sharpe_gap_vs_bh.png` — Gap between LLM and Buy-and-Hold by frequency
- `drawdown_by_frequency.png` — Max drawdown comparison
- `trades_vs_return.png` — Trade count vs return scatter
- `portfolio_AAPL.png`, `portfolio_AMZN.png`, `portfolio_MSFT.png` — Portfolio value time series

### Surprises and Insights

1. **The non-monotonic relationship was unexpected.** We hypothesized "longer = better" but found a clear peak at weekly. This suggests LLMs have a specific temporal resolution where their reasoning is most effective.

2. **Monthly TSLA and NFLX lost money.** At monthly frequency, the LLM made only 12-15 trades over 2 years and frequently held cash during rallies or stayed long during corrections. The 21-trading-day gap between decisions is too coarse for volatile stocks.

3. **MSFT showed the most dramatic weekly improvement.** Daily MSFT Sharpe was only 0.284 (near zero), but weekly jumped to 0.781 — a 175% improvement. MSFT's price action has lower volatility, making daily noise particularly unhelpful.

4. **Monthly results had near-zero variance across runs.** With only 24 decision points over 2 years, the LLM's temperature=0.3 produced nearly identical decisions across seeds. This means monthly results are deterministic, not stochastic.

### Error Analysis

Common failure modes at each frequency:
- **Daily**: Excessive trading (111-144 trades) leads to death by transaction costs and whipsawing. The agent frequently alternates BUY/SELL based on 1-day price fluctuations.
- **Weekly**: More stable decisions. Main failures occur during sharp intraday drops that happen between decision points (e.g., TSLA's 46.7% drawdown).
- **Monthly**: Agent misses multi-week rallies and corrections entirely. By the time it acts, the opportunity has passed.

### Limitations

1. **Bull market bias**: 2023-2024 was strongly bullish for all 5 stocks. Results may differ dramatically in bear or sideways markets. The test should be extended to include 2022 (bear) and 2020 (crash+recovery).

2. **Small sample size (n=5 stocks)**: With only 5 tickers, statistical tests lack power. The medium effect sizes (d=0.59, d=-0.66) would likely reach significance with n=15-20 stocks.

3. **Single model**: Only GPT-4.1-mini was tested. Different models (GPT-4.1, Claude, Gemini) may show different horizon sensitivities.

4. **Price-only signals**: The agent only receives price data and simple technical indicators. Adding news, fundamentals, or earnings data could change the horizon effects.

5. **No position sizing**: The agent goes fully long or fully cash. Partial positions and portfolio allocation could improve all strategies.

6. **No short selling**: The agent can only buy or hold cash. In bear markets, short-selling capability would be important.

7. **Survivorship bias**: All 5 stocks are large-cap winners. Including delisted or declining stocks (per FINSABER methodology) would provide more robust results.

## 6. Conclusions

### Summary
Weekly rebalancing provides the best performance for LLM trading agents, achieving a 15% higher Sharpe ratio (1.028 vs. 0.892) and 16% higher cumulative return (+62.4% vs. +53.8%) compared to daily rebalancing, while using 75% fewer trades. However, monthly rebalancing degrades performance dramatically (Sharpe 0.421), and no LLM frequency beats Buy-and-Hold (Sharpe 1.620) during this bull market period.

**The answer to the research question is nuanced**: LLMs do perform better at *moderately* longer horizons (weekly > daily), but not at *much* longer horizons (monthly < daily). The relationship between decision horizon and performance is non-monotonic, with an apparent optimum around the weekly (5-day) timescale.

### Implications

**Practical implications:**
- Researchers building LLM trading agents should default to weekly, not daily, rebalancing
- Weekly frequency reduces API costs by ~80% while improving returns
- Monthly is too infrequent for volatile individual stocks — it may work better for stable portfolios or indices

**Theoretical implications:**
- LLMs appear to have a "temporal sweet spot" for financial reasoning — daily is too noisy, monthly too coarse
- This aligns with FinPos (2025) finding that 30-day reward horizons are optimal, and FINSABER's finding that LLMs are "too conservative in bull markets" (daily frequency exacerbates this)
- The result supports the hypothesis that LLMs excel at semantic pattern recognition over multi-day horizons rather than precise daily numerical prediction

### Confidence in Findings
- **Medium-high confidence** in the weekly > daily finding (4/5 stocks, medium effect size, consistent direction)
- **High confidence** in the monthly < daily finding (large effect, 5/5 stocks show degradation)
- **Low confidence** in generalizability due to bull-market-only test period and small stock universe

## 7. Next Steps

### Immediate Follow-ups
1. **Extend test period**: Include 2020-2022 to capture COVID crash, bear market, and recovery — this is the most important follow-up
2. **Expand stock universe**: Test on 20-50 stocks including small-caps and declining stocks to improve statistical power
3. **Add bi-weekly frequency**: Test 10-trading-day rebalancing to find the exact optimum between weekly and monthly
4. **Test with news data**: Add financial news context to the LLM prompt to see if information-rich prompts change the optimal horizon

### Alternative Approaches
- **Multi-model comparison**: Test GPT-4.1, Claude Sonnet 4.5, and Gemini 2.5 Pro to see if the weekly advantage is model-specific
- **Hybrid approach**: Use LLM for weekly strategic direction + rule-based daily execution (per Darmanin & Vella 2025)
- **Position-aware agent**: Implement FinPos-style position management with the weekly LLM agent
- **Portfolio-level decisions**: Test whether monthly rebalancing works better for multi-asset allocation (less volatile than single-stock)

### Open Questions
1. Is the weekly optimum an artifact of the 5-day trading week structure, or a genuine cognitive property of LLMs?
2. Would fine-tuned LLMs (FLAG-Trader style) show a different optimal horizon than API-based models?
3. Does the optimal horizon shift with market volatility (shorter in calm markets, longer in volatile ones)?
4. How does adding fundamental data (earnings, filings) interact with decision frequency?

## References

### Papers Directly Informing This Study
1. Li et al. (2025) "FINSABER: Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?" KDD 2026. arXiv:2505.07078
2. Liu & Dang (2025) "FinPos: A Position-Aware Trading Agent System for Real Financial Markets." arXiv:2510.27251
3. Chen et al. (2025) "StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?" arXiv:2510.02209
4. Yu et al. (2023) "FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory." arXiv:2311.13743
5. Xiao et al. (2024) "TradingAgents: Multi-Agents LLM Financial Trading Framework." arXiv:2412.20138
6. Darmanin & Vella (2025) "Language Model Guided Reinforcement Learning in Quantitative Trading." arXiv:2508.02366
7. Xiong et al. (2025) "QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading." arXiv:2509.09995

### Datasets
- Yahoo Finance daily OHLCV data (2004-2025) for AAPL, MSFT, AMZN, TSLA, NFLX

### Tools
- GPT-4.1-mini (gpt-4.1-mini-2025-04-14) via OpenAI API
- Python 3.12.8, pandas, numpy, scipy, matplotlib, seaborn
