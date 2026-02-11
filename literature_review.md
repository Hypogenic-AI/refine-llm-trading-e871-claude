# Literature Review: Refining LLM Trading — Long-Term vs. Short-Term Decision Horizons

## Research Hypothesis
Large language model agents may perform better in financial trading when making longer-term trading decisions, as opposed to optimizing for short-term, day-to-day profit.

---

## 1. Research Area Overview

LLM-based financial trading agents have emerged as a rapidly growing research area since 2023. These systems leverage the natural language understanding, reasoning, and in-context learning capabilities of large language models to make investment decisions. The field spans a wide spectrum of approaches: from simple sentiment-driven signals to sophisticated multi-agent architectures that mimic institutional trading firms.

A critical but under-explored question in this domain is **how trading horizon affects LLM agent performance**. Most existing systems operate on daily buy/sell/hold decisions with automatic position liquidation each day ("single-step trading tasks"). However, recent evidence suggests this setup may fundamentally underutilize LLMs' strengths in semantic reasoning and long-term pattern recognition, while exposing their weaknesses in precise numerical optimization and high-frequency reaction.

---

## 2. Key Papers

### 2.1 Core Papers on Trading Horizons (Directly Relevant to Hypothesis)

#### FINSABER: Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?
- **Authors**: Li, Kim, Cucuringu, Ma (2025) — Accepted at KDD 2026
- **arXiv**: 2505.07078
- **Key Contribution**: First comprehensive benchmarking framework for LLM trading strategies with bias mitigation over 20 years of data and 100+ symbols
- **Methodology**: Rolling-window backtests comparing LLM agents (FinMem, FinAgent) against rule-based, ML, RL, and traditional strategies across S&P 500 constituents (including delisted stocks)
- **Critical Findings**:
  - LLM advantages reported in short-period evaluations **vanish under longer, broader testing**
  - Neither FinMem nor FinAgent generates statistically significant alpha (all p-values > 0.34)
  - LLMs are **too conservative in bull markets** (Sharpe 0.12 for FinAgent, -0.19 for FinMem) and **too aggressive in bear markets** (Sharpe -0.38 and -0.97)
  - Model complexity does not equate to market competence — ARIMA consistently outperforms LLM agents on risk-adjusted metrics
  - Buy-and-Hold significantly outperforms both LLM strategies under bias-mitigated evaluation (p < 0.001)
- **Relevance**: Directly challenges the premise of LLM trading superiority but also reveals the core problem is **regime-awareness**, not model capability — suggesting that adapting the trading horizon could be the key intervention

#### FinPos: A Position-Aware Trading Agent System for Real Financial Markets
- **Authors**: Liu, Dang (2025/2026)
- **arXiv**: 2510.27251
- **Key Contribution**: First LLM trading agent with explicit position management and multi-timescale reward design
- **Methodology**: Dual-agent architecture (Direction Decision Agent + Quantity/Risk Decision Agent) with rewards computed across 1-day, 7-day, and 30-day horizons
- **Critical Findings**:
  - Position-aware (longer-term) trading dramatically outperforms single-step (daily) trading: **62.15% CR on TSLA vs. negative returns for all baselines** including FinMem (-36.48%), FinAgent (-65.07%)
  - Multi-timescale reward is the architectural backbone — removing it drops CR below 20% for all stocks
  - **Performance peaks at 30-day reward horizon** — shorter windows (7-14 days) cause overreaction to noise; longer windows (60+ days) cause signal dilution
  - LLMs excel at "extracting long-term trends and underlying causal structures from complex semantic information rather than performing high-frequency precise numerical optimization"
- **Relevance**: **Most directly supports our hypothesis** — demonstrates that designing LLM agents for longer-term decision-making with position continuity unlocks dramatically better performance

#### QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading
- **Authors**: Xiong, Zhang, Feng, Sun, You (2025)
- **arXiv**: 2509.09995
- **Key Contribution**: First multi-agent LLM framework for high-frequency trading using only OHLC data
- **Methodology**: Four specialized agents (Indicator, Pattern, Trend, Risk) operating on 1-hour and 4-hour bars
- **Critical Findings**:
  - Achieves up to 80% directional accuracy at short horizons
  - Operates solely on price-derived signals, explicitly avoiding textual data which "typically lags price discovery"
  - Acknowledges that existing LLM frameworks are "ill-suited for the high-speed, precision-critical demands of HFT"
- **Relevance**: Provides the **short-term counterpoint** — shows LLMs can work at short horizons but require fundamentally different architectures (structured numerical signals rather than text reasoning)

#### Language Model Guided Reinforcement Learning in Quantitative Trading
- **Authors**: Darmanin, Vella (2025)
- **arXiv**: 2508.02366
- **Key Contribution**: Hybrid framework where LLMs generate high-level strategies to guide RL agents
- **Methodology**: LLMs provide strategic direction while RL handles tactical execution
- **Finding**: "Algorithmic trading requires short-term tactical decisions consistent with long-term financial objectives" — LLM guidance improves both Sharpe Ratio and Maximum Drawdown
- **Relevance**: Supports a **division of labor** where LLMs handle longer-term strategy and RL handles short-term execution

### 2.2 Benchmarks and Evaluation Frameworks

#### StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?
- **Authors**: Chen et al. (2025) — arXiv: 2510.02209
- **Methodology**: Multi-month sequential trading with daily signals (prices, fundamentals, news)
- **Finding**: Most LLM agents **struggle to outperform buy-and-hold**; "excelling at static financial knowledge tasks does not necessarily translate into successful trading strategies"
- **Relevance**: Confirms that daily-frequency LLM trading is difficult; suggests looking at alternative horizons

#### INVESTORBENCH: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent
- **Authors**: Li et al. (2024) — arXiv: 2412.18174
- **Key Contribution**: Standardized benchmark across stocks, crypto, and ETFs with 13 LLM backbones
- **Datasets**: Curated multi-modal datasets for financial decision-making
- **Relevance**: Provides reusable benchmark infrastructure

#### AI-Trader: Benchmarking Autonomous Agents in Real-Time Financial Markets
- **Authors**: Fan et al. (2025) — arXiv: 2512.10971
- **Key Finding**: "General intelligence does not automatically translate to effective trading capability" — tests across US stocks, A-shares, and crypto at multiple trading granularities
- **Relevance**: Multi-granularity testing enables direct comparison of performance at different frequencies

### 2.3 Multi-Agent Trading Architectures

#### FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory
- **Authors**: Yu et al. (2023) — 145 citations
- **Architecture**: Profile + Memory + Decision modules; layered memory with decay mechanisms
- **Datasets**: TSLA, NFLX, AMZN, MSFT, DOW30 (6-month evaluation period)
- **Limitation**: Operates under single-step daily trading; FINSABER showed this advantage vanishes under broader evaluation

#### TradingAgents: Multi-Agents LLM Financial Trading Framework
- **Authors**: Xiao et al. (2024) — arXiv: 2412.20138, 90 citations
- **Architecture**: Mimics institutional trading firms with analyst, researcher, trader, and risk management agents
- **Code**: https://github.com/TauricResearch/TradingAgents
- **Evaluation**: 3 months, 3 stocks — limited evaluation period

#### FinCon: A Synthesized LLM Multi-Agent System
- **Authors**: Yu et al. (2024) — arXiv: 2407.06567, 95 citations
- **Architecture**: Manager-analyst hierarchy with conceptual verbal reinforcement
- **Key Feature**: Self-critiquing mechanism to update investment beliefs episodically
- **Evaluation**: 8 months, 8 stocks

#### FLAG-Trader: Fusion LLM-Agent with Gradient-based RL
- **Authors**: Xiong et al. (2025) — arXiv: 2502.11433
- **Architecture**: LLM as policy network with parameter-efficient fine-tuning via trading reward gradients
- **Relevance**: Shows how RL optimization can adapt LLM behavior to financial domain

### 2.4 Surveys

#### Large Language Model Agent in Financial Trading: A Survey
- **Authors**: Ding et al. (2024) — arXiv: 2408.06361, 51 citations
- **Taxonomy**: LLM as Trader (news-driven, reflection-driven, debate-driven, RL-driven) vs. LLM as Alpha Miner
- **Data types**: Numerical (OHLCV), Textual (news, reports), Visual (charts), Simulated
- **Key insight**: GPT-3.5 used more than GPT-4 due to cost-effectiveness and latency

#### A Survey of LLMs for Financial Applications (Nie et al., 2024)
- 125 citations, covers linguistic tasks, sentiment analysis, time series, agent-based modeling

---

## 3. Common Methodologies

### Agent Architectures
- **Single-agent with memory/reflection**: FinMem, FinAgent — layered memory with decay, reflection modules
- **Multi-agent with role specialization**: TradingAgents, FinCon, QuantAgent HFT — agents in analyst/trader/risk roles
- **Hybrid LLM+RL**: FLAG-Trader, LM-guided RL — LLM provides strategy, RL handles execution
- **Position-aware with multi-timescale rewards**: FinPos — dual decision agents with 1/7/30-day reward horizons

### Trading Task Formulations
- **Single-step daily**: Most common; position auto-liquidated each day; actions = {Buy, Sell, Hold}
- **Position-aware continuous**: FinPos; position persists across days; agent manages exposure
- **High-frequency**: QuantAgent; 1h/4h bars; price-only signals
- **Portfolio-level**: MASS, AlphaAgents; multi-asset allocation decisions

### Common LLM Backbones
- GPT-4o (most common in 2024-2025 papers)
- GPT-3.5 (cost-effective, lower latency)
- Open-source: Qwen, LLaMA, DeepSeek-V3

---

## 4. Standard Baselines

| Baseline | Type | Description |
|----------|------|-------------|
| Buy-and-Hold | Passive | Purchase and hold for entire period |
| SMA Crossover | Rule-based | Moving average crossover signals |
| Bollinger Bands | Rule-based | Volatility-based bands |
| MACD / RSI | Rule-based | Momentum indicators |
| ARIMA | Predictor | Time-series forecasting |
| XGBoost | ML | Gradient boosting on features |
| A2C, PPO, SAC, TD3 | RL | Deep reinforcement learning agents |
| Random | Baseline | Random buy/sell decisions |

---

## 5. Evaluation Metrics

| Metric | Category | Description |
|--------|----------|-------------|
| Cumulative Return (CR) | Return | Total return over evaluation period |
| Annualized Return (AR) | Return | Annualized percentage return |
| Sharpe Ratio (SPR) | Risk-adjusted | Excess return per unit volatility |
| Sortino Ratio (STR) | Risk-adjusted | Return per unit downside risk |
| Maximum Drawdown (MDD) | Risk | Worst peak-to-trough loss |
| Annualized Volatility (AV) | Risk | Standard deviation of returns |
| Calmar Ratio | Risk-adjusted | Return per unit max drawdown |
| Alpha / Beta (CAPM) | Decomposition | Excess return vs. market exposure |

---

## 6. Datasets in the Literature

| Dataset | Used By | Type | Coverage |
|---------|---------|------|----------|
| Yahoo Finance OHLCV | FinPos, FinMem, FINSABER | Stock prices | 2000-present |
| Finnhub News API | FinPos, FinMem | Company/macro news | Real-time |
| SEC EDGAR (10-K, 10-Q) | FinPos, MarketSenseAI | Financial filings | 1993-present |
| FNSPID | FinRL-DeepSeek | Financial news time series | KDD '24 dataset |
| S&P 500 constituents (historical) | FINSABER | Index membership | 2000-2024 |
| Crypto OHLCV | QuantAgent, AI-Trader | Crypto prices | Various |
| Nasdaq futures | QuantAgent | Futures data | 1h/4h bars |

---

## 7. Gaps and Opportunities

### Gap 1: No Systematic Comparison of Trading Horizons
No existing work directly compares LLM agent performance across multiple trading horizons (daily, weekly, monthly) using the same architecture and data. Papers either operate at a single frequency or mention horizon tangentially.

### Gap 2: Position-Aware Trading is Underdeveloped
FinPos is the first paper to introduce position-aware trading for LLMs. All prior work uses single-step daily tasks that "fundamentally undermine continuous position management."

### Gap 3: Regime-Awareness is Missing
FINSABER shows LLMs are "pathologically miscalibrated" across market regimes. No work has studied whether longer holding periods naturally provide better regime alignment.

### Gap 4: Multi-Timescale Reward Design is Promising but Unexplored
FinPos shows that a 30-day reward horizon optimally balances noise vs. signal dilution. The sensitivity of performance to reward timescale deserves systematic investigation.

### Gap 5: Cost-Performance Tradeoff at Different Horizons
Longer-term strategies require fewer LLM API calls per unit time, potentially offering better cost-adjusted returns. This has not been studied.

---

## 8. Recommendations for Our Experiment

### Recommended Experimental Design
1. **Implement a single LLM trading agent** with configurable decision frequency (daily, weekly, monthly rebalancing)
2. **Test across multiple stocks** (at minimum: TSLA, AAPL, AMZN, NFLX, MSFT) to avoid survivorship bias
3. **Include both bull and bear market periods** (2020 COVID crash, 2022 bear, 2023-2024 bull)
4. **Compare against standard baselines**: Buy-and-Hold, ARIMA, moving average crossover

### Recommended Datasets
- **Primary**: Yahoo Finance daily OHLCV data (already downloaded, 2004-2025)
- **Supplementary**: Financial news from Finnhub for sentiment signals
- **For bias mitigation**: Include diverse stocks, not just FAANG winners

### Recommended Metrics
- Cumulative Return, Sharpe Ratio, Maximum Drawdown (standard trio)
- Annualized Return and Sortino Ratio (additional risk-adjusted)
- Trading frequency / turnover (to measure cost implications)
- Win rate at each time horizon

### Recommended Baselines
- Buy-and-Hold (passive benchmark)
- SMA Crossover (simple active strategy)
- ARIMA predictor (ML baseline)
- Same LLM agent at different frequencies (the key experimental variable)

### Methodological Considerations
- Use **rolling-window evaluation** to avoid data-snooping bias (per FINSABER methodology)
- Include **transaction costs** in all returns (per FINSABER: $0.0049/share)
- Evaluate across **market regimes** (bull/bear/sideways) separately
- Use at least **1 year** of test data to capture multiple market conditions
- Consider **position-aware vs. single-step** task formulation (per FinPos)
- Set LLM temperature to 0.7 (per FinPos) or test sensitivity
- Use **GPT-4o-mini** for cost efficiency in large-scale experiments

### Key Variables to Manipulate
1. **Trading frequency**: Daily, weekly (5-day), bi-weekly (10-day), monthly (21-day) rebalancing
2. **Information horizon**: How much historical context the LLM receives (1 week, 1 month, 3 months)
3. **Decision complexity**: Simple direction (buy/sell/hold) vs. position-sizing
4. **Data modality**: Price-only vs. price + news vs. price + news + filings
