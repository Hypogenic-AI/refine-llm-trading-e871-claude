# Resources Catalog: Refining LLM Trading

## Summary
This document catalogs all resources gathered for the research project investigating whether LLM trading agents perform better with longer-term trading decisions. Resources include 19 academic papers, stock price datasets for 7 major tickers, and 5 code repositories spanning the full spectrum from high-frequency to long-horizon trading.

---

## Papers
Total papers downloaded: 19

| # | Title | Authors | Year | File | Citations | Key Info |
|---|-------|---------|------|------|-----------|----------|
| 1 | FinMem: Performance-Enhanced LLM Trading Agent | Yu et al. | 2023 | papers/2311.13743v2_finmem_*.pdf | 145 | Layered memory architecture |
| 2 | FinAgent: Multimodal Foundation Agent for Trading | Zhang et al. | 2024 | papers/2402.18485_a_multimodal_*.pdf | 122 | Multimodal (text+charts) |
| 3 | FinCon: Multi-Agent System with Verbal RL | Yu et al. | 2024 | papers/2407.06567_fincon_*.pdf | 95 | Manager-analyst hierarchy |
| 4 | TradingAgents: Multi-Agent Framework | Xiao et al. | 2024 | papers/2412.20138_tradingagents_*.pdf | 90 | Institutional trading firm simulation |
| 5 | LLM Agent in Financial Trading: A Survey | Ding et al. | 2024 | papers/2408.06361_large_language_*.pdf | 51 | Comprehensive survey |
| 6 | INVESTORBENCH: Benchmark for Financial Decisions | Li et al. | 2024 | papers/2412.18174v1_investorbench_*.pdf | 26 | Multi-asset benchmark |
| 7 | FLAG-Trader: LLM-Agent with RL | Xiong et al. | 2025 | papers/2502.11433_flagtrader_*.pdf | 17 | LLM as RL policy network |
| 8 | StockBench: Can LLM Agents Trade Profitably? | Chen et al. | 2025 | papers/2510.02209_stockbench_*.pdf | 6 | Multi-month sequential trading |
| 9 | To Trade or Not to Trade | Emmanoulopoulos et al. | 2025 | papers/2507.08584_to_trade_*.pdf | 1 | Agentic stochastic DE |
| 10 | LiveTradeBench: Real-World Alpha | Yu et al. | 2025 | papers/2511.03628_livetradebench_*.pdf | 3 | Live trading benchmark |
| 11 | AI-Trader: Autonomous Agents Benchmark | Fan et al. | 2025 | papers/2512.10971_aitrader_*.pdf | 2 | Multi-market, multi-granularity |
| 12 | QuantAgents: Simulated Trading | Li et al. | 2025 | papers/2510.04643_quantagents_*.pdf | 3 | Long-term prediction via simulation |
| 13 | FINSABER: LLM Strategies in Long Run | Li et al. | 2025 | papers/2505.07078v4_can_llmbased_*.pdf | 7 | **KDD 2026** — 20-year backtest |
| 14 | FinPos: Position-Aware Trading Agent | Liu, Dang | 2025 | papers/2510.27251v2_finpos_*.pdf | 2 | **Multi-timescale rewards** |
| 15 | LM-Guided RL in Quant Trading | Darmanin, Vella | 2025 | papers/2508.02366_language_model_*.pdf | 1 | LLM strategy + RL execution |
| 16 | QuantAgent: HFT Multi-Agent LLMs | Xiong et al. | 2025 | papers/2509.09995_quantagent_*.pdf | 5 | 1h/4h bar trading |
| 17 | Autonomous Market Intelligence | Pu, Chen | 2026 | papers/2601.11958_autonomous_*.pdf | 0 | Agentic AI nowcasting |
| 18 | MarketSenseAI 2.0 | Fatouros et al. | 2025 | papers/2502.00415v2_marketsenseai_*.pdf | 11 | RAG + LLM agents |
| 19 | Behavioral Consistency of LLM Agents | Li et al. | 2026 | papers/2602.07023_behavioral_*.pdf | 0 | Trading-style switching |

See `papers/` directory for all PDFs.

---

## Datasets
Total datasets downloaded: 1 (stock prices with 7 tickers)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Stock OHLCV Data | Yahoo Finance | 34,837 rows (7 tickers) | Trading backtesting | datasets/stock_prices/ | Daily data 2004-2025 |

### Stock Price Coverage

| Ticker | Rows | Start Date | End Date |
|--------|------|------------|----------|
| TSLA | 3,901 | 2010-06-29 | 2025-12-30 |
| AAPL | 5,534 | 2004-01-02 | 2025-12-30 |
| AMZN | 5,534 | 2004-01-02 | 2025-12-30 |
| NFLX | 5,534 | 2004-01-02 | 2025-12-30 |
| MSFT | 5,534 | 2004-01-02 | 2025-12-30 |
| GOOG | 5,376 | 2004-08-19 | 2025-12-30 |
| META | 3,424 | 2012-05-18 | 2025-12-30 |

### Additional Datasets (Documented, Not Downloaded)

| Name | Source | Size | Notes |
|------|--------|------|-------|
| FNSPID | KDD '24 / HuggingFace | Large | Financial news time series dataset |
| Finnhub News | Finnhub API | Streaming | Company and macro news articles |
| SEC EDGAR Filings | SEC EDGAR API | Large | 10-K and 10-Q quarterly/annual reports |
| S&P 500 Historical Constituents | FINSABER repo | ~500 symbols | Including delisted stocks |

See `datasets/README.md` for download instructions for all datasets.

---

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Key Info |
|------|-----|---------|----------|----------|
| FINSABER | github.com/waylonli/FINSABER | 20-year backtest framework | code/FINSABER/ | KDD 2026, bias-mitigated evaluation |
| TradingAgents | github.com/TauricResearch/TradingAgents | Multi-agent trading firm | code/TradingAgents/ | LangGraph-based orchestration |
| StockAgent | github.com/MingyuJ666/Stockagent | Market simulation | code/Stockagent/ | Tests external factor impact |
| QuantAgent | github.com/Y-Research-SBU/QuantAgent | HFT multi-agent | code/QuantAgent/ | 1h/4h bars, price-only signals |
| FinRL | github.com/AI4Finance-Foundation/FinRL | Deep RL trading framework | code/FinRL/ | Foundation framework for RL trading |

See `code/README.md` for detailed documentation of each repository.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with query "LLM agents financial trading stock market" in diligent mode (87 papers found)
2. Supplementary search on "trading horizon time scale long-term short-term investment LLM"
3. Filtered by relevance score >= 2, prioritizing:
   - Papers directly addressing trading horizons
   - Comprehensive surveys and benchmarks
   - Highly-cited foundational systems (FinMem, FinCon, TradingAgents)
   - Recent 2025-2026 papers with novel methodologies

### Selection Criteria
- **Horizon-related papers**: Any paper discussing short-term vs. long-term trading, multi-timescale rewards, or position-aware trading
- **Benchmark papers**: Frameworks enabling fair comparison (FINSABER, StockBench, INVESTORBENCH)
- **Foundational systems**: Most-cited LLM trading agents that serve as baselines
- **Contrasting approaches**: Both HFT (QuantAgent) and long-horizon (FinPos, FINSABER) to cover the spectrum

### Challenges Encountered
- Many papers are behind proprietary model access (GPT-4o API costs)
- Several papers lack released code despite claiming reproducibility
- FNSPID and other large news datasets require API keys or are behind access controls
- Some arXiv papers had corpus IDs only (no direct arXiv links), requiring additional search

### Gaps and Workarounds
- **No direct horizon comparison paper exists**: This is precisely the gap our research fills
- **News datasets**: Documented download instructions rather than downloading large datasets
- **Live market data**: Documented API endpoints; real-time data not needed for backtesting experiments

---

## Recommendations for Experiment Design

Based on gathered resources, we recommend:

### 1. Primary Dataset
**Yahoo Finance daily OHLCV data** (already downloaded) for 7 tickers spanning 2004-2025. This provides:
- Multiple market regimes (2008 crisis, 2020 COVID, 2022 bear, 2023-2024 bull)
- Easy resampling to weekly/monthly for horizon comparison
- Direct comparability with FINSABER and FinPos results

### 2. Baseline Methods
From FINSABER and FinPos papers:
- **Buy-and-Hold** (passive benchmark — hardest to beat over long periods)
- **SMA Crossover** (simple active rule-based)
- **ARIMA** (statistical forecasting baseline)
- **Random** (lower bound)

### 3. Evaluation Metrics
Standard trio used across all surveyed papers:
- **Cumulative Return (CR%)** — overall profitability
- **Sharpe Ratio (SR)** — risk-adjusted return
- **Maximum Drawdown (MDD%)** — worst-case loss

### 4. Code to Adapt/Reuse
- **FINSABER** (`code/FINSABER/`): Rolling-window backtesting pipeline, bias-mitigation methodology, baseline implementations
- **FinRL** (`code/FinRL/`): Data download utilities, RL baselines, environment implementations
- **TradingAgents** (`code/TradingAgents/`): Multi-agent LLM architecture template

### 5. Experimental Variables
The key experiment should vary:
1. **Rebalancing frequency**: Daily / Weekly / Bi-weekly / Monthly
2. **Holding the LLM architecture constant** across frequencies
3. **Measuring performance across market regimes** (bull/bear/sideways per FINSABER methodology)
