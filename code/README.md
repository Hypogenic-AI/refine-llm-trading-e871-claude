# Cloned Research Repositories

This directory contains shallow clones of key repositories relevant to our research on
**refining LLM-based trading across different time horizons** (from high-frequency to
long-term investing). Each repository was cloned with `--depth 1` to conserve disk space.

---

## 1. FINSABER

| Field | Detail |
|-------|--------|
| **URL** | <https://github.com/waylonli/FINSABER> |
| **Paper** | "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?" (KDD 2026, arXiv 2505.07078) |
| **License** | Apache 2.0 |

### Purpose

FINSABER is a comprehensive **backtesting and benchmarking framework** designed to
evaluate trading strategies — both traditional (technical/ML) and LLM-based — over
**long time horizons** (up to 20 years of rolling-window backtests on S&P 500 data).
It directly addresses the central question of whether LLM traders can beat the market
in the long run.

### Key Files and Entry Points

| Path | Description |
|------|-------------|
| `backtest/run_baselines_exp.py` | Run non-LLM baseline strategies (SMA, Bollinger, Buy-and-Hold, XGBoost, etc.) |
| `backtest/run_llm_traders_exp.py` | Run LLM-based trading strategies (FinMem, FinAgent, FinCon) |
| `backtest/finsaber.py` / `finsaber_bt.py` | Core backtesting engine (wraps Backtrader) |
| `backtest/strategy/timing/` | Traditional timing strategies: `sma_crossover.py`, `bollinger_band.py`, `atr_band.py`, `xgboost_predictor.py`, `buy_and_hold.py`, `finrl.py`, etc. |
| `backtest/strategy/timing_llm/` | LLM timing strategies: `finmem.py`, `finagent.py`, `base_strategy_iso.py` |
| `backtest/strategy/selection/` | Ticker selection strategies for portfolio construction |
| `backtest/data_util/` | Data loaders (`backtest_dataset.py`, `finmem_dataset.py`) |
| `llm_traders/` | LLM agent implementations: `finagent/`, `finmem/`, `fincon_selector/` |
| `strats_configs/` | JSON/TOML configs for LLM strategy experiments |
| `backtest/results_aggregation.ipynb` | Jupyter notebook for aggregating and visualizing results |

### Dependencies

- Python 3.10, conda environment recommended
- Backtrader, pandas, numpy, scikit-learn, xgboost, torch, transformers
- OpenAI API key for LLM strategies
- Large dataset downloads from Google Drive (up to 10 GB for full S&P 500)
- Full list: `requirements.txt` (core) / `requirements-complete.txt` (all)

### Relevance to Our Research

**Directly central.** FINSABER is the primary benchmark for answering whether LLM
trading strategies degrade, hold steady, or improve as the investment horizon lengthens.
Its rolling-window backtesting infrastructure and head-to-head LLM-vs-baseline
comparisons are exactly the experimental framework we need. The `BaseStrategyIso`
and `BaseStrategy` classes provide clean extension points for plugging in new LLM
agent architectures at different time horizons.

---

## 2. TradingAgents

| Field | Detail |
|-------|--------|
| **URL** | <https://github.com/TauricResearch/TradingAgents> |
| **Paper** | "TradingAgents: Multi-Agents LLM Financial Trading Framework" (arXiv 2412.20138) |
| **License** | Apache 2.0 |

### Purpose

TradingAgents is a **multi-agent LLM framework** that mirrors the structure of a
real-world trading firm. It decomposes the trading decision into specialized roles —
analysts (fundamental, sentiment, news, technical), bull/bear researchers who debate,
a trader agent, and a risk management team — all orchestrated via LangGraph.

### Key Files and Entry Points

| Path | Description |
|------|-------------|
| `main.py` | Quick-start entry point: instantiate `TradingAgentsGraph`, call `.propagate(ticker, date)` |
| `tradingagents/graph/trading_graph.py` | Core LangGraph orchestration of the multi-agent pipeline |
| `tradingagents/agents/analysts/` | Four analyst agents: `fundamentals_analyst.py`, `market_analyst.py`, `news_analyst.py`, `social_media_analyst.py` |
| `tradingagents/agents/researchers/` | `bull_researcher.py`, `bear_researcher.py` — structured debate |
| `tradingagents/agents/trader/trader.py` | Final trading decision agent |
| `tradingagents/agents/managers/` | `research_manager.py`, `risk_manager.py` |
| `tradingagents/agents/risk_mgmt/` | Risk debate: `aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py` |
| `tradingagents/agents/utils/` | Shared state, tools, memory, technical indicator tools |
| `tradingagents/dataflows/` | Alpha Vantage + yfinance data fetching modules |
| `tradingagents/default_config.py` | Configuration (LLM provider, model names, debate rounds, data vendors) |
| `cli/main.py` | Interactive CLI for running analyses |
| `test.py` | Example test script |

### Dependencies

- Python 3.13, LangChain + LangGraph ecosystem
- Multi-provider LLM support: OpenAI, Google, Anthropic, xAI, OpenRouter, Ollama
- Alpha Vantage API key or yfinance for market data
- redis (optional, for caching)
- Full list: `requirements.txt`

### Relevance to Our Research

**High relevance for multi-agent architecture study.** TradingAgents demonstrates how
decomposing a trading decision across specialized LLM agents (with structured debate)
affects performance. This is a key comparison point for our research: does the
multi-agent approach scale differently across time horizons compared to single-agent
LLM traders? The configurable debate rounds and risk management layers also let us
study how deliberation depth interacts with trading frequency.

---

## 3. Stockagent (StockAgent)

| Field | Detail |
|-------|--------|
| **URL** | <https://github.com/MingyuJ666/Stockagent> |
| **Paper** | "When AI Meets Finance (StockAgent): Large Language Model-based Stock Trading in Simulated Real-world Environments" (arXiv 2407.18957) |
| **License** | Not specified |

### Purpose

StockAgent is a **multi-agent market simulation** where LLM-powered agents act as
individual investors trading in a simulated stock market. It investigates the impact of
external factors (macroeconomics, policy changes, company fundamentals, global events)
on LLM trading behavior. Importantly, it is designed to avoid test-set leakage by
preventing models from leveraging prior knowledge of test data.

### Key Files and Entry Points

| Path | Description |
|------|-------------|
| `main.py` | Entry point: `python main.py --model MODEL_NAME` starts a full simulation |
| `agent.py` | Core LLM agent implementation — trading logic, decision-making |
| `secretary.py` | "Secretary" agent that manages information flow and event handling |
| `stock.py` | Stock market simulation mechanics |
| `record.py` | Trade recording and logging |
| `util.py` | Utility functions |
| `prompt/agent_prompt.py` | Prompt templates for LLM agents |
| `log/custom_logger.py` | Logging infrastructure |

### Dependencies

- Python 3.9
- openai, pandas, tiktoken, requests, colorama
- Requires PromptCoder library: <https://github.com/dhh1995/PromptCoder>
- Supports GPT models and Gemini (gemini-pro as default)
- Minimal dependency footprint: `requirements.txt`

### Relevance to Our Research

**Valuable for behavioral analysis.** StockAgent provides a simulation environment where
we can study how LLM agents react to different market regimes and external shocks. Its
multi-phase architecture (Initial, Trading, Post-Trading, Special Events) maps naturally
to studying time-horizon effects: how do LLMs handle daily vs. quarterly information
releases? The test-set leakage prevention is also methodologically important for our
evaluation framework.

---

## 4. QuantAgent

| Field | Detail |
|-------|--------|
| **URL** | <https://github.com/Y-Research-SBU/QuantAgent> |
| **Paper** | "QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading" (arXiv 2509.09995) |
| **License** | MIT |

### Purpose

QuantAgent is a **multi-agent system for high-frequency trading (HFT)** analysis. It
uses four specialized agents — Indicator Agent (RSI, MACD, Stochastic), Pattern Agent
(chart pattern recognition via vision), Trend Agent (trend channel analysis), and
Decision Agent (synthesis) — built with LangChain/LangGraph. It targets short
timeframes (1-minute to daily) and includes benchmark data for multiple assets (BTC,
ES, NQ, SPX, DJI, etc.).

### Key Files and Entry Points

| Path | Description |
|------|-------------|
| `web_interface.py` | Flask web app entry point (`python web_interface.py`, serves at port 5000) |
| `trading_graph.py` | Core LangGraph trading analysis pipeline |
| `graph_setup.py` | Graph node/edge configuration |
| `graph_util.py` | Graph utility functions |
| `indicator_agent.py` | Technical indicator computation agent (RSI, MACD, Stochastic Oscillator) |
| `pattern_agent.py` | Visual chart pattern recognition agent (uses image input LLMs) |
| `trend_agent.py` | Trend channel analysis agent |
| `decision_agent.py` | Final decision synthesis agent (LONG/SHORT with entry, exit, stop-loss) |
| `default_config.py` | LLM configuration (provider, model, temperature) |
| `static_util.py` | Static analysis utility functions |
| `agent_state.py` | LangGraph agent state definitions |
| `benchmark/` | Pre-built benchmark datasets: `1h/`, `btc/`, `cl/`, `dji/`, `es/`, `nq/`, `qqq/`, `spx/`, `vix/` |
| `templates/` | Web interface HTML templates |

### Dependencies

- Python 3.11
- Flask, yfinance, matplotlib, mplfinance, scipy
- TA-Lib (requires conda install for C library)
- LangChain + LangGraph, OpenAI, Anthropic, or Qwen LLMs
- **Requires vision-capable LLMs** (agents analyze generated chart images)
- Full list: `requirements.txt`

### Relevance to Our Research

**Critical for the high-frequency end of the time-horizon spectrum.** QuantAgent
operates at 1-minute to hourly timeframes, making it the shortest-horizon system in
our collection. Its use of vision-based pattern recognition (chart images as LLM input)
is a distinctive approach that may behave very differently from text-only agents at
longer horizons. Comparing QuantAgent's HFT performance against FINSABER's long-term
backtests directly addresses our core research question about LLM trading effectiveness
across time horizons.

---

## 5. FinRL

| Field | Detail |
|-------|--------|
| **URL** | <https://github.com/AI4Finance-Foundation/FinRL> |
| **Papers** | Multiple (NeurIPS 2018/2020, ICAIF 2021, NeurIPS 2022, Springer Nature 2024) |
| **License** | MIT |

### Purpose

FinRL is the **first open-source framework for financial reinforcement learning**. It
provides a three-layer architecture (market environments, DRL agents, applications)
with a train-test-trade pipeline. It supports multiple RL algorithms (via
Stable-Baselines3, ElegantRL, RLlib), multiple data sources (15+), and multiple
financial applications (stock trading, portfolio allocation, crypto, HFT).

### Key Files and Entry Points

| Path | Description |
|------|-------------|
| `finrl/main.py` | Main entry point for the train-test-trade pipeline |
| `finrl/train.py` | Training script for DRL agents |
| `finrl/test.py` | Testing/evaluation script |
| `finrl/trade.py` | Live/paper trading script |
| `finrl/config.py` | Global configuration |
| `finrl/config_tickers.py` | Predefined ticker lists (DOW 30, S&P 500, NASDAQ, etc.) |
| `finrl/agents/` | DRL agent wrappers: ElegantRL, RLlib, Stable-Baselines3 |
| `finrl/meta/` | Market environments, data processors, preprocessing |
| `finrl/meta/data_processor.py` | Unified data processor for 15+ data sources |
| `finrl/applications/` | Ready-made applications: stock trading, portfolio allocation, crypto, HFT |
| `finrl/plot.py` | Visualization utilities |
| `examples/` | Jupyter notebooks: NeurIPS 2018 stock trading, ensemble strategies, paper trading, portfolio optimization |
| `unit_tests/` | Test suite |
| `docker/` | Docker containerization |

### Dependencies

- Python 3.6+
- Stable-Baselines3, ElegantRL, Ray/RLlib (DRL backends)
- gymnasium, numpy, pandas, matplotlib, scikit-learn
- Multiple data source APIs: yfinance, Alpaca, CCXT, WRDS, etc.
- TA-Lib for technical indicators
- torch/tensorflow (via DRL backends)
- Full list: `requirements.txt`

### Relevance to Our Research

**Essential baseline and infrastructure.** FinRL provides the RL-based trading baselines
that LLM approaches must be compared against. Its gym-style market environments can be
reused for LLM agent evaluation. The framework already spans multiple time horizons
(from HFT to daily portfolio allocation) and multiple asset classes, making it ideal
infrastructure for controlled horizon-comparison experiments. FINSABER already
integrates FinRL as one of its baseline strategies (`backtest/strategy/timing/finrl.py`),
confirming the complementary relationship.

---

## Summary: Time-Horizon Coverage

| Repository | Primary Time Horizon | Approach | Role in Our Research |
|------------|---------------------|----------|---------------------|
| **QuantAgent** | Minutes to hours (HFT) | Multi-agent LLM with vision | Shortest-horizon LLM trading |
| **TradingAgents** | Daily | Multi-agent LLM with debate | Mid-horizon multi-agent architecture |
| **Stockagent** | Daily (simulation) | Multi-agent LLM market sim | Behavioral analysis under market events |
| **FINSABER** | Months to years (rolling window) | Benchmarking framework | Long-horizon evaluation infrastructure |
| **FinRL** | Minutes to daily (configurable) | Deep reinforcement learning | Non-LLM baseline + shared environments |

Together, these repositories span the full spectrum from high-frequency (minutes) to
long-term (years), enabling systematic study of how LLM-based trading agents perform
across different investment horizons.

---

*Last updated: 2026-02-11*
*Cloned with `git clone --depth 1` (shallow clones, no full history)*
