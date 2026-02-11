# Datasets for LLM Trading Agent Experiments

This directory contains (or documents how to obtain) the datasets used in our
research on LLM-based trading agents across multiple time horizons.

## Research Context

Our hypothesis is that LLM agents perform better at **longer-term** trading
decisions (weekly/monthly) compared to daily trading, because LLMs excel at
reasoning over qualitative information (news, filings, macro trends) rather
than short-term price patterns.

We need data that supports experiments at three rebalancing frequencies:
- **Daily** — baseline, mostly price-driven
- **Weekly** — moderate horizon, can incorporate weekly news summaries
- **Monthly** — longer horizon, can incorporate earnings reports and SEC filings

---

## 1. Stock Price Data (OHLCV) — Yahoo Finance

**Status:** Available via download script
**Tickers:** TSLA, AAPL, AMZN, NFLX, MSFT, GOOG, META
**Period:** 2004–2025 (varies by ticker IPO date)
**Granularity:** Daily OHLCV (Open, High, Low, Close, Volume)

### How to download

```bash
source .venv/bin/activate
python code/download_stock_data.py
```

This creates:
- `datasets/stock_prices/{TICKER}.csv` — one file per ticker
- `datasets/stock_prices/all_stocks.csv` — combined long-format file

### Resampling to weekly/monthly

Daily data can be resampled for longer-horizon experiments:

```python
import pandas as pd

df = pd.read_csv("datasets/stock_prices/AAPL.csv", parse_dates=["Date"])
df = df.set_index("Date")

# Weekly OHLCV
weekly = df.resample("W").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
})

# Monthly OHLCV
monthly = df.resample("ME").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
})
```

### References
- [yfinance documentation](https://github.com/ranaroussi/yfinance)
- Used in: FinRL, FinGPT, and most LLM trading papers as baseline price data

---

## 2. FNSPID — Financial News and Stock Price Integration Dataset

**Status:** Manual download required (large dataset)
**Source:** [Hugging Face — Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
**Paper:** "FNSPID: A Comprehensive Financial News Dataset in Time Series" (2024)
**Size:** ~15 GB (29.7M news records, 2009–2023)

### Description

FNSPID pairs financial news articles with corresponding stock price movements.
It is used by FinRL-DeepSeek, FinGPT, and other LLM trading frameworks. The
dataset contains:
- News headlines and article text
- Publication timestamps aligned with trading days
- Stock ticker associations
- Sentiment labels (in some versions)

### How to download

```bash
# Option 1: Hugging Face CLI
pip install huggingface_hub
huggingface-cli download Zihan1004/FNSPID --local-dir datasets/fnspid/

# Option 2: Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Zihan1004/FNSPID",
                  repo_type="dataset",
                  local_dir="datasets/fnspid/")
```

### References
- Zihan Dong et al., "FNSPID: A Comprehensive Financial News Dataset in Time
  Series" (2024)
- Used in: FinRL-DeepSeek, FinGPT research

---

## 3. Financial News — Finnhub API

**Status:** Requires API key (free tier available)
**Source:** [Finnhub.io](https://finnhub.io/)
**Coverage:** Real-time and historical news for US stocks

### Description

Finnhub provides company-level financial news with:
- Headlines and summaries
- Source attribution
- Sentiment scores (via their NLP pipeline)
- Ticker associations

Free tier allows 60 API calls/minute, which is sufficient for historical
research on a small number of tickers.

### How to obtain

1. Register at [finnhub.io](https://finnhub.io/) for a free API key
2. Set environment variable: `export FINNHUB_API_KEY=your_key`
3. Use the Finnhub Python client:

```bash
pip install finnhub-python
```

```python
import finnhub
import os

client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])
news = client.company_news("AAPL", _from="2024-01-01", to="2024-12-31")
```

### References
- Used in: FinGPT, multiple LLM trading agent papers
- [Finnhub API docs](https://finnhub.io/docs/api)

---

## 4. SEC Filings (10-K, 10-Q) — EDGAR

**Status:** Publicly available, no API key needed
**Source:** [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch)
**Coverage:** All US public companies

### Description

SEC filings provide the fundamental qualitative data that LLMs can reason
about for longer-term trading decisions:
- **10-K** — Annual reports (risk factors, MD&A, financial statements)
- **10-Q** — Quarterly reports
- **8-K** — Material event disclosures

### How to obtain

```bash
pip install sec-edgar-downloader
```

```python
from sec_edgar_downloader import Downloader

dl = Downloader("MyCompany", "email@example.com",
                "datasets/sec_filings/")
dl.get("10-K", "AAPL", after="2020-01-01", before="2025-01-01")
dl.get("10-Q", "AAPL", after="2020-01-01", before="2025-01-01")
```

### References
- Used in: FinanceBench, various LLM financial reasoning papers
- [SEC EDGAR full-text search](https://efts.sec.gov/LATEST/search-index?q=%22artificial+intelligence%22)

---

## 5. Additional Datasets (for future work)

| Dataset | Source | Description |
|---------|--------|-------------|
| **StockNet** | [GitHub](https://github.com/yumoxu/stocknet-dataset) | Tweet + price dataset for stock movement prediction |
| **EDT** | [GitHub](https://github.com/Zhihan1996/TradeTheEvent) | Earnings call transcripts + stock returns |
| **FinanceBench** | [HuggingFace](https://huggingface.co/datasets/PatronusAI/financebench) | Q&A benchmark over SEC filings |
| **AlphaFin** | Paper | Multi-source financial dataset for LLM evaluation |

---

## Data Directory Structure

```
datasets/
├── README.md              ← this file
├── .gitignore             ← excludes large data files from git
├── stock_prices/          ← OHLCV from Yahoo Finance
│   ├── AAPL.csv
│   ├── AMZN.csv
│   ├── GOOG.csv
│   ├── META.csv
│   ├── MSFT.csv
│   ├── NFLX.csv
│   ├── TSLA.csv
│   └── all_stocks.csv
├── fnspid/                ← FNSPID dataset (download separately)
├── news/                  ← Finnhub news dumps
└── sec_filings/           ← SEC EDGAR filings
```
