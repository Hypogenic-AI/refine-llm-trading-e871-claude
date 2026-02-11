#!/usr/bin/env python3
"""
Download OHLCV stock price data from Yahoo Finance.

Downloads daily price data for a set of tickers used in LLM trading agent
research. The data supports experiments at multiple time horizons:
  - Daily rebalancing
  - Weekly rebalancing (resample from daily)
  - Monthly rebalancing (resample from daily)

Usage:
    python code/download_stock_data.py

Output:
    datasets/stock_prices/{TICKER}.csv   -- one file per ticker
    datasets/stock_prices/all_stocks.csv -- combined long-format file
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────────
TICKERS = ["TSLA", "AAPL", "AMZN", "NFLX", "MSFT", "GOOG", "META"]
START_DATE = "2004-01-01"
END_DATE = "2025-12-31"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "stock_prices"


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker."""
    print(f"  Downloading {ticker} ({start} to {end}) ...")
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        print(f"  WARNING: No data returned for {ticker}")
        return df

    # Clean up columns: keep standard OHLCV + Dividends/Stock Splits
    df.index.name = "Date"
    df = df.reset_index()

    # Ensure Date is date-only (no timezone)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date

    print(f"  {ticker}: {len(df)} rows, {df['Date'].min()} to {df['Date'].max()}")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []
    summary_rows = []

    for ticker in TICKERS:
        df = download_ticker(ticker, START_DATE, END_DATE)
        if df.empty:
            continue

        # Save individual CSV
        out_path = OUTPUT_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")

        # Track for combined file
        df_copy = df.copy()
        df_copy.insert(0, "Ticker", ticker)
        all_frames.append(df_copy)

        summary_rows.append({
            "Ticker": ticker,
            "Rows": len(df),
            "Start": str(df["Date"].min()),
            "End": str(df["Date"].max()),
            "File": f"{ticker}.csv",
        })

        # Be polite to Yahoo Finance servers
        time.sleep(1)

    # Save combined long-format CSV
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = OUTPUT_DIR / "all_stocks.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Saved combined file: {combined_path} ({len(combined)} total rows)")

    # Print summary table
    if summary_rows:
        print("\n" + "=" * 65)
        print("Download Summary")
        print("=" * 65)
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        print("=" * 65)
    else:
        print("ERROR: No data was downloaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
