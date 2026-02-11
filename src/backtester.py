"""
Backtesting engine for comparing LLM trading agents at different decision horizons.
Supports configurable rebalancing frequency (daily, weekly, monthly).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class TradeResult:
    """Result of a single backtest run."""
    ticker: str
    strategy: str
    frequency: str
    dates: list
    portfolio_values: list
    positions: list  # 1=long, 0=cash, -1=short (we only use 1/0)
    decisions: list  # raw decisions from strategy
    num_trades: int


def compute_metrics(result: TradeResult) -> dict:
    """Compute standard trading metrics from a backtest result."""
    pv = np.array(result.portfolio_values, dtype=float)
    returns = np.diff(pv) / pv[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return {
            "ticker": result.ticker,
            "strategy": result.strategy,
            "frequency": result.frequency,
            "cumulative_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility_pct": 0.0,
            "num_trades": result.num_trades,
            "win_rate_pct": 0.0,
        }

    # Cumulative return
    cum_return = (pv[-1] / pv[0] - 1) * 100

    # Annualized return
    n_days = len(pv) - 1
    n_years = n_days / 252
    ann_return = ((pv[-1] / pv[0]) ** (1 / max(n_years, 0.01)) - 1) * 100

    # Annualized volatility
    vol = np.std(returns) * np.sqrt(252) * 100

    # Sharpe ratio (rf=0)
    sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0

    # Sortino ratio
    downside = returns[returns < 0]
    downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-10
    sortino = (np.mean(returns) * 252) / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak
    max_dd = np.min(drawdown) * 100

    # Win rate (% of positive return periods)
    win_rate = (np.sum(returns > 0) / len(returns)) * 100 if len(returns) > 0 else 0.0

    return {
        "ticker": result.ticker,
        "strategy": result.strategy,
        "frequency": result.frequency,
        "cumulative_return_pct": round(cum_return, 2),
        "annualized_return_pct": round(ann_return, 2),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "volatility_pct": round(vol, 2),
        "num_trades": result.num_trades,
        "win_rate_pct": round(win_rate, 2),
    }


def get_rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> list:
    """Get the dates on which rebalancing occurs."""
    if frequency == "daily":
        return list(dates)
    elif frequency == "weekly":
        # Every 5 trading days
        return list(dates[::5])
    elif frequency == "monthly":
        # Every 21 trading days
        return list(dates[::21])
    else:
        raise ValueError(f"Unknown frequency: {frequency}")


def run_backtest(
    prices: pd.Series,
    decision_func,
    frequency: str,
    ticker: str,
    strategy_name: str,
    initial_capital: float = 10000.0,
    transaction_cost_pct: float = 0.001,  # 0.1% per trade (conservative)
) -> TradeResult:
    """
    Run a backtest with the given decision function and rebalancing frequency.

    Args:
        prices: pd.Series of daily closing prices indexed by date
        decision_func: callable(context_dict) -> str ("BUY", "SELL", "HOLD")
        frequency: "daily", "weekly", or "monthly"
        ticker: stock ticker symbol
        strategy_name: name of the strategy
        initial_capital: starting capital
        transaction_cost_pct: transaction cost as fraction of trade value
    """
    dates = prices.index
    rebalance_dates = set(get_rebalance_dates(dates, frequency))

    cash = initial_capital
    shares = 0.0
    position = 0  # 0=cash, 1=long
    portfolio_values = []
    positions = []
    decisions = []
    num_trades = 0

    for i, date in enumerate(dates):
        current_price = prices.iloc[i]

        # Portfolio value
        pv = cash + shares * current_price
        portfolio_values.append(pv)

        if date in rebalance_dates:
            # Build context for decision function
            lookback = min(i, 60)  # up to 60 days of history
            context = {
                "date": date,
                "ticker": ticker,
                "current_price": current_price,
                "price_history": prices.iloc[max(0, i - lookback):i + 1],
                "position": position,
                "portfolio_value": pv,
                "frequency": frequency,
            }

            decision = decision_func(context)
            decisions.append(decision)

            # Execute trade
            if decision == "BUY" and position == 0:
                # Buy with all cash minus transaction cost
                cost = cash * transaction_cost_pct
                shares = (cash - cost) / current_price
                cash = 0.0
                position = 1
                num_trades += 1
            elif decision == "SELL" and position == 1:
                # Sell all shares
                proceeds = shares * current_price
                cost = proceeds * transaction_cost_pct
                cash = proceeds - cost
                shares = 0.0
                position = 0
                num_trades += 1
            # HOLD: do nothing
        else:
            decisions.append("NO_REBALANCE")

        positions.append(position)

    return TradeResult(
        ticker=ticker,
        strategy=strategy_name,
        frequency=frequency,
        dates=list(dates),
        portfolio_values=portfolio_values,
        positions=positions,
        decisions=decisions,
        num_trades=num_trades,
    )
