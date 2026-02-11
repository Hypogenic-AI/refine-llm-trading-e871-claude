"""
Trading strategies: baselines and LLM-based agent.
"""
import numpy as np
import random


def buy_and_hold_strategy(context: dict) -> str:
    """Always buy and hold — the hardest baseline to beat."""
    if context["position"] == 0:
        return "BUY"
    return "HOLD"


def random_strategy(context: dict, seed_offset: int = 0) -> str:
    """Random buy/sell/hold decisions."""
    # Deterministic per date for reproducibility
    date_hash = hash(str(context["date"]) + context["ticker"]) + seed_offset
    rng = random.Random(date_hash)
    return rng.choice(["BUY", "SELL", "HOLD"])


def sma_crossover_strategy(context: dict, short_window: int = 20, long_window: int = 50) -> str:
    """SMA crossover: buy when short SMA > long SMA, sell otherwise."""
    prices = context["price_history"]
    if len(prices) < long_window:
        return "HOLD"

    short_sma = prices.iloc[-short_window:].mean()
    long_sma = prices.iloc[-long_window:].mean()

    if short_sma > long_sma and context["position"] == 0:
        return "BUY"
    elif short_sma < long_sma and context["position"] == 1:
        return "SELL"
    return "HOLD"


def create_llm_strategy(client, model: str = "gpt-4.1-mini", temperature: float = 0.3, run_seed: int = 0):
    """
    Create an LLM-based trading strategy that calls the OpenAI API.

    Returns a callable decision function.
    """
    def llm_strategy(context: dict) -> str:
        prices = context["price_history"]
        current_price = context["current_price"]
        ticker = context["ticker"]
        date = context["date"]
        position = context["position"]
        frequency = context["frequency"]

        # Compute technical indicators for context
        if len(prices) >= 20:
            sma_20 = prices.iloc[-20:].mean()
            sma_5 = prices.iloc[-5:].mean()
        else:
            sma_20 = prices.mean()
            sma_5 = prices.mean()

        # Price changes at different horizons
        changes = {}
        for period, label in [(1, "1d"), (5, "5d"), (10, "10d"), (20, "20d")]:
            if len(prices) > period:
                pct = (current_price / prices.iloc[-period - 1] - 1) * 100
                changes[label] = f"{pct:+.1f}%"

        # Recent price summary (last 10 data points)
        recent = prices.iloc[-10:]
        price_summary = ", ".join([f"{p:.2f}" for p in recent.values])

        position_str = "LONG (holding shares)" if position == 1 else "CASH (no position)"
        horizon_map = {"daily": "next trading day", "weekly": "next week", "monthly": "next month"}
        horizon = horizon_map.get(frequency, frequency)

        prompt = f"""You are a financial trading analyst making a {frequency} trading decision for {ticker}.

Date: {date.strftime('%Y-%m-%d')}
Current price: ${current_price:.2f}
Current position: {position_str}
Decision horizon: {horizon}

Recent prices (last 10 data points): {price_summary}
20-day SMA: ${sma_20:.2f}
5-day SMA: ${sma_5:.2f}
Price changes: {', '.join(f'{k}: {v}' for k, v in changes.items())}

Based on the price action, technical indicators, and your {frequency} decision horizon, should you BUY, SELL, or HOLD?

Consider:
- For {frequency} decisions, focus on {horizon} outlook
- Risk management: avoid large drawdowns
- Current trend direction and momentum

Respond with EXACTLY one word: BUY, SELL, or HOLD"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a disciplined financial trading agent. Respond with exactly one word: BUY, SELL, or HOLD. No explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=5,
                seed=run_seed,
            )
            decision = response.choices[0].message.content.strip().upper()
            # Parse: accept only valid decisions
            if "BUY" in decision:
                return "BUY"
            elif "SELL" in decision:
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            print(f"  API error for {ticker} on {date}: {e}")
            return "HOLD"  # Default to hold on error

    return llm_strategy
