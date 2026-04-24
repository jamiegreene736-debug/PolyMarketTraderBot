"""
Polymarket Fee & Rebate Calculator
==================================
Fee formula:
  Taker fee    = feeRate × quantity × price × (1 − price)
  Maker rebate = rebateRate × quantity × price × (1 − price)

The formula means fees are HIGHEST at price=$0.50 and approach
zero near $0.01 or $0.99 — rewarding near-certain trades.

The live fee rate is per-market/per-token. The fixed 0.05 helpers below are
legacy defaults; use taker_fee_for_rate() when quoting live closes.
"""


# ── Per-share calculations ──────────────────────────────────────────────────

def taker_fee_per_share(price: float) -> float:
    """Taker fee paid per share at the given price."""
    return 0.05 * price * (1 - price)


def maker_rebate_per_share(price: float) -> float:
    """Maker rebate earned per share at the given price."""
    return 0.0125 * price * (1 - price)


# ── Total order calculations ──────────────────────────────────────────────────

def taker_fee(quantity: float, price: float) -> float:
    """Total taker fee for an order."""
    return 0.05 * quantity * price * (1 - price)


def taker_fee_for_rate(quantity: float, price: float, fee_rate_bps: int | float | None) -> float:
    """
    Total taker fee for an order using Polymarket's token-specific base fee.

    The CLOB fee-rate endpoint returns values like 30 for a 0.03 fee rate.
    """
    try:
        raw_rate = float(fee_rate_bps or 0)
    except (TypeError, ValueError):
        raw_rate = 0.0
    fee_rate = raw_rate / 1000.0
    return quantity * fee_rate * price * (1 - price)


def maker_rebate(quantity: float, price: float) -> float:
    """Total maker rebate earned for a filled limit order."""
    return 0.0125 * quantity * price * (1 - price)


# ── Strategy-specific helpers ─────────────────────────────────────────────────

def effective_taker_cost_per_share(price: float) -> float:
    """
    True cost per share for a taker buy, after fees.
    Use this to calculate real profitability of near-certainty trades.
    """
    return price + taker_fee_per_share(price)


def net_profit_near_certainty(entry_price: float, exit_price: float = 1.0) -> float:
    """
    Net profit per share for a near-certainty trade.
    Buys at entry_price as taker, contract resolves at exit_price ($1.00).
    """
    return exit_price - effective_taker_cost_per_share(entry_price)


def net_profit_pct_near_certainty(entry_price: float) -> float:
    """Net profit percentage for a near-certainty buy resolving at $1.00."""
    profit = net_profit_near_certainty(entry_price)
    cost   = effective_taker_cost_per_share(entry_price)
    return round(profit / cost * 100, 3) if cost > 0 else 0.0


def market_making_rebate_round_trip(mid: float, quantity: float) -> float:
    """
    Total rebate earned for one complete market-making round trip
    (one buy fill + one sell fill at approximately the same mid price).
    """
    return 2 * maker_rebate(quantity, mid)


def min_viable_spread(mid: float) -> float:
    """
    Minimum spread (ask - bid, total) needed to profitably market-make,
    accounting for the maker rebate already covering most costs.
    At mid=$0.50, rebates earn ~$0.0063/share per side, so a 1-cent
    spread is already strongly profitable.
    """
    rebate_buffer = 2 * maker_rebate_per_share(mid)
    return max(0.01, 0.015 - rebate_buffer)


def arb_profit_after_fees(buy_price: float, sell_price: float, quantity: float) -> float:
    """
    Net profit for a cross-platform arb: buy on one platform, sell on another.
    Both legs are taker orders.
    """
    gross = (sell_price - buy_price) * quantity
    fees  = taker_fee(quantity, buy_price) + taker_fee(quantity, sell_price)
    return gross - fees


def is_arb_profitable(buy_price: float, sell_price: float, min_profit_pct: float = 0.02) -> bool:
    """Returns True if a 100-share arb position is profitable after fees."""
    profit = arb_profit_after_fees(buy_price, sell_price, 100)
    cost   = buy_price * 100
    return profit / cost >= min_profit_pct if cost > 0 else False
