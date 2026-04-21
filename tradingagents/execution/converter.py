"""Convert portfolio decisions to trade orders."""

from __future__ import annotations

import logging
import math
from typing import Optional

from .schemas import OrderSide, OrderType, TradeOrder

logger = logging.getLogger(__name__)

# 5-tier rating → (side, size_fraction)
# size_fraction is the multiplier applied to position_size_pct
_RATING_MAP: dict[str, tuple[Optional[OrderSide], float]] = {
    "BUY": (OrderSide.BUY, 1.0),
    "OVERWEIGHT": (OrderSide.BUY, 0.5),
    "HOLD": (None, 0.0),
    "UNDERWEIGHT": (OrderSide.SELL, 0.5),
    "SELL": (OrderSide.SELL, 1.0),
}


def decision_to_orders(
    ticker: str,
    rating: str,
    buying_power: float,
    current_price: float,
    position_size_pct: float = 2.0,
    held_qty: float = 0.0,
) -> list[TradeOrder]:
    """Map a 5-tier rating to trade orders.

    Args:
        ticker: Stock symbol.
        rating: One of BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL.
        buying_power: Available buying power in account.
        current_price: Current market price per share.
        position_size_pct: Max percent of buying power per trade (default 2%).
        held_qty: Shares currently held (for sell sizing).

    Returns:
        List of TradeOrder (empty for HOLD or when qty rounds to 0).
    """
    rating_upper = rating.strip().upper()
    mapping = _RATING_MAP.get(rating_upper)
    if mapping is None:
        logger.warning("Unknown rating '%s' for %s — treating as HOLD", rating, ticker)
        return []

    side, fraction = mapping
    if side is None:
        logger.info("%s: HOLD — no order", ticker)
        return []

    if current_price <= 0:
        logger.warning("%s: invalid price %.4f — skipping", ticker, current_price)
        return []

    if side == OrderSide.BUY:
        dollar_amount = buying_power * (position_size_pct / 100.0) * fraction
        qty = math.floor(dollar_amount / current_price)
    else:
        # SELL/UNDERWEIGHT: sell fraction of held position
        if held_qty <= 0:
            logger.info("%s: %s but no position held — skipping", ticker, rating_upper)
            return []
        qty = math.floor(held_qty * fraction) if fraction < 1.0 else held_qty

    if qty <= 0:
        logger.info("%s: computed qty=0 — skipping", ticker)
        return []

    return [
        TradeOrder(
            ticker=ticker,
            side=side,
            qty=float(qty),
            order_type=OrderType.MARKET,
        )
    ]
