"""Alpaca broker client implementation using alpaca-py SDK."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from .base import BrokerClient
from .schemas import (
    ExecutionConfig,
    ExecutionMode,
    OrderResult,
    OrderStatus,
    OrderType,
    TradeOrder,
)

logger = logging.getLogger(__name__)

try:
    from alpaca.common.exceptions import APIError
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
    from alpaca.trading.requests import (
        LimitOrderRequest,
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
        TrailingStopOrderRequest,
    )

    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False

_TIF_MAP = {
    "day": "day",
    "gtc": "gtc",
    "ioc": "ioc",
    "fok": "fok",
}


def _generate_client_order_id(ticker: str, date: str) -> str:
    """Generate idempotent client order ID: {TICKER}_{date}_{8-char-uuid}."""
    return f"{ticker.upper()}_{date}_{uuid.uuid4().hex[:8]}"


class AlpacaClient(BrokerClient):
    """Alpaca Trading API client via alpaca-py SDK."""

    def __init__(self) -> None:
        self._client: Optional[TradingClient] = None
        self._config: Optional[ExecutionConfig] = None

    def connect(self, config: ExecutionConfig) -> None:
        """Create TradingClient. Paper=True for PAPER, False for LIVE, skip for DRY_RUN."""
        self._config = config
        if config.mode == ExecutionMode.DRY_RUN:
            logger.info("DRY_RUN mode — no Alpaca client created")
            return
        if not config.api_key or not config.api_secret:
            raise ValueError("api_key and api_secret are required for PAPER/LIVE mode")
        if not _HAS_ALPACA:
            raise RuntimeError(
                "alpaca-py is not installed. Install with: pip install alpaca-py"
            )
        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.api_secret,
            paper=(config.mode == ExecutionMode.PAPER),
        )
        logger.info("Connected to Alpaca (%s mode)", config.mode.value)

    def submit_order(self, order: TradeOrder) -> OrderResult:
        """Submit order to Alpaca. DRY_RUN returns SKIPPED without API call."""
        if self._config and self._config.mode == ExecutionMode.DRY_RUN:
            logger.info("DRY_RUN: skipping %s %s x%.2f", order.side.value, order.ticker, order.qty)
            return OrderResult(order=order, status=OrderStatus.SKIPPED)
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        # Generate client_order_id if not set
        if not order.client_order_id:
            from datetime import date as date_mod
            order.client_order_id = _generate_client_order_id(
                order.ticker, date_mod.today().isoformat()
            )

        side = OrderSide.BUY if order.side.value == "buy" else OrderSide.SELL
        tif = TimeInForce(_TIF_MAP.get(order.time_in_force, "day"))

        try:
            request = self._build_request(order, side, tif)
            response = self._client.submit_order(request)
            logger.info(
                "Order submitted: %s %s x%.2f → %s",
                order.side.value, order.ticker, order.qty, response.status,
            )
            return OrderResult(
                order=order,
                status=OrderStatus.SUBMITTED,
                broker_order_id=str(response.id),
            )
        except APIError as exc:
            logger.error("Order rejected: %s — %s", order.ticker, exc)
            return OrderResult(
                order=order, status=OrderStatus.REJECTED, error=str(exc)
            )

    def get_account(self) -> dict:
        """Return account info with PDT warning."""
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        acct = self._client.get_account()
        daytrade_count = int(acct.daytrade_count or 0)
        if daytrade_count >= 3:
            logger.warning(
                "PDT warning: %d day trades in rolling 5-day window", daytrade_count
            )
        return {
            "status": str(acct.status),
            "buying_power": float(acct.buying_power),
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "daytrade_count": daytrade_count,
            "pattern_day_trader": bool(acct.pattern_day_trader),
        }

    def get_positions(self) -> list[dict]:
        """Return current open positions."""
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in self._client.get_all_positions()
        ]

    @staticmethod
    def _build_request(order: TradeOrder, side: OrderSide, tif: TimeInForce):
        """Build the appropriate alpaca-py order request."""
        common = dict(
            symbol=order.ticker,
            qty=order.qty,
            side=side,
            time_in_force=tif,
            client_order_id=order.client_order_id,
        )
        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(**common)
        if order.order_type == OrderType.LIMIT:
            return LimitOrderRequest(**common, limit_price=order.limit_price)
        if order.order_type == OrderType.BRACKET:
            return MarketOrderRequest(
                **common,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
                stop_loss=StopLossRequest(stop_price=order.stop_price),
            )
        if order.order_type == OrderType.TRAILING_STOP:
            return TrailingStopOrderRequest(
                **common, trail_percent=order.trail_percent
            )
        raise ValueError(f"Unsupported order type: {order.order_type}")
