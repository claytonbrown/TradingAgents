"""Data schemas for broker execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    BRACKET = "bracket"
    TRAILING_STOP = "trailing_stop"


class ExecutionMode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class TradeOrder:
    """A single trade order to submit to a broker."""

    ticker: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    take_profit_price: Optional[float] = None
    time_in_force: str = "day"
    client_order_id: Optional[str] = None


@dataclass
class OrderResult:
    """Result of an order submission."""

    order: TradeOrder
    status: OrderStatus
    broker_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_qty: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ExecutionConfig:
    """Configuration for broker execution."""

    mode: ExecutionMode = ExecutionMode.DRY_RUN
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    position_size_pct: float = 2.0
    max_order_value: float = 10_000.0
    max_total_value: float = 50_000.0
    buying_power: Optional[float] = None
    log_path: str = "data/execution_log.jsonl"
