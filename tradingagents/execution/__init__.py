"""Broker execution module for TradingAgents.

Converts portfolio decisions into broker orders. Supports dry-run (default),
paper trading, and live execution with safety gates.

Usage:
    from tradingagents.execution import (
        BrokerClient, AlpacaClient,
        TradeOrder, OrderResult, ExecutionConfig,
        decision_to_orders, ExecutionLogger,
    )
"""

from .schemas import TradeOrder, OrderResult, ExecutionConfig, ExecutionMode
from .base import BrokerClient
from .alpaca_client import AlpacaClient
from .approval import request_execution_approval
from .converter import decision_to_orders
from .logger import ExecutionLogger
from .executor import ExecutionError, resolve_mode, execute_orders

__all__ = [
    "BrokerClient",
    "AlpacaClient",
    "TradeOrder",
    "OrderResult",
    "ExecutionConfig",
    "ExecutionMode",
    "decision_to_orders",
    "ExecutionLogger",
    "ExecutionError",
    "resolve_mode",
    "execute_orders",
    "request_execution_approval",
]
