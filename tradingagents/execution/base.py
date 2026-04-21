"""Broker-agnostic client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import ExecutionConfig, OrderResult, TradeOrder


class BrokerClient(ABC):
    """Abstract base class for broker integrations."""

    @abstractmethod
    def connect(self, config: ExecutionConfig) -> None:
        """Establish connection to the broker."""

    @abstractmethod
    def submit_order(self, order: TradeOrder) -> OrderResult:
        """Submit a single order. Returns result with status."""

    @abstractmethod
    def get_account(self) -> dict:
        """Return account info including status and buying power."""

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Return current open positions."""
