"""Safety-gated execution orchestrator.

Resolves execution mode from CLI flags + env vars, validates credentials,
enforces value guardrails, and submits orders through AlpacaClient.
"""

from __future__ import annotations

import logging
import os

from .alpaca_client import AlpacaClient
from .approval import request_execution_approval
from .logger import ExecutionLogger
from .schemas import (
    ExecutionConfig,
    ExecutionMode,
    OrderResult,
    OrderStatus,
    TradeOrder,
)

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when execution safety checks fail."""


def resolve_mode(
    execute: bool = False,
    paper: bool = True,
    execute_live: bool = False,
) -> ExecutionMode:
    """Resolve execution mode from CLI flags + env vars.

    Default: DRY_RUN
    execute=True, paper=True: PAPER
    execute_live=True + APCA_PAPER=false env: LIVE (double opt-in)
    """
    if not execute and not execute_live:
        return ExecutionMode.DRY_RUN

    if execute_live:
        env_paper = os.environ.get("APCA_PAPER", "true").strip().lower()
        if env_paper != "false":
            raise ExecutionError(
                "Live execution requires APCA_PAPER=false env var. "
                f"Got APCA_PAPER='{env_paper}'. Set APCA_PAPER=false to confirm live trading."
            )
        return ExecutionMode.LIVE

    # --execute without --execute-live
    env_paper = os.environ.get("APCA_PAPER", "true").strip().lower()
    if env_paper == "false" and not execute_live:
        raise ExecutionError(
            "APCA_PAPER=false but --execute-live not set. "
            "Use --execute-live to confirm live trading, or set APCA_PAPER=true for paper."
        )

    return ExecutionMode.PAPER


def _check_credentials() -> tuple[str, str]:
    """Return (api_key, api_secret) from env or raise ExecutionError."""
    api_key = os.environ.get("APCA_API_KEY_ID", "").strip()
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    missing = []
    if not api_key:
        missing.append("APCA_API_KEY_ID")
    if not api_secret:
        missing.append("APCA_API_SECRET_KEY")
    if missing:
        raise ExecutionError(
            f"Missing required credentials: {', '.join(missing)}. "
            "Set environment variables before executing trades."
        )
    return api_key, api_secret


def _check_value_limits(
    orders: list[TradeOrder],
    prices: dict[str, float],
    config: ExecutionConfig,
) -> None:
    """Raise ExecutionError if any order or batch total exceeds limits."""
    cumulative = 0.0
    for order in orders:
        price = prices.get(order.ticker, 0.0)
        if price <= 0:
            raise ExecutionError(
                f"{order.ticker}: no valid price available — cannot verify value limits"
            )
        value = order.qty * price
        if value > config.max_order_value:
            raise ExecutionError(
                f"{order.ticker}: order value ${value:,.2f} exceeds "
                f"max_order_value ${config.max_order_value:,.2f}"
            )
        cumulative += value
        if cumulative > config.max_total_value:
            raise ExecutionError(
                f"Batch total ${cumulative:,.2f} exceeds "
                f"max_total_value ${config.max_total_value:,.2f}"
            )


def execute_orders(
    orders: list[TradeOrder],
    prices: dict[str, float],
    config: ExecutionConfig | None = None,
    execute: bool = False,
    paper: bool = True,
    execute_live: bool = False,
    context_id: str | None = None,
) -> list[OrderResult]:
    """Main entry point. Resolves mode, validates, submits.

    Args:
        orders: Trade orders to execute.
        prices: {ticker: current_price} for value limit checks.
        config: Optional config (defaults created if None).
        execute: True to enable paper trading.
        paper: Paper mode flag (default True).
        execute_live: True to enable live trading (requires APCA_PAPER=false).

    Returns:
        List of OrderResult for each order.
    """
    mode = resolve_mode(execute=execute, paper=paper, execute_live=execute_live)

    if config is None:
        config = ExecutionConfig(mode=mode)
    else:
        config.mode = mode

    # Credential check + client connection
    client = AlpacaClient()
    if mode in (ExecutionMode.PAPER, ExecutionMode.LIVE):
        api_key, api_secret = _check_credentials()
        config.api_key = api_key
        config.api_secret = api_secret
        client.connect(config)
        try:
            account = client.get_account()
            config.buying_power = account["buying_power"]
        except Exception as exc:
            logger.warning("Could not fetch account info: %s", exc)
    else:
        client.connect(config)

    # Value guardrails (all modes — catch issues before any submission)
    _check_value_limits(orders, prices, config)

    # Approval gate — LIVE mode requires explicit human approval
    if mode == ExecutionMode.LIVE:
        result = request_execution_approval(orders, prices, config, context_id)
        if result != "APPROVED":
            logger.warning("Live execution %s — skipping all orders", result.lower())
            return [
                OrderResult(order=o, status=OrderStatus.SKIPPED, error=f"Approval {result.lower()}")
                for o in orders
            ]

    exec_logger = ExecutionLogger(config.log_path)

    logger.info(
        "Executing %d orders in %s mode", len(orders), mode.value
    )

    results = []
    for order in orders:
        result = client.submit_order(order)
        exec_logger.log(order, result, mode.value)
        results.append(result)

    submitted = sum(1 for r in results if r.status == OrderStatus.SUBMITTED)
    skipped = sum(1 for r in results if r.status == OrderStatus.SKIPPED)
    rejected = sum(1 for r in results if r.status == OrderStatus.REJECTED)
    logger.info(
        "Execution complete: %d submitted, %d skipped, %d rejected",
        submitted, skipped, rejected,
    )

    return results
