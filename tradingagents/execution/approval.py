"""Live execution approval gate.

Requires human approval before submitting live orders. Uses the messaging
framework (spec 038) when available, falls back to interactive TTY prompt,
and rejects if neither is available (fail-safe).
"""

from __future__ import annotations

import sys

from .schemas import ExecutionConfig, TradeOrder


def _build_order_summary(
    orders: list[TradeOrder],
    prices: dict[str, float],
    config: ExecutionConfig,
) -> str:
    """Build a text table summarising orders for approval."""
    lines = [
        f"{'Ticker':<8} {'Side':<6} {'Qty':>8} {'Type':<10} {'Est. Cost':>12}",
        "-" * 50,
    ]
    running = 0.0
    for o in orders:
        price = prices.get(o.ticker, 0.0)
        cost = o.qty * price
        running += cost
        lines.append(
            f"{o.ticker:<8} {o.side.value:<6} {o.qty:>8.2f} {o.order_type.value:<10} ${cost:>11,.2f}"
        )
    lines.append("-" * 50)
    lines.append(f"{'Total':>34} ${running:>11,.2f}")
    bp = getattr(config, "buying_power", None)
    if bp is not None:
        lines.append(f"{'Buying power remaining':>34} ${bp - running:>11,.2f}")
    return "\n".join(lines)


def request_execution_approval(
    orders: list[TradeOrder],
    prices: dict[str, float],
    config: ExecutionConfig,
    context_id: str | None = None,
) -> str:
    """Request approval for live execution. Returns 'APPROVED', 'REJECTED', or 'TIMEOUT'."""
    summary = _build_order_summary(orders, prices, config)

    # Try messaging framework first (spec 038)
    try:
        from tradingagents.messaging import request_approval  # type: ignore[import-not-found]

        return request_approval(
            context_id=context_id,
            topic="approval.execution",
            title="Live Execution Approval",
            body=summary,
            timeout_minutes=30,
        )
    except ImportError:
        pass

    # Fallback: interactive TTY prompt
    if sys.stdin.isatty():
        print("\n=== LIVE EXECUTION APPROVAL REQUIRED ===\n")
        print(summary)
        answer = input("\nApprove execution? [y/N]: ").strip().lower()
        return "APPROVED" if answer == "y" else "REJECTED"

    # No messaging, no TTY — fail-safe
    return "REJECTED"
