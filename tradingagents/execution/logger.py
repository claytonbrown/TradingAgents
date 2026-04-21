"""Execution audit logger."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from .schemas import OrderResult, TradeOrder


class ExecutionLogger:
    """Append-only JSONL logger for all execution activity."""

    def __init__(self, log_path: str = "data/execution_log.jsonl"):
        self.log_path = log_path

    def log(self, order: TradeOrder, result: OrderResult, mode: str) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "ticker": order.ticker,
            "side": order.side.value,
            "qty": order.qty,
            "type": order.order_type.value,
            "client_order_id": order.client_order_id,
            "status": result.status.value,
            "broker_order_id": result.broker_order_id,
            "fill_price": result.fill_price,
            "fill_qty": result.fill_qty,
            "error": result.error,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
