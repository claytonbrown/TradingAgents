"""Integration tests for Alpaca paper trading.

All tests are opt-in: requires @pytest.mark.alpaca marker, alpaca-py installed,
and APCA_API_KEY_ID + APCA_API_SECRET_KEY env vars set with paper credentials.

Run: pytest tests/test_execution_integration.py -m alpaca -v
"""

import os
import time

import pytest

try:
    import alpaca  # noqa: F401
    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False

pytestmark = pytest.mark.alpaca

_SKIP_NO_SDK = pytest.mark.skipif(not _HAS_ALPACA, reason="alpaca-py not installed")
_SKIP_NO_CREDS = pytest.mark.skipif(
    not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")),
    reason="APCA_API_KEY_ID / APCA_API_SECRET_KEY not set",
)


def _make_client():
    from tradingagents.execution.alpaca_client import AlpacaClient
    from tradingagents.execution.schemas import ExecutionConfig, ExecutionMode

    client = AlpacaClient()
    client.connect(ExecutionConfig(
        mode=ExecutionMode.PAPER,
        api_key=os.environ["APCA_API_KEY_ID"],
        api_secret=os.environ["APCA_API_SECRET_KEY"],
    ))
    return client


@_SKIP_NO_SDK
@_SKIP_NO_CREDS
def test_paper_market_buy():
    """Submit a 1-share market buy, verify SUBMITTED + broker_order_id."""
    from tradingagents.execution.schemas import OrderSide, OrderStatus, OrderType, TradeOrder

    client = _make_client()
    order = TradeOrder(ticker="AAPL", side=OrderSide.BUY, qty=1, order_type=OrderType.MARKET)
    result = client.submit_order(order)

    assert result.status == OrderStatus.SUBMITTED
    assert result.broker_order_id is not None


@_SKIP_NO_SDK
@_SKIP_NO_CREDS
def test_get_account():
    """Verify account info has expected keys."""
    client = _make_client()
    acct = client.get_account()

    assert "buying_power" in acct
    assert "equity" in acct
    assert "status" in acct
    assert isinstance(acct["buying_power"], float)


@_SKIP_NO_SDK
@_SKIP_NO_CREDS
def test_get_positions():
    """Verify get_positions returns a list."""
    client = _make_client()
    positions = client.get_positions()

    assert isinstance(positions, list)


@_SKIP_NO_SDK
@_SKIP_NO_CREDS
def test_limit_order_submit_and_cancel():
    """Submit a limit buy well below market, verify SUBMITTED, then cancel."""
    from tradingagents.execution.schemas import OrderSide, OrderStatus, OrderType, TradeOrder

    client = _make_client()
    order = TradeOrder(
        ticker="AAPL",
        side=OrderSide.BUY,
        qty=1,
        order_type=OrderType.LIMIT,
        limit_price=1.00,  # well below market — won't fill
    )
    result = client.submit_order(order)

    assert result.status == OrderStatus.SUBMITTED
    assert result.broker_order_id is not None

    # Cancel via the underlying alpaca-py client
    time.sleep(0.5)  # brief pause for order to register
    client._client.cancel_order_by_id(result.broker_order_id)
