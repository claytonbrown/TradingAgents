"""Unit tests for tradingagents.execution module.

Task 21: mock TradingClient, verify order construction, safety gate logic,
credential validation, dry-run output.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.execution.schemas import (
    ExecutionConfig,
    ExecutionMode,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeOrder,
)
from tradingagents.execution.converter import decision_to_orders
from tradingagents.execution.executor import (
    ExecutionError,
    _check_credentials,
    _check_value_limits,
    execute_orders,
    resolve_mode,
)
from tradingagents.execution.logger import ExecutionLogger


# ---------------------------------------------------------------------------
# resolve_mode
# ---------------------------------------------------------------------------


class TestResolveMode:
    def test_default_is_dry_run(self):
        assert resolve_mode() == ExecutionMode.DRY_RUN

    def test_execute_paper(self):
        assert resolve_mode(execute=True, paper=True) == ExecutionMode.PAPER

    @patch.dict(os.environ, {"APCA_PAPER": "false"})
    def test_execute_live_double_optin(self):
        assert resolve_mode(execute_live=True) == ExecutionMode.LIVE

    @patch.dict(os.environ, {"APCA_PAPER": "true"})
    def test_execute_live_without_env_raises(self):
        with pytest.raises(ExecutionError, match="APCA_PAPER=false"):
            resolve_mode(execute_live=True)

    @patch.dict(os.environ, {"APCA_PAPER": "false"})
    def test_execute_without_live_flag_raises(self):
        with pytest.raises(ExecutionError, match="--execute-live not set"):
            resolve_mode(execute=True, paper=True)


# ---------------------------------------------------------------------------
# _check_credentials
# ---------------------------------------------------------------------------


class TestCheckCredentials:
    @patch.dict(os.environ, {"APCA_API_KEY_ID": "TESTKEY", "APCA_API_SECRET_KEY": "TESTSECRET"})
    def test_valid_credentials(self):
        key, secret = _check_credentials()
        assert key == "TESTKEY"
        assert secret == "TESTSECRET"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_both_raises(self):
        # Remove any existing keys
        os.environ.pop("APCA_API_KEY_ID", None)
        os.environ.pop("APCA_API_SECRET_KEY", None)
        with pytest.raises(ExecutionError, match="APCA_API_KEY_ID"):
            _check_credentials()

    @patch.dict(os.environ, {"APCA_API_KEY_ID": "KEY"})
    def test_missing_secret_raises(self):
        os.environ.pop("APCA_API_SECRET_KEY", None)
        with pytest.raises(ExecutionError, match="APCA_API_SECRET_KEY"):
            _check_credentials()


# ---------------------------------------------------------------------------
# _check_value_limits
# ---------------------------------------------------------------------------


class TestValueLimits:
    def test_within_limits(self):
        orders = [TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=5)]
        config = ExecutionConfig(max_order_value=10_000, max_total_value=50_000)
        _check_value_limits(orders, {"NVDA": 100.0}, config)  # $500 — OK

    def test_single_order_exceeds(self):
        orders = [TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=200)]
        config = ExecutionConfig(max_order_value=10_000, max_total_value=50_000)
        with pytest.raises(ExecutionError, match="max_order_value"):
            _check_value_limits(orders, {"NVDA": 100.0}, config)  # $20k > $10k

    def test_batch_total_exceeds(self):
        orders = [
            TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=90),
            TradeOrder(ticker="META", side=OrderSide.BUY, qty=90),
        ]
        config = ExecutionConfig(max_order_value=10_000, max_total_value=10_000)
        with pytest.raises(ExecutionError, match="max_total_value"):
            _check_value_limits(orders, {"NVDA": 100.0, "META": 100.0}, config)

    def test_missing_price_raises(self):
        orders = [TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=5)]
        config = ExecutionConfig()
        with pytest.raises(ExecutionError, match="no valid price"):
            _check_value_limits(orders, {}, config)


# ---------------------------------------------------------------------------
# decision_to_orders (converter)
# ---------------------------------------------------------------------------


class TestDecisionToOrders:
    def test_buy(self):
        orders = decision_to_orders("NVDA", "BUY", buying_power=100_000, current_price=100.0)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].qty == 20  # 2% of 100k / 100 = 20

    def test_overweight_half_size(self):
        orders = decision_to_orders("NVDA", "OVERWEIGHT", buying_power=100_000, current_price=100.0)
        assert len(orders) == 1
        assert orders[0].qty == 10  # 2% * 0.5 = 1% → 10 shares

    def test_hold_no_order(self):
        assert decision_to_orders("NVDA", "HOLD", buying_power=100_000, current_price=100.0) == []

    def test_sell_full_position(self):
        orders = decision_to_orders("NVDA", "SELL", buying_power=100_000, current_price=100.0, held_qty=50)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].qty == 50

    def test_underweight_partial_sell(self):
        orders = decision_to_orders("NVDA", "UNDERWEIGHT", buying_power=100_000, current_price=100.0, held_qty=50)
        assert len(orders) == 1
        assert orders[0].qty == 25  # floor(50 * 0.5)

    def test_sell_no_position_skips(self):
        assert decision_to_orders("NVDA", "SELL", buying_power=100_000, current_price=100.0, held_qty=0) == []

    def test_unknown_rating_skips(self):
        assert decision_to_orders("NVDA", "STRONG_BUY", buying_power=100_000, current_price=100.0) == []

    def test_zero_price_skips(self):
        assert decision_to_orders("NVDA", "BUY", buying_power=100_000, current_price=0.0) == []

    def test_custom_position_size(self):
        orders = decision_to_orders("NVDA", "BUY", buying_power=100_000, current_price=100.0, position_size_pct=5.0)
        assert orders[0].qty == 50  # 5% of 100k / 100


# ---------------------------------------------------------------------------
# AlpacaClient (mocked SDK)
# ---------------------------------------------------------------------------


class TestAlpacaClientDryRun:
    def test_dry_run_skips_order(self):
        from tradingagents.execution.alpaca_client import AlpacaClient

        client = AlpacaClient()
        client.connect(ExecutionConfig(mode=ExecutionMode.DRY_RUN))
        order = TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=10)
        result = client.submit_order(order)
        assert result.status == OrderStatus.SKIPPED

    def test_dry_run_no_sdk_needed(self):
        """DRY_RUN should work even without alpaca-py installed."""
        from tradingagents.execution.alpaca_client import AlpacaClient

        client = AlpacaClient()
        client.connect(ExecutionConfig(mode=ExecutionMode.DRY_RUN))
        assert client._client is None

    def test_connect_without_credentials_raises(self):
        from tradingagents.execution.alpaca_client import AlpacaClient

        client = AlpacaClient()
        with pytest.raises(ValueError, match="api_key and api_secret"):
            client.connect(ExecutionConfig(mode=ExecutionMode.PAPER))


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("alpaca"),
    reason="alpaca-py not installed",
)
class TestAlpacaClientMocked:
    """Tests with mocked alpaca-py SDK."""

    def test_market_order_construction(self):
        from tradingagents.execution.alpaca_client import AlpacaClient

        mock_response = MagicMock()
        mock_response.id = "mock-order-id"
        mock_response.status = "accepted"

        with patch("tradingagents.execution.alpaca_client.TradingClient") as MockTC:
            mock_tc = MockTC.return_value
            mock_tc.submit_order.return_value = mock_response

            client = AlpacaClient()
            client.connect(ExecutionConfig(
                mode=ExecutionMode.PAPER, api_key="KEY", api_secret="SECRET"
            ))
            order = TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=10)
            result = client.submit_order(order)

            assert result.status == OrderStatus.SUBMITTED
            assert result.broker_order_id == "mock-order-id"
            mock_tc.submit_order.assert_called_once()

    def test_limit_order_construction(self):
        from tradingagents.execution.alpaca_client import AlpacaClient

        mock_response = MagicMock()
        mock_response.id = "limit-id"
        mock_response.status = "accepted"

        with patch("tradingagents.execution.alpaca_client.TradingClient") as MockTC:
            mock_tc = MockTC.return_value
            mock_tc.submit_order.return_value = mock_response

            client = AlpacaClient()
            client.connect(ExecutionConfig(
                mode=ExecutionMode.PAPER, api_key="KEY", api_secret="SECRET"
            ))
            order = TradeOrder(
                ticker="NVDA", side=OrderSide.BUY, qty=10,
                order_type=OrderType.LIMIT, limit_price=130.0,
            )
            result = client.submit_order(order)
            assert result.status == OrderStatus.SUBMITTED

            # Verify the request passed to SDK was a LimitOrderRequest
            call_args = mock_tc.submit_order.call_args[0][0]
            assert hasattr(call_args, "limit_price")

    def test_bracket_order_construction(self):
        from tradingagents.execution.alpaca_client import AlpacaClient

        mock_response = MagicMock()
        mock_response.id = "bracket-id"
        mock_response.status = "accepted"

        with patch("tradingagents.execution.alpaca_client.TradingClient") as MockTC:
            mock_tc = MockTC.return_value
            mock_tc.submit_order.return_value = mock_response

            client = AlpacaClient()
            client.connect(ExecutionConfig(
                mode=ExecutionMode.PAPER, api_key="KEY", api_secret="SECRET"
            ))
            order = TradeOrder(
                ticker="NVDA", side=OrderSide.BUY, qty=10,
                order_type=OrderType.BRACKET,
                take_profit_price=160.0, stop_price=120.0,
            )
            result = client.submit_order(order)
            assert result.status == OrderStatus.SUBMITTED


# ---------------------------------------------------------------------------
# ExecutionLogger
# ---------------------------------------------------------------------------


class TestExecutionLogger:
    def test_log_creates_file(self, tmp_path):
        log_path = str(tmp_path / "exec.jsonl")
        logger = ExecutionLogger(log_path)
        order = TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=10)
        result = OrderResult(order=order, status=OrderStatus.SKIPPED)
        logger.log(order, result, "dry_run")

        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert entry["ticker"] == "NVDA"
        assert entry["mode"] == "dry_run"
        assert entry["status"] == "skipped"

    def test_log_appends(self, tmp_path):
        log_path = str(tmp_path / "exec.jsonl")
        logger = ExecutionLogger(log_path)
        for ticker in ("NVDA", "META"):
            order = TradeOrder(ticker=ticker, side=OrderSide.BUY, qty=5)
            result = OrderResult(order=order, status=OrderStatus.SKIPPED)
            logger.log(order, result, "dry_run")

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# execute_orders (dry-run end-to-end)
# ---------------------------------------------------------------------------


class TestExecuteOrdersDryRun:
    def test_dry_run_returns_skipped(self, tmp_path):
        orders = [TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=10)]
        config = ExecutionConfig(log_path=str(tmp_path / "exec.jsonl"))
        results = execute_orders(orders, {"NVDA": 100.0}, config=config)
        assert len(results) == 1
        assert results[0].status == OrderStatus.SKIPPED

    def test_dry_run_no_credentials_needed(self, tmp_path):
        """Dry-run should work without any env vars set."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")}
        with patch.dict(os.environ, env, clear=True):
            orders = [TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=5)]
            config = ExecutionConfig(log_path=str(tmp_path / "exec.jsonl"))
            results = execute_orders(orders, {"NVDA": 100.0}, config=config)
            assert results[0].status == OrderStatus.SKIPPED

    def test_dry_run_still_checks_value_limits(self, tmp_path):
        orders = [TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=200)]
        config = ExecutionConfig(
            max_order_value=1_000, log_path=str(tmp_path / "exec.jsonl")
        )
        with pytest.raises(ExecutionError, match="max_order_value"):
            execute_orders(orders, {"NVDA": 100.0}, config=config)


# ---------------------------------------------------------------------------
# Approval gate in execute_orders (LIVE mode)
# ---------------------------------------------------------------------------


class TestExecuteOrdersApproval:
    """Tests for the approval gate in execute_orders() LIVE mode."""

    def _make_orders(self):
        return [TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=5)]

    def _make_prices(self):
        return {"NVDA": 100.0}

    def _live_patches(self, approval_return):
        """Stack of patches needed for LIVE mode: env, creds, client, approval."""
        return [
            patch.dict(os.environ, {"APCA_PAPER": "false"}),
            patch("tradingagents.execution.executor._check_credentials", return_value=("K", "S")),
            patch("tradingagents.execution.executor.AlpacaClient"),
            patch(
                "tradingagents.execution.executor.request_execution_approval",
                return_value=approval_return,
            ),
        ]

    def test_approved_submits_orders(self, tmp_path):
        patches = self._live_patches("APPROVED")
        for p in patches:
            p.start()
        try:
            # Configure mock client to return a real OrderResult
            from tradingagents.execution.executor import AlpacaClient as MockedClass
            mock_client = MockedClass.return_value
            mock_client.submit_order.return_value = OrderResult(
                order=self._make_orders()[0], status=OrderStatus.SUBMITTED,
                broker_order_id="mock-id",
            )
            mock_client.get_account.return_value = {"buying_power": 100_000.0}

            config = ExecutionConfig(log_path=str(tmp_path / "exec.jsonl"))
            results = execute_orders(
                self._make_orders(), self._make_prices(),
                config=config, execute_live=True,
            )
            assert results[0].status == OrderStatus.SUBMITTED
        finally:
            for p in reversed(patches):
                p.stop()

    def test_rejected_skips_orders(self, tmp_path):
        patches = self._live_patches("REJECTED")
        for p in patches:
            p.start()
        try:
            config = ExecutionConfig(log_path=str(tmp_path / "exec.jsonl"))
            results = execute_orders(
                self._make_orders(), self._make_prices(),
                config=config, execute_live=True,
            )
            assert results[0].status == OrderStatus.SKIPPED
            assert "rejected" in results[0].error
        finally:
            for p in reversed(patches):
                p.stop()

    def test_timeout_skips_orders(self, tmp_path):
        patches = self._live_patches("TIMEOUT")
        for p in patches:
            p.start()
        try:
            config = ExecutionConfig(log_path=str(tmp_path / "exec.jsonl"))
            results = execute_orders(
                self._make_orders(), self._make_prices(),
                config=config, execute_live=True,
            )
            assert results[0].status == OrderStatus.SKIPPED
            assert "timeout" in results[0].error
        finally:
            for p in reversed(patches):
                p.stop()


# ---------------------------------------------------------------------------
# request_execution_approval (approval.py direct tests)
# ---------------------------------------------------------------------------


from tradingagents.execution.approval import (
    _build_order_summary,
    request_execution_approval,
)


class TestRequestExecutionApproval:
    def _orders_and_prices(self):
        orders = [TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=10)]
        return orders, {"NVDA": 130.0}

    def test_messaging_framework_calls_request_approval(self):
        orders, prices = self._orders_and_prices()
        mock_ra = MagicMock(return_value="APPROVED")
        with patch.dict("sys.modules", {"tradingagents.messaging": MagicMock(request_approval=mock_ra)}):
            result = request_execution_approval(orders, prices, ExecutionConfig(), context_id="t1")
        assert result == "APPROVED"
        mock_ra.assert_called_once()
        assert mock_ra.call_args[1]["context_id"] == "t1"

    def test_tty_fallback_approve(self):
        orders, prices = self._orders_and_prices()
        # Ensure messaging import fails, then mock TTY
        with patch.dict("sys.modules", {"tradingagents.messaging": None}):
            with patch("tradingagents.execution.approval.sys") as mock_sys:
                mock_sys.stdin.isatty.return_value = True
                with patch("builtins.input", return_value="y"):
                    result = request_execution_approval(orders, prices, ExecutionConfig())
        assert result == "APPROVED"

    def test_no_tty_returns_rejected(self):
        orders, prices = self._orders_and_prices()
        with patch.dict("sys.modules", {"tradingagents.messaging": None}):
            with patch("tradingagents.execution.approval.sys") as mock_sys:
                mock_sys.stdin.isatty.return_value = False
                result = request_execution_approval(orders, prices, ExecutionConfig())
        assert result == "REJECTED"


class TestBuildOrderSummary:
    def test_summary_format(self):
        orders = [
            TradeOrder(ticker="NVDA", side=OrderSide.BUY, qty=10),
            TradeOrder(ticker="META", side=OrderSide.SELL, qty=5),
        ]
        text = _build_order_summary(orders, {"NVDA": 130.0, "META": 500.0}, ExecutionConfig())
        assert "NVDA" in text
        assert "META" in text
        assert "1,300.00" in text
        assert "2,500.00" in text
        assert "Total" in text
        assert "3,800.00" in text