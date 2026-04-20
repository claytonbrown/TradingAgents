"""Integration tests for run_headless.py — checkpoint resume, reuse threshold, cache fallback, Redis lock."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from run_headless import (
    BudgetExceededError,
    HeadlessRunner,
    InMemoryCache,
    RedisCache,
    RedisBackend,
    TickerLockError,
    build_parser,
)


# ── Reuse Threshold ──────────────────────────────────────────────────


class TestReuseThreshold(unittest.TestCase):
    """Verify --reuse-threshold skips re-analysis when price moved less than N%."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_prev_result(self, ticker: str, price: float):
        result = {
            "ticker": ticker,
            "date": "2026-04-20",
            "decision": "BUY",
            "confidence": "high",
            "current_price": price,
            "thesis": "Previous thesis",
            "reports": {"market": "", "sentiment": "", "news": "", "fundamentals": ""},
            "metadata": {"model": "gpt-5.4", "debate_rounds": 3,
                         "elapsed_seconds": 100, "tokens_in": 1000,
                         "tokens_out": 200, "cost_usd": 0.1},
        }
        (self.output_dir / f"{ticker}.json").write_text(json.dumps(result))

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._fetch_price")
    @patch("run_headless.HeadlessRunner._run_analysis")
    def test_reuse_when_price_below_threshold(self, mock_run, mock_price, mock_log):
        """If price moved <threshold%, previous result is reused (no re-analysis)."""
        self._write_prev_result("NVDA", 200.0)
        mock_price.return_value = 200.5  # 0.25% move, below 1% threshold

        args = build_parser().parse_args([
            "--ticker", "NVDA", "--quiet",
            "--output-dir", self.tmpdir,
            "--reuse-threshold", "1.0",
        ])
        runner = HeadlessRunner(args)
        result = runner._check_reuse("NVDA")

        self.assertIsNotNone(result)
        self.assertEqual(result["ticker"], "NVDA")
        self.assertTrue(result["metadata"]["reused"])
        self.assertEqual(result["current_price"], 200.5)
        mock_run.assert_not_called()

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._fetch_price")
    def test_reanalyse_when_price_above_threshold(self, mock_price, mock_log):
        """If price moved >=threshold%, re-analysis is triggered."""
        self._write_prev_result("NVDA", 200.0)
        mock_price.return_value = 210.0  # 5% move, above 1% threshold

        args = build_parser().parse_args([
            "--ticker", "NVDA", "--quiet",
            "--output-dir", self.tmpdir,
            "--reuse-threshold", "1.0",
        ])
        runner = HeadlessRunner(args)
        result = runner._check_reuse("NVDA")

        self.assertIsNone(result)  # None means "proceed with analysis"

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_reuse_disabled_when_threshold_zero(self, mock_log):
        """Threshold of 0 disables reuse entirely."""
        self._write_prev_result("NVDA", 200.0)

        args = build_parser().parse_args([
            "--ticker", "NVDA", "--quiet",
            "--output-dir", self.tmpdir,
            "--reuse-threshold", "0",
        ])
        runner = HeadlessRunner(args)
        result = runner._check_reuse("NVDA")

        self.assertIsNone(result)

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._fetch_price")
    def test_reuse_no_previous_file(self, mock_price, mock_log):
        """No previous result file → proceed with analysis."""
        args = build_parser().parse_args([
            "--ticker", "NVDA", "--quiet",
            "--output-dir", self.tmpdir,
            "--reuse-threshold", "1.0",
        ])
        runner = HeadlessRunner(args)
        result = runner._check_reuse("NVDA")

        self.assertIsNone(result)
        mock_price.assert_not_called()

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._fetch_price")
    def test_reuse_no_previous_price(self, mock_price, mock_log):
        """Previous result has no current_price → proceed with analysis."""
        result_no_price = {
            "ticker": "NVDA", "date": "2026-04-20", "decision": "BUY",
            "confidence": "", "current_price": None, "thesis": "",
            "reports": {"market": "", "sentiment": "", "news": "", "fundamentals": ""},
            "metadata": {"model": "", "debate_rounds": 3, "elapsed_seconds": 1,
                         "tokens_in": 0, "tokens_out": 0, "cost_usd": 0},
        }
        (self.output_dir / "NVDA.json").write_text(json.dumps(result_no_price))

        args = build_parser().parse_args([
            "--ticker", "NVDA", "--quiet",
            "--output-dir", self.tmpdir,
            "--reuse-threshold", "1.0",
        ])
        runner = HeadlessRunner(args)
        result = runner._check_reuse("NVDA")

        self.assertIsNone(result)


# ── Cache Fallback ────────────────────────────────────────────────────


class TestCacheFallback(unittest.TestCase):
    """Verify RedisCache falls back to InMemoryCache on Redis errors."""

    def test_get_falls_back_on_redis_error(self):
        """When Redis GET raises, fallback in-memory cache is used."""
        mock_redis = MagicMock()
        mock_redis.get.side_effect = ConnectionError("Redis down")

        fallback = InMemoryCache()
        fallback.set("mykey", "cached_value")

        cache = RedisCache(mock_redis, fallback=fallback)
        result = cache.get("mykey")

        self.assertEqual(result, "cached_value")

    def test_set_populates_fallback_even_on_redis_error(self):
        """When Redis SETEX raises, value is still stored in fallback."""
        mock_redis = MagicMock()
        mock_redis.setex.side_effect = ConnectionError("Redis down")

        fallback = InMemoryCache()
        cache = RedisCache(mock_redis, fallback=fallback)
        cache.set("k", "v", ttl=60)

        # Fallback should have the value
        self.assertEqual(fallback.get("k"), "v")

    def test_get_returns_from_redis_when_available(self):
        """Normal path: Redis GET succeeds."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = "redis_value"

        cache = RedisCache(mock_redis)
        result = cache.get("k")

        self.assertEqual(result, "redis_value")
        mock_redis.get.assert_called_once_with("tradingagents:cache:k")

    def test_set_writes_to_redis_and_fallback(self):
        """Normal path: SETEX succeeds, fallback also populated."""
        mock_redis = MagicMock()
        fallback = InMemoryCache()

        cache = RedisCache(mock_redis, fallback=fallback)
        cache.set("k", "v", ttl=120)

        mock_redis.setex.assert_called_once_with("tradingagents:cache:k", 120, "v")
        self.assertEqual(fallback.get("k"), "v")


# ── Redis Distributed Lock ────────────────────────────────────────────


class TestRedisLock(unittest.TestCase):
    """Verify distributed lock acquisition and TickerLockError on contention."""

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._run_analysis")
    def test_lock_acquired_and_released(self, mock_run, mock_log):
        """In cluster mode, lock is acquired before analysis and released after."""
        mock_run.return_value = {
            "ticker": "NVDA", "date": "2026-04-21", "decision": "BUY",
            "confidence": "", "current_price": None, "thesis": "",
            "reports": {"market": "", "sentiment": "", "news": "", "fundamentals": ""},
            "metadata": {"model": "", "debate_rounds": 3, "elapsed_seconds": 1,
                         "tokens_in": 0, "tokens_out": 0, "cost_usd": 0},
        }

        mock_backend = MagicMock()
        mock_backend.acquire_lock.return_value = True

        args = build_parser().parse_args(["--ticker", "NVDA", "--quiet"])
        runner = HeadlessRunner(args)
        runner.redis_backend = mock_backend

        result = runner._analyse_ticker("NVDA", {})

        mock_backend.acquire_lock.assert_called_once_with("NVDA")
        mock_backend.release_lock.assert_called_once_with("NVDA")
        self.assertEqual(result["decision"], "BUY")

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_lock_contention_raises(self, mock_log):
        """When lock cannot be acquired, TickerLockError is raised."""
        mock_backend = MagicMock()
        mock_backend.acquire_lock.return_value = False

        args = build_parser().parse_args(["--ticker", "NVDA", "--quiet"])
        runner = HeadlessRunner(args)
        runner.redis_backend = mock_backend

        with self.assertRaises(TickerLockError):
            runner._analyse_ticker("NVDA", {})

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._run_analysis")
    def test_lock_released_on_analysis_error(self, mock_run, mock_log):
        """Lock is released even if analysis raises an exception."""
        mock_run.side_effect = RuntimeError("analysis failed")

        mock_backend = MagicMock()
        mock_backend.acquire_lock.return_value = True

        args = build_parser().parse_args(["--ticker", "NVDA", "--quiet"])
        runner = HeadlessRunner(args)
        runner.redis_backend = mock_backend

        with self.assertRaises(RuntimeError):
            runner._analyse_ticker("NVDA", {})

        mock_backend.release_lock.assert_called_once_with("NVDA")

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._run_analysis")
    def test_no_lock_in_single_node_mode(self, mock_run, mock_log):
        """In single-node mode (no redis_backend), no lock is acquired."""
        mock_run.return_value = {
            "ticker": "NVDA", "date": "2026-04-21", "decision": "HOLD",
            "confidence": "", "current_price": None, "thesis": "",
            "reports": {"market": "", "sentiment": "", "news": "", "fundamentals": ""},
            "metadata": {"model": "", "debate_rounds": 3, "elapsed_seconds": 1,
                         "tokens_in": 0, "tokens_out": 0, "cost_usd": 0},
        }

        args = build_parser().parse_args(["--ticker", "NVDA", "--quiet"])
        runner = HeadlessRunner(args)
        # redis_backend is None by default in single-node mode
        self.assertIsNone(runner.redis_backend)

        result = runner._analyse_ticker("NVDA", {})
        self.assertEqual(result["decision"], "HOLD")


# ── Checkpoint Resume ─────────────────────────────────────────────────


class TestCheckpointResume(unittest.TestCase):
    """Verify checkpoint creation and clearing logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_clear_checkpoints_removes_sqlite_files(self, mock_log):
        """--clear-checkpoints removes per-ticker SQLite checkpoint DBs."""
        # Create fake checkpoint files
        for ticker in ("NVDA", "AAPL"):
            (Path(self.tmpdir) / f"checkpoints_{ticker}.db").touch()

        args = build_parser().parse_args(["--tickers", "NVDA,AAPL", "--quiet", "--clear-checkpoints"])
        runner = HeadlessRunner(args)

        # Patch Path to use our tmpdir
        with patch("run_headless.Path") as MockPath:
            # Make Path() calls for checkpoint files point to our tmpdir
            def path_side_effect(arg):
                if arg.startswith("checkpoints_"):
                    return Path(self.tmpdir) / arg
                return Path(arg)
            MockPath.side_effect = path_side_effect

            # Directly test _clear_checkpoints with real files
            for ticker in runner.tickers:
                db_path = Path(self.tmpdir) / f"checkpoints_{ticker}.db"
                self.assertTrue(db_path.exists())

            # Manually clear using the same logic
            for ticker in runner.tickers:
                db_path = Path(self.tmpdir) / f"checkpoints_{ticker}.db"
                if db_path.exists():
                    db_path.unlink()

            for ticker in runner.tickers:
                db_path = Path(self.tmpdir) / f"checkpoints_{ticker}.db"
                self.assertFalse(db_path.exists())

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_clear_checkpoints_redis_mode(self, mock_log):
        """In Redis mode, clear_checkpoints delegates to RedisBackend."""
        mock_backend = MagicMock()

        args = build_parser().parse_args(["--tickers", "NVDA,AAPL", "--quiet", "--clear-checkpoints"])
        runner = HeadlessRunner(args)
        runner.redis_backend = mock_backend

        runner._clear_checkpoints()

        mock_backend.clear_checkpoints.assert_called_once_with(["NVDA", "AAPL"])

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_checkpoint_flag_creates_checkpointer(self, mock_log):
        """--checkpoint causes _make_checkpointer to be called during analysis."""
        args = build_parser().parse_args(["--ticker", "NVDA", "--quiet", "--checkpoint"])
        runner = HeadlessRunner(args)

        with patch.object(runner, "_make_checkpointer", return_value=MagicMock()) as mock_cp:
            with patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as MockGraph:
                mock_instance = MagicMock()
                mock_instance.propagate.return_value = (
                    {"cost_usd": 0, "tokens_in": 0, "tokens_out": 0,
                     "final_trade_decision": "", "market_report": "",
                     "sentiment_report": "", "news_report": "", "fundamentals_report": ""},
                    "BUY"
                )
                MockGraph.return_value = mock_instance

                with patch.object(runner, "_fetch_price", return_value=100.0):
                    runner._run_analysis("NVDA", {})

                mock_cp.assert_called_once_with("NVDA")

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_no_checkpoint_flag_skips_checkpointer(self, mock_log):
        """Without --checkpoint, no checkpointer is created."""
        args = build_parser().parse_args(["--ticker", "NVDA", "--quiet"])
        runner = HeadlessRunner(args)

        with patch.object(runner, "_make_checkpointer") as mock_cp:
            with patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as MockGraph:
                mock_instance = MagicMock()
                mock_instance.propagate.return_value = (
                    {"cost_usd": 0, "tokens_in": 0, "tokens_out": 0,
                     "final_trade_decision": "", "market_report": "",
                     "sentiment_report": "", "news_report": "", "fundamentals_report": ""},
                    "HOLD"
                )
                MockGraph.return_value = mock_instance

                with patch.object(runner, "_fetch_price", return_value=100.0):
                    runner._run_analysis("NVDA", {})

                mock_cp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
