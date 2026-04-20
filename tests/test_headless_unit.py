"""Unit tests for run_headless.py — CLI args, JSON schema, exit codes, config overrides."""

from __future__ import annotations

import json
import textwrap
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from run_headless import (
    DEPTH_TO_ROUNDS,
    MODEL_TIERS,
    BudgetExceededError,
    HeadlessRunner,
    SpendTracker,
    InMemoryCache,
    CostLogger,
    build_parser,
    parse_config_overrides,
    parse_tickers,
    _coerce_value,
)


# ── CLI argument parsing ──────────────────────────────────────────────


class TestBuildParser(unittest.TestCase):
    """Verify argparse defaults and flag parsing."""

    def _parse(self, *argv: str) -> Namespace:
        return build_parser().parse_args(list(argv))

    def test_defaults(self):
        args = self._parse("--ticker", "NVDA")
        self.assertEqual(args.ticker, "NVDA")
        self.assertEqual(args.depth, "medium")
        self.assertEqual(args.workers, 1)
        self.assertFalse(args.json_output)
        self.assertFalse(args.quiet)
        self.assertFalse(args.checkpoint)
        self.assertFalse(args.enqueue)
        self.assertFalse(args.consume)
        self.assertEqual(args.max_cost, 0)
        self.assertEqual(args.model_tier, "auto")
        self.assertEqual(args.stagger_delay, 5.0)
        self.assertIsNone(args.output_dir)
        self.assertIsNone(args.redis)

    def test_depth_choices(self):
        for depth in ("shallow", "medium", "deep"):
            args = self._parse("--ticker", "X", "--depth", depth)
            self.assertEqual(args.depth, depth)

    def test_invalid_depth_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("--ticker", "X", "--depth", "ultra")

    def test_model_tier_choices(self):
        for tier in ("auto", "deep", "standard", "light"):
            args = self._parse("--ticker", "X", "--model-tier", tier)
            self.assertEqual(args.model_tier, tier)

    def test_json_flag(self):
        args = self._parse("--ticker", "X", "--json")
        self.assertTrue(args.json_output)

    def test_config_append(self):
        args = self._parse("--ticker", "X", "--config", "a=1", "--config", "b=hello")
        self.assertEqual(args.config, ["a=1", "b=hello"])

    def test_output_dir_is_path(self):
        args = self._parse("--ticker", "X", "--output-dir", "/tmp/out")
        self.assertIsInstance(args.output_dir, Path)

    def test_workers_int(self):
        args = self._parse("--ticker", "X", "--workers", "4")
        self.assertEqual(args.workers, 4)


# ── Ticker parsing ────────────────────────────────────────────────────


class TestParseTickers(unittest.TestCase):

    def _ns(self, **kw) -> Namespace:
        defaults = {"ticker": None, "tickers": None}
        defaults.update(kw)
        return Namespace(**defaults)

    def test_single_ticker(self):
        self.assertEqual(parse_tickers(self._ns(ticker="nvda")), ["NVDA"])

    def test_comma_separated(self):
        self.assertEqual(parse_tickers(self._ns(tickers="aapl,msft, goog")), ["AAPL", "MSFT", "GOOG"])

    def test_empty(self):
        self.assertEqual(parse_tickers(self._ns()), [])

    def test_file_reference(self, tmp_path=None):
        """@file.txt reads tickers from file."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("TSLA\nMETA\n")
            f.flush()
            try:
                result = parse_tickers(self._ns(tickers=f"@{f.name}"))
                self.assertEqual(result, ["TSLA", "META"])
            finally:
                os.unlink(f.name)

    def test_ticker_takes_precedence(self):
        """--ticker is used when both --ticker and --tickers are set."""
        self.assertEqual(parse_tickers(self._ns(ticker="NVDA", tickers="AAPL,MSFT")), ["NVDA"])


# ── Config overrides ──────────────────────────────────────────────────


class TestConfigOverrides(unittest.TestCase):

    def test_key_value_parsing(self):
        result = parse_config_overrides(["max_debate_rounds=5", "deep_think_llm=gpt-5.4"])
        self.assertEqual(result, {"max_debate_rounds": 5, "deep_think_llm": "gpt-5.4"})

    def test_bool_coercion(self):
        result = parse_config_overrides(["a=true", "b=false", "c=yes", "d=no"])
        self.assertEqual(result, {"a": True, "b": False, "c": True, "d": False})

    def test_none_coercion(self):
        result = parse_config_overrides(["x=none", "y=null", "z="])
        self.assertEqual(result, {"x": None, "y": None, "z": None})

    def test_float_coercion(self):
        result = parse_config_overrides(["cost=3.14"])
        self.assertEqual(result, {"cost": 3.14})

    def test_malformed_ignored(self):
        result = parse_config_overrides(["no_equals_sign"])
        self.assertEqual(result, {})

    def test_value_with_equals(self):
        """Values containing '=' are preserved."""
        result = parse_config_overrides(["url=http://host?a=1"])
        self.assertEqual(result, {"url": "http://host?a=1"})


class TestCoerceValue(unittest.TestCase):

    def test_int(self):
        self.assertEqual(_coerce_value("42"), 42)

    def test_float(self):
        self.assertEqual(_coerce_value("3.14"), 3.14)

    def test_bool(self):
        self.assertIs(_coerce_value("true"), True)
        self.assertIs(_coerce_value("False"), False)

    def test_string_passthrough(self):
        self.assertEqual(_coerce_value("hello"), "hello")


# ── Depth / model tier mappings ───────────────────────────────────────


class TestMappings(unittest.TestCase):

    def test_depth_to_rounds(self):
        self.assertEqual(DEPTH_TO_ROUNDS, {"shallow": 1, "medium": 3, "deep": 5})

    def test_model_tiers_keys(self):
        self.assertEqual(set(MODEL_TIERS.keys()), {"deep", "standard", "light"})

    def test_model_tiers_are_tuples(self):
        for tier, val in MODEL_TIERS.items():
            self.assertIsInstance(val, tuple)
            self.assertEqual(len(val), 2)


# ── SpendTracker ──────────────────────────────────────────────────────


class TestSpendTracker(unittest.TestCase):

    def test_unlimited(self):
        t = SpendTracker(max_cost=0)
        t.add(1000)
        t.check()  # should not raise
        self.assertEqual(t.total, 1000)

    def test_budget_exceeded(self):
        t = SpendTracker(max_cost=1.0)
        t.add(1.5)
        with self.assertRaises(BudgetExceededError):
            t.check()

    def test_under_budget(self):
        t = SpendTracker(max_cost=10.0)
        t.add(5.0)
        t.check()  # should not raise


# ── InMemoryCache ─────────────────────────────────────────────────────


class TestInMemoryCache(unittest.TestCase):

    def test_get_set(self):
        c = InMemoryCache()
        self.assertIsNone(c.get("k"))
        c.set("k", "v")
        self.assertEqual(c.get("k"), "v")


# ── CostLogger ────────────────────────────────────────────────────────


class TestCostLogger(unittest.TestCase):

    def test_appends_jsonl(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        try:
            cl = CostLogger(path)
            cl.log("NVDA", 0.42, tokens_in=100)
            cl.log("AAPL", 0.10)
            lines = path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            entry = json.loads(lines[0])
            self.assertEqual(entry["ticker"], "NVDA")
            self.assertAlmostEqual(entry["cost_usd"], 0.42)
            self.assertEqual(entry["tokens_in"], 100)
            self.assertIn("ts", entry)
        finally:
            os.unlink(path)


# ── JSON output schema ────────────────────────────────────────────────


class TestJSONOutputSchema(unittest.TestCase):
    """Verify format_report produces valid JSON matching the spec schema."""

    REQUIRED_KEYS = {"ticker", "date", "decision", "confidence", "current_price", "thesis", "reports", "metadata"}
    REPORT_KEYS = {"market", "sentiment", "news", "fundamentals"}
    METADATA_KEYS = {"model", "debate_rounds", "elapsed_seconds", "tokens_in", "tokens_out", "cost_usd"}

    def _make_result(self, **overrides) -> dict:
        base = {
            "ticker": "NVDA",
            "date": "2026-04-17",
            "decision": "BUY",
            "confidence": "high",
            "current_price": 198.35,
            "thesis": "Strong growth outlook.",
            "reports": {"market": "m", "sentiment": "s", "news": "n", "fundamentals": "f"},
            "metadata": {
                "model": "gpt-5.4",
                "debate_rounds": 3,
                "elapsed_seconds": 142,
                "tokens_in": 45000,
                "tokens_out": 8200,
                "cost_usd": 0.42,
            },
        }
        base.update(overrides)
        return base

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_json_output_valid(self, mock_log):
        args = build_parser().parse_args(["--ticker", "NVDA", "--json"])
        runner = HeadlessRunner(args)
        results = [self._make_result()]
        output = runner.format_report(results)
        parsed = json.loads(output)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        item = parsed[0]
        self.assertTrue(self.REQUIRED_KEYS.issubset(item.keys()))
        self.assertTrue(self.REPORT_KEYS.issubset(item["reports"].keys()))
        self.assertTrue(self.METADATA_KEYS.issubset(item["metadata"].keys()))

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_non_json_output_is_text(self, mock_log):
        args = build_parser().parse_args(["--ticker", "NVDA"])
        runner = HeadlessRunner(args)
        results = [self._make_result()]
        output = runner.format_report(results)
        self.assertIn("NVDA", output)
        self.assertIn("BUY", output)
        # Should NOT be valid JSON
        with self.assertRaises(json.JSONDecodeError):
            json.loads(output)


# ── Exit codes ────────────────────────────────────────────────────────


class TestExitCodes(unittest.TestCase):
    """Verify run() returns correct exit codes per spec."""

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_no_tickers_returns_2(self, mock_log):
        args = build_parser().parse_args([])
        runner = HeadlessRunner(args)
        self.assertEqual(runner.run(), 2)

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._analyse_ticker")
    def test_all_succeed_returns_0(self, mock_analyse, mock_log):
        mock_analyse.return_value = {
            "ticker": "NVDA", "date": "2026-04-17", "decision": "BUY",
            "confidence": "", "current_price": None, "thesis": "",
            "reports": {"market": "", "sentiment": "", "news": "", "fundamentals": ""},
            "metadata": {"model": "", "debate_rounds": 3, "elapsed_seconds": 1,
                         "tokens_in": 0, "tokens_out": 0, "cost_usd": 0},
        }
        args = build_parser().parse_args(["--ticker", "NVDA", "--quiet"])
        runner = HeadlessRunner(args)
        self.assertEqual(runner.run(), 0)

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._analyse_ticker")
    def test_partial_failure_returns_1(self, mock_analyse, mock_log):
        def side_effect(ticker, ctx):
            if ticker == "BAD":
                raise RuntimeError("boom")
            return {
                "ticker": ticker, "date": "2026-04-17", "decision": "HOLD",
                "confidence": "", "current_price": None, "thesis": "",
                "reports": {"market": "", "sentiment": "", "news": "", "fundamentals": ""},
                "metadata": {"model": "", "debate_rounds": 3, "elapsed_seconds": 1,
                             "tokens_in": 0, "tokens_out": 0, "cost_usd": 0},
            }
        mock_analyse.side_effect = side_effect
        args = build_parser().parse_args(["--tickers", "NVDA,BAD", "--quiet"])
        runner = HeadlessRunner(args)
        self.assertEqual(runner.run(), 1)

    @patch("run_headless.HeadlessRunner._setup_logging")
    @patch("run_headless.HeadlessRunner._analyse_ticker")
    def test_all_fail_returns_2(self, mock_analyse, mock_log):
        mock_analyse.side_effect = RuntimeError("boom")
        args = build_parser().parse_args(["--ticker", "NVDA", "--quiet"])
        runner = HeadlessRunner(args)
        self.assertEqual(runner.run(), 2)

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_enqueue_without_redis_exits(self, mock_log):
        with self.assertRaises(SystemExit):
            args = build_parser().parse_args(["--ticker", "X", "--enqueue"])
            HeadlessRunner(args)

    @patch("run_headless.HeadlessRunner._setup_logging")
    def test_consume_without_redis_exits(self, mock_log):
        with self.assertRaises(SystemExit):
            args = build_parser().parse_args(["--consume"])
            HeadlessRunner(args)


if __name__ == "__main__":
    unittest.main()
