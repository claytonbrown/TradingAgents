#!/usr/bin/env python3
"""Headless runner for TradingAgents — CLI entry point for cron/CI/batch analysis."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger("headless")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_headless",
        description="TradingAgents headless runner — batch analysis for cron/CI",
    )

    # --- Core ---
    g = p.add_argument_group("core")
    g.add_argument("--ticker", type=str, help="Single ticker symbol")
    g.add_argument("--tickers", type=str, help="Comma-separated tickers or @file.txt")
    g.add_argument("--date", type=str, default=str(date.today()), help="Analysis date (default: today)")
    g.add_argument("--depth", choices=["shallow", "medium", "deep"], default="medium", help="Debate depth")
    g.add_argument("--output-dir", type=Path, help="Write per-ticker JSON + markdown here")
    g.add_argument("--json", action="store_true", dest="json_output", help="Structured JSON to stdout")
    g.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    g.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="Override DEFAULT_CONFIG keys")

    # --- Scaling ---
    g = p.add_argument_group("scaling")
    g.add_argument("--workers", type=int, default=1, help="Concurrency level")
    g.add_argument("--stagger-delay", type=float, default=5.0, help="Seconds between worker launches")
    g.add_argument("--redis", type=str, default=None, help="Redis URL — enables cluster mode")
    g.add_argument("--enqueue", action="store_true", help="Push tickers to Redis queue and exit")
    g.add_argument("--consume", action="store_true", help="Pull tickers from Redis queue")

    # --- Resilience ---
    g = p.add_argument_group("resilience")
    g.add_argument("--checkpoint", action="store_true", help="Enable crash recovery")
    g.add_argument("--clear-checkpoints", action="store_true", help="Force fresh start")
    g.add_argument("--reuse-threshold", type=float, default=0, help="Skip if price moved <N%% since last run")

    # --- Cost control ---
    g = p.add_argument_group("cost control")
    g.add_argument("--max-cost", type=float, default=0, help="USD budget cap (0=unlimited)")
    g.add_argument("--model-tier", choices=["auto", "deep", "standard", "light"], default="auto")

    # --- Observability ---
    g = p.add_argument_group("observability")
    g.add_argument("--verbose", action="store_true", help="Full LangGraph step tracing")

    return p


def parse_tickers(args: argparse.Namespace) -> list[str]:
    """Resolve ticker list from --ticker / --tickers flags."""
    if args.ticker:
        return [args.ticker.upper()]
    if not args.tickers:
        return []
    raw = args.tickers
    if raw.startswith("@"):
        path = Path(raw[1:])
        raw = path.read_text().strip()
    # Support both comma-separated and newline-separated tickers
    raw = raw.replace("\n", ",")
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _coerce_value(v: str) -> Any:
    """Best-effort coerce string to int/float/bool/None, else keep as str."""
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    if v.lower() in ("none", "null", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_config_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse --config KEY=VALUE pairs into a dict with type coercion."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            logger.warning("Ignoring malformed --config %r (expected KEY=VALUE)", pair)
            continue
        k, v = pair.split("=", 1)
        overrides[k.strip()] = _coerce_value(v.strip())
    return overrides


DEPTH_TO_ROUNDS = {"shallow": 1, "medium": 3, "deep": 5}


class HeadlessRunner:
    """Base headless runner — subclass and override hooks for customisation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tickers = parse_tickers(args)
        self.config_overrides = parse_config_overrides(args.config)
        self._setup_logging()

    # ------------------------------------------------------------------
    # Hooks (override in subclasses)
    # ------------------------------------------------------------------

    def pre_analysis(self) -> dict[str, Any]:
        """Called once before the analysis loop. Return a context dict."""
        return {}

    def post_analysis(self, ticker: str, result: dict[str, Any]) -> dict[str, Any]:
        """Called after each ticker's analysis. Enrich/transform the result."""
        return result

    def format_report(self, results: list[dict[str, Any]]) -> str:
        """Format final output. JSON if --json, else human-readable summary."""
        import json

        if self.args.json_output:
            return json.dumps(results, indent=2, default=str)

        lines: list[str] = []
        for r in results:
            lines.append(f"{r['ticker']}  {r['decision']}  ({r['metadata'].get('elapsed_seconds', '?')}s)")
            if r.get("thesis"):
                # First line of thesis as summary
                first = r["thesis"].split("\n", 1)[0][:120]
                lines.append(f"  {first}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Main entry point. Returns exit code (0/1/2)."""
        if not self.tickers:
            logger.error("No tickers specified (use --ticker or --tickers)")
            return 2

        total = len(self.tickers)
        logger.info("Batch: %d ticker(s) — %s", total, ", ".join(self.tickers))

        ctx = self.pre_analysis()
        results: list[dict[str, Any]] = []
        failures = 0

        for i, ticker in enumerate(self.tickers, 1):
            logger.info("[%d/%d] %s", i, total, ticker)
            try:
                result = self._analyse_ticker(ticker, ctx)
                result = self.post_analysis(ticker, result)
                results.append(result)
            except Exception:
                logger.exception("Failed: %s", ticker)
                failures += 1

        logger.info("Done: %d succeeded, %d failed", len(results), failures)

        # Write per-ticker files if --output-dir
        if self.args.output_dir and results:
            self._write_output_dir(results)

        # Output
        if results:
            output = self.format_report(results)
            if output:
                print(output)

        # Exit codes: 0=all ok, 1=partial, 2=total failure
        if failures == 0:
            return 0
        if results:
            return 1
        return 2

    def _analyse_ticker(self, ticker: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Run analysis for a single ticker via TradingAgentsGraph.propagate()."""
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = DEPTH_TO_ROUNDS[self.args.depth]
        config.update(self.config_overrides)

        logger.info("Analysing %s (date=%s, depth=%s)", ticker, self.args.date, self.args.depth)
        t0 = time.time()

        ta = TradingAgentsGraph(debug=self.args.verbose, config=config)
        final_state, decision = ta.propagate(ticker, self.args.date)

        elapsed = time.time() - t0

        return {
            "ticker": ticker,
            "date": self.args.date,
            "decision": decision.strip().upper(),
            "confidence": "",
            "thesis": final_state.get("final_trade_decision", ""),
            "reports": {
                "market": final_state.get("market_report", ""),
                "sentiment": final_state.get("sentiment_report", ""),
                "news": final_state.get("news_report", ""),
                "fundamentals": final_state.get("fundamentals_report", ""),
            },
            "metadata": {
                "model": config.get("deep_think_llm", ""),
                "debate_rounds": config["max_debate_rounds"],
                "elapsed_seconds": round(elapsed, 1),
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_output_dir(self, results: list[dict[str, Any]]) -> None:
        """Write per-ticker JSON and markdown decision files to --output-dir."""
        import json

        out = self.args.output_dir
        out.mkdir(parents=True, exist_ok=True)

        for r in results:
            ticker = r["ticker"]
            # JSON
            (out / f"{ticker}.json").write_text(json.dumps(r, indent=2, default=str))
            # Markdown
            md = f"# {ticker} — {r['decision']}\n\n"
            md += f"Date: {r['date']}\n\n"
            if r.get("thesis"):
                md += f"## Decision\n\n{r['thesis']}\n"
            (out / f"{ticker}.md").write_text(md)

        logger.info("Wrote %d result(s) to %s", len(results), out)

    def _setup_logging(self) -> None:
        level = logging.WARNING if self.args.quiet else (logging.DEBUG if self.args.verbose else logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root = logging.getLogger("headless")
        root.setLevel(level)
        root.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runner = HeadlessRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
