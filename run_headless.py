#!/usr/bin/env python3
"""Headless runner for TradingAgents — CLI entry point for cron/CI/batch analysis."""

from __future__ import annotations

import argparse
import json as _json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("headless")

# Queue name used for Redis task queue (tasks 15+)
REDIS_QUEUE_KEY = "tradingagents:queue"


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


class BudgetExceededError(RuntimeError):
    """Raised when --max-cost budget cap is reached."""


class SpendTracker:
    """Thread-safe in-memory USD spend tracker for --max-cost budget cap."""

    def __init__(self, max_cost: float = 0) -> None:
        self.max_cost = max_cost  # 0 = unlimited
        self._total = 0.0
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        with self._lock:
            return self._total

    def check(self) -> None:
        """Raise BudgetExceededError if budget is exhausted."""
        if self.max_cost > 0 and self.total >= self.max_cost:
            raise BudgetExceededError(f"Budget cap ${self.max_cost:.2f} reached (spent ${self._total:.2f})")

    def add(self, amount: float) -> None:
        """Record spend. Raises BudgetExceededError if cap exceeded after add."""
        with self._lock:
            self._total += amount
        if self.max_cost > 0 and self._total > self.max_cost:
            logger.warning("Budget cap exceeded: $%.2f / $%.2f", self._total, self.max_cost)


class CostLogger:
    """Thread-safe append-only JSONL cost logger."""

    def __init__(self, path: Path = Path("cost_log.jsonl")) -> None:
        self._path = path
        self._lock = threading.Lock()

    def log(self, ticker: str, cost_usd: float, **extra: Any) -> None:
        """Append a cost entry as a single JSON line."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "cost_usd": cost_usd,
            **extra,
        }
        line = _json.dumps(entry, default=str) + "\n"
        with self._lock:
            with self._path.open("a") as f:
                f.write(line)


def _require_redis(module_name: str = "redis"):
    """Import and return the redis module, raising a clear error if missing."""
    try:
        import redis
        return redis
    except ImportError:
        raise SystemExit(
            f"Redis mode requires the '{module_name}' package. "
            "Install with: pip install tradingagents[distributed]"
        )


class RedisSpendTracker:
    """Redis-backed USD spend tracker using INCRBYFLOAT for cross-node budget."""

    REDIS_KEY = "tradingagents:spend_total"

    def __init__(self, rconn: Any, max_cost: float = 0) -> None:
        self.max_cost = max_cost
        self._r = rconn

    @property
    def total(self) -> float:
        val = self._r.get(self.REDIS_KEY)
        return float(val) if val else 0.0

    def check(self) -> None:
        if self.max_cost > 0 and self.total >= self.max_cost:
            raise BudgetExceededError(
                f"Budget cap ${self.max_cost:.2f} reached (spent ${self.total:.2f})"
            )

    def add(self, amount: float) -> None:
        new_total = self._r.incrbyfloat(self.REDIS_KEY, amount)
        if self.max_cost > 0 and float(new_total) > self.max_cost:
            logger.warning("Budget cap exceeded: $%.2f / $%.2f", float(new_total), self.max_cost)


class RedisCostLogger:
    """Cost logger that writes to a Redis stream (XADD) with local JSONL fallback."""

    STREAM_KEY = "tradingagents:cost_log"

    def __init__(self, rconn: Any, fallback_path: Path = Path("cost_log.jsonl")) -> None:
        self._r = rconn
        self._fallback = CostLogger(fallback_path)

    def log(self, ticker: str, cost_usd: float, **extra: Any) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "cost_usd": str(cost_usd),
            **{k: str(v) for k, v in extra.items()},
        }
        try:
            self._r.xadd(self.STREAM_KEY, entry)
        except Exception:
            logger.debug("Redis XADD failed, falling back to local JSONL")
            self._fallback.log(ticker, cost_usd, **extra)


class InMemoryCache:
    """Thread-safe in-memory cache for single-node mode (per-run only)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """Store value. TTL is ignored for in-memory (cache lives per-run)."""
        with self._lock:
            self._store[key] = value


class RedisCache:
    """Redis-backed cache using GET/SETEX with TTL and graceful fallback."""

    KEY_PREFIX = "tradingagents:cache:"

    def __init__(self, rconn: Any, fallback: InMemoryCache | None = None) -> None:
        self._r = rconn
        self._fallback = fallback or InMemoryCache()

    def get(self, key: str) -> str | None:
        rkey = self.KEY_PREFIX + key
        try:
            return self._r.get(rkey)
        except Exception:
            logger.debug("Redis GET failed for %s, falling back to memory", key)
            return self._fallback.get(key)

    def set(self, key: str, value: str, ttl: int = 3600) -> None:
        rkey = self.KEY_PREFIX + key
        try:
            self._r.setex(rkey, ttl, value)
        except Exception:
            logger.debug("Redis SETEX failed for %s, falling back to memory", key)
        # Always populate fallback so reads degrade gracefully
        self._fallback.set(key, value, ttl)


class RedisBackend:
    """Holds the Redis connection and provides factory methods for Redis-backed components."""

    def __init__(self, redis_url: str) -> None:
        redis_mod = _require_redis()
        self.url = redis_url
        self.conn = redis_mod.Redis.from_url(redis_url, decode_responses=True)
        # Verify connectivity
        self.conn.ping()
        logger.info("Redis connected: %s", redis_url)

    def make_spend_tracker(self, max_cost: float) -> RedisSpendTracker:
        return RedisSpendTracker(self.conn, max_cost=max_cost)

    def make_cost_logger(self, fallback_path: Path) -> RedisCostLogger:
        return RedisCostLogger(self.conn, fallback_path=fallback_path)

    def make_cache(self) -> RedisCache:
        return RedisCache(self.conn)

    def make_checkpointer(self, ticker: str) -> Any:
        """Create a Redis-backed checkpointer keyed by ticker."""
        try:
            from langgraph.checkpoint.redis import RedisSaver
        except ImportError:
            raise SystemExit(
                "Redis checkpoints require 'langgraph-checkpoint-redis'. "
                "Install with: pip install tradingagents[distributed]"
            )
        return RedisSaver.from_conn_string(self.url)

    def clear_checkpoints(self, tickers: list[str]) -> None:
        """Delete Redis checkpoint keys for the given tickers."""
        for ticker in tickers:
            pattern = f"checkpoint:{ticker}:*"
            cursor = 0
            while True:
                cursor, keys = self.conn.scan(cursor, match=pattern, count=100)
                if keys:
                    self.conn.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Cleared Redis checkpoints for %s", ticker)


class HeadlessRunner:
    """Base headless runner — subclass and override hooks for customisation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tickers = parse_tickers(args)
        self.config_overrides = parse_config_overrides(args.config)
        self._setup_logging()

        # Validate Redis-only flags
        if (args.enqueue or args.consume) and not args.redis:
            raise SystemExit("--enqueue and --consume require --redis")

        # Mode selection: --redis → cluster, else single-node
        self.redis_backend: RedisBackend | None = None
        cost_log_path = args.output_dir / "cost_log.jsonl" if args.output_dir else Path("cost_log.jsonl")

        if args.redis:
            self.redis_backend = RedisBackend(args.redis)
            self.spend = self.redis_backend.make_spend_tracker(max_cost=args.max_cost)
            self.cost_logger = self.redis_backend.make_cost_logger(fallback_path=cost_log_path)
            self.cache: InMemoryCache | RedisCache = self.redis_backend.make_cache()
            logger.info("Cluster mode: all shared state via Redis")
        else:
            self.spend = SpendTracker(max_cost=args.max_cost)
            self.cost_logger = CostLogger(cost_log_path)
            self.cache = InMemoryCache()
            logger.info("Single-node mode: threads + SQLite")

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

        if self.args.clear_checkpoints:
            self._clear_checkpoints()

        total = len(self.tickers)
        logger.info("Batch: %d ticker(s) — %s", total, ", ".join(self.tickers))

        ctx = self.pre_analysis()
        workers = max(1, self.args.workers)

        try:
            if workers > 1 and total > 1:
                results, failures = self._run_parallel(ctx, workers)
            else:
                results, failures = self._run_sequential(ctx)
        except BudgetExceededError as exc:
            logger.error("%s", exc)
            return 2

        logger.info("Done: %d succeeded, %d failed", len(results), failures)

        if self.args.output_dir and results:
            self._write_output_dir(results)

        if results:
            output = self.format_report(results)
            if output:
                print(output)

        if failures == 0:
            return 0
        if results:
            return 1
        return 2

    def _run_sequential(self, ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
        """Run tickers one at a time."""
        results: list[dict[str, Any]] = []
        failures = 0
        total = len(self.tickers)
        for i, ticker in enumerate(self.tickers, 1):
            self.spend.check()
            logger.info("[%d/%d] %s", i, total, ticker)
            try:
                result = self._analyse_ticker(ticker, ctx)
                result = self.post_analysis(ticker, result)
                results.append(result)
            except Exception:
                logger.exception("Failed: %s", ticker)
                failures += 1
        return results, failures

    def _run_parallel(self, ctx: dict[str, Any], workers: int) -> tuple[list[dict[str, Any]], int]:
        """Run tickers concurrently via ThreadPoolExecutor."""
        results: list[dict[str, Any]] = []
        failures = 0
        stagger = self.args.stagger_delay

        def _worker(worker_id: int, ticker: str) -> dict[str, Any]:
            wlog = logging.getLogger(f"headless.W{worker_id}")
            wlog.info("[W%d] Starting %s", worker_id, ticker)
            result = self._analyse_ticker(ticker, ctx)
            result = self.post_analysis(ticker, result)
            wlog.info("[W%d] Finished %s — %s", worker_id, ticker, result["decision"])
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, ticker in enumerate(self.tickers):
                if i > 0 and stagger > 0:
                    time.sleep(stagger)
                worker_id = (i % workers) + 1
                fut = pool.submit(_worker, worker_id, ticker)
                futures[fut] = ticker

            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    results.append(fut.result())
                except Exception:
                    logger.exception("Failed: %s", ticker)
                    failures += 1

        return results, failures

    def _analyse_ticker(self, ticker: str, ctx: dict[str, Any]) -> dict[str, Any]:
        """Run analysis for a single ticker via TradingAgentsGraph.propagate()."""
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = DEPTH_TO_ROUNDS[self.args.depth]
        config.update(self.config_overrides)

        checkpointer = self._make_checkpointer(ticker) if self.args.checkpoint else None

        logger.info("Analysing %s (date=%s, depth=%s)", ticker, self.args.date, self.args.depth)
        t0 = time.time()

        ta = TradingAgentsGraph(debug=self.args.verbose, config=config, checkpointer=checkpointer)
        final_state, decision = ta.propagate(ticker, self.args.date)

        elapsed = time.time() - t0

        # Log cost per ticker (estimate from token usage if available)
        cost_usd = final_state.get("cost_usd", 0.0)
        if cost_usd:
            self.spend.add(cost_usd)
            self.cost_logger.log(ticker, cost_usd, elapsed_seconds=round(elapsed, 1))

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

    def _clear_checkpoints(self) -> None:
        """Delete existing checkpoints (SQLite files or Redis keys)."""
        if self.redis_backend:
            self.redis_backend.clear_checkpoints(self.tickers)
            return
        for ticker in self.tickers:
            db_path = Path(f"checkpoints_{ticker}.db")
            if db_path.exists():
                db_path.unlink()
                logger.info("Cleared checkpoint: %s", db_path)

    def _make_checkpointer(self, ticker: str):
        """Create a per-ticker checkpointer (SQLite or Redis based on mode)."""
        if self.redis_backend:
            return self.redis_backend.make_checkpointer(ticker)
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = Path(f"checkpoints_{ticker}.db")
        return SqliteSaver.from_conn_string(str(db_path))

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
