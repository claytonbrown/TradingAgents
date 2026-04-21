"""Optional Redis cache for TradingAgents API responses.

When Redis is available, caches API responses with configurable TTLs per
namespace. When unavailable, all methods gracefully return None so the
system operates identically to an uncached run.

Namespaces and default TTLs (seconds):
    market        3600   — yfinance price/target/consensus
    analysis     86400   — per-ticker LLM analysis results
    news         43200   — news/sentiment API responses (12h)
    fundamentals 86400   — financial statements, ratios (1 day)
    indicators   14400   — technical indicator computations
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTLS: dict[str, int] = {
    "market": 3600,
    "analysis": 86400,
    "news": 43200,
    "fundamentals": 86400,
    "indicators": 14400,
}


def _find_recent_analysis(
    ticker: str,
    date: str,
    ttl_seconds: int,
    analyses_dir: str | Path,
) -> dict[str, Any] | None:
    """Scan ``analyses_dir/{TICKER}_*/summary.json`` and return the most recent within *ttl_seconds* of *date*.

    Args:
        ticker: Stock symbol (e.g. ``"NVDA"``).
        date: Reference date as ``YYYY-MM-DD``.
        ttl_seconds: Maximum age in seconds for a cached analysis to be valid.
        analyses_dir: Root directory containing ``{TICKER}_{date}[_{tier}]/summary.json`` folders.

    Returns:
        Parsed summary dict from the most recent valid ``summary.json``, or ``None``.
    """
    base = Path(analyses_dir)
    if not base.is_dir():
        return None

    ref = datetime.strptime(date, "%Y-%m-%d")
    best: tuple[datetime, dict[str, Any]] | None = None

    for summary_path in sorted(base.glob(f"{ticker}_*/summary.json"), reverse=True):
        dir_name = summary_path.parent.name  # e.g. NVDA_2026-04-14 or NVDA_2026-04-14_bronze
        parts = dir_name.split("_", 1)
        if len(parts) < 2:
            continue
        # Extract date portion (second segment, may have trailing _tier)
        date_str = parts[1][:10]
        try:
            analysis_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        age_seconds = (ref - analysis_date).total_seconds()
        if age_seconds < 0 or age_seconds > ttl_seconds:
            continue

        # Candidate is within TTL window — keep the most recent
        if best is None or analysis_date > best[0]:
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                best = (analysis_date, data)
            except (json.JSONDecodeError, OSError):
                continue

    if best is None:
        return None

    age_days = (ref - best[0]).total_seconds() / 86400
    logger.info(
        "[TTL SCAN] %s: reusing %s analysis (%.1fd old, TTL %dd)",
        ticker,
        best[0].strftime("%Y-%m-%d"),
        age_days,
        ttl_seconds // 86400,
    )
    return best[1]


class AnalysisCache:
    """Redis-backed cache with namespace isolation and graceful fallback."""

    def __init__(self, url: str = "redis://localhost:6379/0", ttls: dict[str, int] | None = None, config: dict | None = None):
        # Start with hardcoded defaults
        base_ttls = dict(DEFAULT_TTLS)
        # If platform config provided, apply its default TTL and per-namespace overrides
        if config:
            default_ttl = config.get("cache_ttl_seconds")
            if default_ttl is not None:
                base_ttls = {ns: default_ttl for ns in base_ttls}
            base_ttls.update(config.get("cache_ttl_overrides", {}))
        # Explicit ttls parameter takes highest precedence
        base_ttls.update(ttls or {})
        self._ttls = base_ttls
        self._stats: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0, "skips": 0})
        self._redis = None
        try:
            import redis
            pool = redis.ConnectionPool.from_url(url)
            self._redis = redis.Redis(connection_pool=pool)
            self._redis.ping()
            logger.info("Redis cache connected: %s", url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s), running without cache", exc)
            self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _key(self, namespace: str, key: str) -> str:
        return f"ta:{namespace}:{key}"

    def skip(self, namespace: str = "market") -> None:
        """Record a cache skip (lookup bypassed because cache is unavailable or disabled)."""
        self._stats[namespace]["skips"] += 1

    def get(self, key: str, namespace: str = "market", *, source: str = "redis",
            ticker: str | None = None, price_now: float | None = None) -> Any | None:
        """Return cached value or None (cache miss / unavailable).

        When a hit occurs, logs staleness metadata including age, source, and
        optional price delta.  Emits a warning when the entry is older than 50%
        of the namespace TTL.
        """
        if not self.available:
            self._stats[namespace]["skips"] += 1
            return None
        try:
            raw = self._redis.get(self._key(namespace, key))
            if raw is None:
                self._stats[namespace]["misses"] += 1
                return None
            self._stats[namespace]["hits"] += 1
            envelope = json.loads(raw)
            # Unwrap envelope written by set()
            if isinstance(envelope, dict) and "_cached_at" in envelope:
                cached_at = envelope["_cached_at"]
                cached_price = envelope.get("_cached_price")
                data = envelope["_data"]
            else:
                # Legacy entry without envelope
                cached_at = None
                cached_price = None
                data = envelope
            self._log_hit(key, namespace, source, ticker, cached_at, cached_price, price_now)
            return data
        except Exception:
            self._stats[namespace]["misses"] += 1
            return None

    def set(self, key: str, data: Any, namespace: str = "market", ttl: int | None = None,
            *, price: float | None = None) -> bool:
        """Store value with TTL. Returns True on success.

        Wraps *data* in an envelope containing ``_cached_at`` (epoch) and an
        optional ``_cached_price`` so that future ``get()`` calls can report
        staleness and price delta.
        """
        if not self.available:
            return False
        try:
            ex = ttl if ttl is not None else self._ttls.get(namespace, 3600)
            envelope: dict[str, Any] = {"_cached_at": time.time(), "_data": data}
            if price is not None:
                envelope["_cached_price"] = price
            self._redis.set(self._key(namespace, key), json.dumps(envelope, default=str), ex=ex)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Staleness logging
    # ------------------------------------------------------------------

    def _log_hit(self, key: str, namespace: str, source: str,
                 ticker: str | None, cached_at: float | None,
                 cached_price: float | None, price_now: float | None) -> None:
        """Emit [CACHE HIT] and optional [CACHE STALE WARNING] log lines."""
        label = f"{ticker} {namespace}" if ticker else f"{namespace}:{key}"
        if cached_at is None:
            logger.info("[CACHE HIT] %s (source=%s)", label, source)
            return

        age_s = time.time() - cached_at
        age_days = age_s / 86400
        cached_date = datetime.fromtimestamp(cached_at, tz=timezone.utc).strftime("%Y-%m-%d")

        # Price delta fragment
        delta_str = ""
        if cached_price and price_now and cached_price != 0:
            delta_pct = (price_now - cached_price) / cached_price * 100
            delta_str = f", {delta_pct:+.1f}% price delta"

        age_label = f"{age_days:.0f}d" if age_days >= 1 else f"{age_s / 3600:.1f}h"
        logger.info("[CACHE HIT] %s: cached %s (%s old%s, source=%s)",
                    label, cached_date, age_label, delta_str, source)

        # Stale warning when age exceeds 50% of namespace TTL
        ns_ttl = self._ttls.get(namespace, 3600)
        if age_s > ns_ttl * 0.5:
            ttl_days = ns_ttl / 86400
            ttl_label = f"{ttl_days:.0f}d" if ttl_days >= 1 else f"{ns_ttl / 3600:.1f}h"
            logger.warning("[CACHE STALE WARNING] %s: cached %s (%s old, approaching %s TTL)",
                           label, cached_date, age_label, ttl_label)

    def stats(self) -> dict[str, dict[str, int]]:
        """Return per-namespace hit/miss counts."""
        return dict(self._stats)

    def log_stats(self) -> None:
        """Log cache statistics summary."""
        total_hits = sum(s["hits"] for s in self._stats.values())
        total_misses = sum(s["misses"] for s in self._stats.values())
        total_skips = sum(s["skips"] for s in self._stats.values())
        logger.info("Cache stats — hits: %d, misses: %d, skips: %d", total_hits, total_misses, total_skips)
        for ns, s in sorted(self._stats.items()):
            logger.info("  %s: hits=%d misses=%d skips=%d", ns, s["hits"], s["misses"], s["skips"])
