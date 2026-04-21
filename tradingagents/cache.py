"""Optional Redis cache for TradingAgents API responses.

When Redis is available, caches API responses with configurable TTLs per
namespace. When unavailable, all methods gracefully return None so the
system operates identically to an uncached run.

Namespaces and default TTLs (seconds):
    market        3600   — yfinance price/target/consensus
    analysis     86400   — per-ticker LLM analysis results
    news         14400   — news API responses
    fundamentals 43200   — financial statements, ratios
    indicators   14400   — technical indicator computations
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTLS: dict[str, int] = {
    "market": 3600,
    "analysis": 86400,
    "news": 14400,
    "fundamentals": 43200,
    "indicators": 14400,
}


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
        self._stats: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
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

    def get(self, key: str, namespace: str = "market") -> Any | None:
        """Return cached value or None (cache miss / unavailable)."""
        if not self.available:
            return None
        try:
            raw = self._redis.get(self._key(namespace, key))
            if raw is None:
                self._stats[namespace]["misses"] += 1
                return None
            self._stats[namespace]["hits"] += 1
            return json.loads(raw)
        except Exception:
            self._stats[namespace]["misses"] += 1
            return None

    def set(self, key: str, data: Any, namespace: str = "market", ttl: int | None = None) -> bool:
        """Store value with TTL. Returns True on success."""
        if not self.available:
            return False
        try:
            ex = ttl if ttl is not None else self._ttls.get(namespace, 3600)
            self._redis.set(self._key(namespace, key), json.dumps(data, default=str), ex=ex)
            return True
        except Exception:
            return False

    def stats(self) -> dict[str, dict[str, int]]:
        """Return per-namespace hit/miss counts."""
        return dict(self._stats)

    def log_stats(self) -> None:
        """Log cache statistics summary."""
        total_hits = sum(s["hits"] for s in self._stats.values())
        total_misses = sum(s["misses"] for s in self._stats.values())
        logger.info("Cache stats — hits: %d, misses: %d", total_hits, total_misses)
        for ns, s in sorted(self._stats.items()):
            logger.info("  %s: hits=%d misses=%d", ns, s["hits"], s["misses"])
