"""Unit tests for cache-hit logging — Task 23.

Verifies [CACHE HIT] log lines with staleness metadata and
[CACHE STALE WARNING] emitted when entry age exceeds 50% of namespace TTL.
"""

from __future__ import annotations

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest


def _make_cache(ttls: dict[str, int] | None = None, **kwargs):
    """Create AnalysisCache with mocked Redis."""
    instance = MagicMock()
    with patch("redis.ConnectionPool.from_url"), \
         patch("redis.Redis", return_value=instance):
        from tradingagents.cache import AnalysisCache
        cache = AnalysisCache(ttls=ttls, **kwargs)
    return cache, instance


def _envelope(data, cached_at: float, price: float | None = None) -> str:
    """Build a JSON envelope as stored by set()."""
    env: dict = {"_cached_at": cached_at, "_data": data}
    if price is not None:
        env["_cached_price"] = price
    return json.dumps(env)


# ---------------------------------------------------------------------------
# [CACHE HIT] basic log line
# ---------------------------------------------------------------------------

def test_cache_hit_logged_with_age(caplog):
    cache, r = _make_cache(ttls={"market": 7200})
    # Entry cached 1 hour ago — under 50% of 7200s TTL
    r.get.return_value = _envelope("data", time.time() - 3600)
    with caplog.at_level(logging.INFO, logger="tradingagents.cache"):
        cache.get("NVDA", namespace="market", source="redis", ticker="NVDA")
    assert "[CACHE HIT]" in caplog.text
    assert "NVDA" in caplog.text
    assert "source=redis" in caplog.text


def test_cache_hit_includes_price_delta(caplog):
    cache, r = _make_cache(ttls={"market": 7200})
    r.get.return_value = _envelope("data", time.time() - 1800, price=100.0)
    with caplog.at_level(logging.INFO, logger="tradingagents.cache"):
        cache.get("AAPL", namespace="market", source="redis",
                  ticker="AAPL", price_now=105.0)
    assert "+5.0% price delta" in caplog.text


def test_cache_hit_negative_price_delta(caplog):
    cache, r = _make_cache(ttls={"market": 7200})
    r.get.return_value = _envelope("data", time.time() - 1800, price=100.0)
    with caplog.at_level(logging.INFO, logger="tradingagents.cache"):
        cache.get("TSLA", namespace="market", source="redis",
                  ticker="TSLA", price_now=90.0)
    assert "-10.0% price delta" in caplog.text


def test_cache_hit_source_filesystem(caplog):
    cache, r = _make_cache(ttls={"analysis": 86400})
    r.get.return_value = _envelope("data", time.time() - 3600)
    with caplog.at_level(logging.INFO, logger="tradingagents.cache"):
        cache.get("VTI", namespace="analysis", source="filesystem", ticker="VTI")
    assert "source=filesystem" in caplog.text


# ---------------------------------------------------------------------------
# [CACHE STALE WARNING] when age > 50% TTL
# ---------------------------------------------------------------------------

def test_stale_warning_emitted_when_age_exceeds_50pct_ttl(caplog):
    cache, r = _make_cache(ttls={"analysis": 86400})  # 1-day TTL
    # Entry cached 13 hours ago — 54% of 86400s
    r.get.return_value = _envelope("data", time.time() - 46800)
    with caplog.at_level(logging.WARNING, logger="tradingagents.cache"):
        cache.get("VTI", namespace="analysis", source="redis", ticker="VTI")
    assert "[CACHE STALE WARNING]" in caplog.text
    assert "VTI" in caplog.text
    assert "approaching" in caplog.text


def test_no_stale_warning_when_age_under_50pct_ttl(caplog):
    cache, r = _make_cache(ttls={"analysis": 86400})
    # Entry cached 10 hours ago — 42% of 86400s
    r.get.return_value = _envelope("data", time.time() - 36000)
    with caplog.at_level(logging.WARNING, logger="tradingagents.cache"):
        cache.get("NVDA", namespace="analysis", source="redis", ticker="NVDA")
    assert "[CACHE STALE WARNING]" not in caplog.text


def test_no_stale_warning_well_under_50pct(caplog):
    """Entry at 40% of TTL should NOT trigger stale warning."""
    cache, r = _make_cache(ttls={"market": 3600})
    r.get.return_value = _envelope("data", time.time() - 1440)  # 40% of 3600
    with caplog.at_level(logging.WARNING, logger="tradingagents.cache"):
        cache.get("SPY", namespace="market", source="redis", ticker="SPY")
    assert "[CACHE STALE WARNING]" not in caplog.text


def test_stale_warning_short_ttl_hours(caplog):
    """Short TTL (1h) — entry 40min old (67%) should warn."""
    cache, r = _make_cache(ttls={"news": 3600})
    r.get.return_value = _envelope("data", time.time() - 2400)
    with caplog.at_level(logging.WARNING, logger="tradingagents.cache"):
        cache.get("headlines", namespace="news", source="redis")
    assert "[CACHE STALE WARNING]" in caplog.text


# ---------------------------------------------------------------------------
# Legacy entries (no envelope) — no staleness info
# ---------------------------------------------------------------------------

def test_legacy_entry_no_stale_warning(caplog):
    """Legacy entries without _cached_at should log a basic hit, no warning."""
    cache, r = _make_cache(ttls={"market": 3600})
    r.get.return_value = json.dumps({"raw": True})
    with caplog.at_level(logging.INFO, logger="tradingagents.cache"):
        cache.get("OLD", namespace="market", source="redis")
    assert "[CACHE HIT]" in caplog.text
    assert "[CACHE STALE WARNING]" not in caplog.text
