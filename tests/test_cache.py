"""Unit tests for tradingagents/cache.py — Task 20.

Covers: mock Redis get/set/TTL, namespace keys, graceful fallback,
stats (hits/misses/skips), and --no-cache bypass.
"""

from __future__ import annotations

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(redis_instance: MagicMock | None = None, **kwargs):
    """Create AnalysisCache with mocked Redis."""
    instance = redis_instance or MagicMock()
    with patch("redis.ConnectionPool.from_url"), \
         patch("redis.Redis", return_value=instance):
        from tradingagents.cache import AnalysisCache
        cache = AnalysisCache(**kwargs)
    return cache, instance


def _envelope(data, *, price: float | None = None) -> dict:
    """Build an envelope dict matching what set() writes."""
    env: dict = {"_cached_at": time.time(), "_data": data}
    if price is not None:
        env["_cached_price"] = price
    return env


# ---------------------------------------------------------------------------
# Key format
# ---------------------------------------------------------------------------

def test_key_format():
    cache, _ = _make_cache()
    assert cache._key("news", "headline1") == "ta:news:headline1"
    assert cache._key("market", "NVDA") == "ta:market:NVDA"


# ---------------------------------------------------------------------------
# set() — envelope wrapping, TTL
# ---------------------------------------------------------------------------

def test_set_writes_envelope_with_ttl():
    cache, r = _make_cache()
    cache.set("NVDA", {"price": 100}, namespace="market")
    r.set.assert_called_once()
    call_args = r.set.call_args
    assert call_args[0][0] == "ta:market:NVDA"
    stored = json.loads(call_args[0][1])
    assert stored["_data"] == {"price": 100}
    assert "_cached_at" in stored
    assert call_args[1]["ex"] == 3600  # market default TTL


def test_set_stores_price_in_envelope():
    cache, r = _make_cache()
    cache.set("NVDA", "d", namespace="market", price=150.0)
    stored = json.loads(r.set.call_args[0][1])
    assert stored["_cached_price"] == 150.0


def test_set_custom_ttl_override():
    cache, r = _make_cache()
    cache.set("k", "v", namespace="market", ttl=999)
    assert r.set.call_args[1]["ex"] == 999


def test_set_returns_false_when_unavailable():
    cache, _ = _make_cache()
    cache._redis = None
    assert cache.set("k", "v") is False


# ---------------------------------------------------------------------------
# Default TTLs per namespace
# ---------------------------------------------------------------------------

def test_default_ttls():
    from tradingagents.cache import DEFAULT_TTLS
    cache, r = _make_cache()
    for ns, expected_ttl in DEFAULT_TTLS.items():
        r.set.reset_mock()
        cache.set("k", "v", namespace=ns)
        assert r.set.call_args[1]["ex"] == expected_ttl, f"{ns} TTL mismatch"


def test_config_overrides_default_ttls():
    cfg = {"cache_ttl_seconds": 100, "cache_ttl_overrides": {"market": 42}}
    cache, r = _make_cache(config=cfg)
    # market uses per-namespace override
    cache.set("k", "v", namespace="market")
    assert r.set.call_args[1]["ex"] == 42
    # analysis uses the platform default (100) since no per-namespace override
    r.set.reset_mock()
    cache.set("k", "v", namespace="analysis")
    assert r.set.call_args[1]["ex"] == 100


# ---------------------------------------------------------------------------
# get() — deserialization, envelope unwrapping
# ---------------------------------------------------------------------------

def test_get_unwraps_envelope():
    cache, r = _make_cache()
    r.get.return_value = json.dumps(_envelope({"score": 0.8}))
    result = cache.get("sent1", namespace="analysis")
    assert result == {"score": 0.8}


def test_get_handles_legacy_non_envelope():
    """Legacy entries without _cached_at envelope are returned as-is."""
    cache, r = _make_cache()
    r.get.return_value = json.dumps({"raw": True})
    assert cache.get("old", namespace="market") == {"raw": True}


def test_get_returns_none_on_miss():
    cache, r = _make_cache()
    r.get.return_value = None
    assert cache.get("missing", namespace="market") is None


def test_get_returns_none_when_unavailable():
    cache, _ = _make_cache()
    cache._redis = None
    assert cache.get("x") is None


# ---------------------------------------------------------------------------
# Graceful fallback when Redis unavailable
# ---------------------------------------------------------------------------

def test_fallback_when_ping_fails():
    instance = MagicMock()
    instance.ping.side_effect = ConnectionError("refused")
    cache, _ = _make_cache(redis_instance=instance)
    assert cache.available is False
    assert cache.get("x") is None
    assert cache.set("x", "y") is False


def test_available_true_when_connected():
    cache, _ = _make_cache()
    assert cache.available is True


# ---------------------------------------------------------------------------
# Stats: hits, misses, skips
# ---------------------------------------------------------------------------

def test_stats_tracks_hits_and_misses():
    cache, r = _make_cache()
    r.get.return_value = json.dumps(_envelope("data"))
    cache.get("a", namespace="market")
    cache.get("b", namespace="market")
    r.get.return_value = None
    cache.get("c", namespace="market")
    s = cache.stats()
    assert s["market"]["hits"] == 2
    assert s["market"]["misses"] == 1
    assert s["market"]["skips"] == 0


def test_stats_skips_when_unavailable():
    cache, _ = _make_cache()
    cache._redis = None
    cache.get("x", namespace="news")
    cache.get("y", namespace="news")
    s = cache.stats()
    assert s["news"]["skips"] == 2
    assert s["news"]["hits"] == 0


def test_skip_method_increments():
    cache, _ = _make_cache()
    cache.skip("market")
    cache.skip("market")
    assert cache.stats()["market"]["skips"] == 2


def test_log_stats(caplog):
    cache, r = _make_cache()
    r.get.return_value = json.dumps(_envelope("d"))
    cache.get("a", namespace="market")
    r.get.return_value = None
    cache.get("b", namespace="news")
    with caplog.at_level(logging.INFO, logger="tradingagents.cache"):
        cache.log_stats()
    assert "hits: 1" in caplog.text
    assert "misses: 1" in caplog.text


# ---------------------------------------------------------------------------
# --no-cache bypass
# ---------------------------------------------------------------------------

def test_no_cache_skips_reads_but_allows_writes():
    """Simulates --no-cache: get() returns None (skip), set() still works."""
    cache, r = _make_cache()
    # Simulate no_cache by making the caller skip reads (as dataflows/interface.py does)
    # The cache itself doesn't enforce no_cache — the caller does.
    # But when cache is available and no_cache is set, callers call skip() instead of get().
    cache.skip("market")
    assert cache.stats()["market"]["skips"] == 1
    # Writes still succeed
    assert cache.set("k", "v", namespace="market") is True
    r.set.assert_called_once()


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------

def test_namespace_isolation():
    cache, r = _make_cache()
    cache.set("k", "market_val", namespace="market")
    cache.set("k", "news_val", namespace="news")
    assert r.set.call_count == 2
    keys_written = [call[0][0] for call in r.set.call_args_list]
    assert "ta:market:k" in keys_written
    assert "ta:news:k" in keys_written
