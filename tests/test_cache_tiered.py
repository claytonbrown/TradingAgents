"""Unit tests for tiered cache writes — Task 22.

Covers: Bronze write populates all 4 keys, Silver writes 3, Gold writes 2,
Platinum writes only platinum key. Each tier reads only its own key.
Platinum always skips reads (TTL=0).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.cache import TIERS, TIER_TTL_ZERO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(**kwargs):
    """Create AnalysisCache with mocked Redis."""
    instance = MagicMock()
    with patch("redis.ConnectionPool.from_url"), \
         patch("redis.Redis", return_value=instance):
        from tradingagents.cache import AnalysisCache
        cache = AnalysisCache(**kwargs)
    return cache, instance


def _written_keys(mock_redis: MagicMock) -> list[str]:
    """Extract all Redis keys written via set()."""
    return [call[0][0] for call in mock_redis.set.call_args_list]


# ---------------------------------------------------------------------------
# set_tiered: write visibility per tier
# ---------------------------------------------------------------------------

def test_bronze_writes_all_four_tiers():
    cache, r = _make_cache()
    cache.set_tiered("NVDA:2026-04-21", {"d": 1}, tier="bronze", namespace="analysis")
    keys = _written_keys(r)
    assert keys == [
        "ta:analysis:NVDA:2026-04-21:bronze",
        "ta:analysis:NVDA:2026-04-21:silver",
        "ta:analysis:NVDA:2026-04-21:gold",
        "ta:analysis:NVDA:2026-04-21:platinum",
    ]


def test_silver_writes_three_tiers():
    cache, r = _make_cache()
    cache.set_tiered("NVDA:2026-04-21", {"d": 1}, tier="silver", namespace="analysis")
    keys = _written_keys(r)
    assert keys == [
        "ta:analysis:NVDA:2026-04-21:silver",
        "ta:analysis:NVDA:2026-04-21:gold",
        "ta:analysis:NVDA:2026-04-21:platinum",
    ]


def test_gold_writes_two_tiers():
    cache, r = _make_cache()
    cache.set_tiered("NVDA:2026-04-21", {"d": 1}, tier="gold", namespace="analysis")
    keys = _written_keys(r)
    assert keys == [
        "ta:analysis:NVDA:2026-04-21:gold",
        "ta:analysis:NVDA:2026-04-21:platinum",
    ]


def test_platinum_writes_only_platinum():
    cache, r = _make_cache()
    cache.set_tiered("NVDA:2026-04-21", {"d": 1}, tier="platinum", namespace="analysis")
    keys = _written_keys(r)
    assert keys == ["ta:analysis:NVDA:2026-04-21:platinum"]


# ---------------------------------------------------------------------------
# Each tier reads only its own key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier", TIERS)
def test_tier_reads_own_key(tier):
    """get() with a tier-suffixed key reads only that tier's key."""
    cache, r = _make_cache()
    r.get.return_value = json.dumps({"_cached_at": 1, "_data": {"tier": tier}})
    result = cache.get(f"NVDA:2026-04-21:{tier}", namespace="analysis")
    r.get.assert_called_once_with(f"ta:analysis:NVDA:2026-04-21:{tier}")
    assert result == {"tier": tier}


# ---------------------------------------------------------------------------
# Platinum always skips reads (TTL=0)
# ---------------------------------------------------------------------------

def test_platinum_tier_constant():
    assert TIER_TTL_ZERO == "platinum"


def test_platinum_skip_reads_pattern():
    """Callers should skip cache reads for platinum tier (TTL=0).

    The cache itself doesn't enforce this — the caller checks TIER_TTL_ZERO
    and calls skip() instead of get(). We verify the skip() path works.
    """
    cache, r = _make_cache()
    tier = "platinum"
    # Simulate caller logic: if tier == TIER_TTL_ZERO, skip reads
    if tier == TIER_TTL_ZERO:
        cache.skip("analysis")
    else:
        cache.get(f"NVDA:2026-04-21:{tier}", namespace="analysis")
    r.get.assert_not_called()
    assert cache.stats()["analysis"]["skips"] == 1


# ---------------------------------------------------------------------------
# set_tiered: data consistency — all tier keys get same data
# ---------------------------------------------------------------------------

def test_all_tier_keys_contain_same_data():
    cache, r = _make_cache()
    data = {"decision": "buy", "confidence": 0.9}
    cache.set_tiered("NVDA:2026-04-21", data, tier="bronze", namespace="analysis")
    for call in r.set.call_args_list:
        stored = json.loads(call[0][1])
        assert stored["_data"] == data


# ---------------------------------------------------------------------------
# set_tiered: unknown tier falls back to plain set()
# ---------------------------------------------------------------------------

def test_unknown_tier_falls_back_to_plain_set():
    cache, r = _make_cache()
    cache.set_tiered("NVDA:2026-04-21", {"d": 1}, tier="unknown", namespace="analysis")
    keys = _written_keys(r)
    assert keys == ["ta:analysis:NVDA:2026-04-21"]


# ---------------------------------------------------------------------------
# set_tiered: price envelope propagated to all tiers
# ---------------------------------------------------------------------------

def test_price_propagated_to_all_tier_keys():
    cache, r = _make_cache()
    cache.set_tiered("NVDA:2026-04-21", {"d": 1}, tier="silver", namespace="analysis", price=850.0)
    for call in r.set.call_args_list:
        stored = json.loads(call[0][1])
        assert stored["_cached_price"] == 850.0
