"""Unit tests for _find_recent_analysis TTL scan — Task 21.

Covers: finds correct analysis within TTL, respects age, skips stale,
per-ticker TTL override, tier filtering, and --reanalyze-pct threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.cache import _find_recent_analysis


@pytest.fixture()
def analyses_dir(tmp_path: Path) -> Path:
    """Create a temp analyses directory with sample summary.json files."""
    return tmp_path


def _write_summary(base: Path, dirname: str, data: dict | None = None) -> None:
    d = base / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(data or {"ticker": dirname}))


# ------------------------------------------------------------------
# Basic: finds most recent within TTL
# ------------------------------------------------------------------

def test_finds_most_recent_within_ttl(analyses_dir):
    _write_summary(analyses_dir, "NVDA_2026-04-10", {"day": 10})
    _write_summary(analyses_dir, "NVDA_2026-04-14", {"day": 14})
    _write_summary(analyses_dir, "NVDA_2026-04-18", {"day": 18})

    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 30, analyses_dir=analyses_dir)
    assert result == {"day": 18}


def test_returns_none_when_no_analyses(analyses_dir):
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400, analyses_dir=analyses_dir)
    assert result is None


def test_returns_none_for_nonexistent_dir(tmp_path):
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400, analyses_dir=tmp_path / "nope")
    assert result is None


# ------------------------------------------------------------------
# Respects age: skips entries outside TTL window
# ------------------------------------------------------------------

def test_skips_stale_analysis(analyses_dir):
    _write_summary(analyses_dir, "NVDA_2026-03-01", {"old": True})
    # 51 days old, TTL is 30 days
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 30, analyses_dir=analyses_dir)
    assert result is None


def test_skips_future_analysis(analyses_dir):
    _write_summary(analyses_dir, "NVDA_2026-04-25", {"future": True})
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 30, analyses_dir=analyses_dir)
    assert result is None


def test_returns_fresh_skips_stale(analyses_dir):
    _write_summary(analyses_dir, "NVDA_2026-01-01", {"stale": True})  # 110 days old
    _write_summary(analyses_dir, "NVDA_2026-04-20", {"fresh": True})  # 1 day old
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 7, analyses_dir=analyses_dir)
    assert result == {"fresh": True}


# ------------------------------------------------------------------
# Per-ticker TTL override (caller passes different ttl_seconds)
# ------------------------------------------------------------------

def test_per_ticker_short_ttl_skips_old(analyses_dir):
    """Volatile ticker with 1-day TTL: 3-day-old analysis is stale."""
    _write_summary(analyses_dir, "NVDA_2026-04-18", {"day": 18})
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400, analyses_dir=analyses_dir)
    assert result is None


def test_per_ticker_long_ttl_keeps_old(analyses_dir):
    """Stable ETF with 30-day TTL: 3-day-old analysis is fresh."""
    _write_summary(analyses_dir, "VTI_2026-04-18", {"day": 18})
    result = _find_recent_analysis("VTI", "2026-04-21", ttl_seconds=86400 * 30, analyses_dir=analyses_dir)
    assert result == {"day": 18}


# ------------------------------------------------------------------
# Tier filtering
# ------------------------------------------------------------------

def test_tier_filter_matches_correct_tier(analyses_dir):
    _write_summary(analyses_dir, "NVDA_2026-04-20_bronze", {"tier": "bronze"})
    _write_summary(analyses_dir, "NVDA_2026-04-20_silver", {"tier": "silver"})

    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 7, analyses_dir=analyses_dir, tier="bronze")
    assert result == {"tier": "bronze"}


def test_tier_filter_skips_wrong_tier(analyses_dir):
    _write_summary(analyses_dir, "NVDA_2026-04-20_bronze", {"tier": "bronze"})
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 7, analyses_dir=analyses_dir, tier="gold")
    assert result is None


def test_no_tier_filter_returns_any(analyses_dir):
    _write_summary(analyses_dir, "NVDA_2026-04-19", {"tier": "none"})
    _write_summary(analyses_dir, "NVDA_2026-04-20_silver", {"tier": "silver"})
    # Without tier filter, returns most recent regardless of tier suffix
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 7, analyses_dir=analyses_dir)
    assert result == {"tier": "silver"}


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_ignores_malformed_directory_names(analyses_dir):
    _write_summary(analyses_dir, "NVDA_notadate", {"bad": True})
    _write_summary(analyses_dir, "NVDA_2026-04-20", {"good": True})
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 7, analyses_dir=analyses_dir)
    assert result == {"good": True}


def test_ignores_corrupt_json(analyses_dir):
    d = analyses_dir / "NVDA_2026-04-20"
    d.mkdir()
    (d / "summary.json").write_text("{corrupt")
    _write_summary(analyses_dir, "NVDA_2026-04-19", {"fallback": True})
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 7, analyses_dir=analyses_dir)
    assert result == {"fallback": True}


def test_exact_ttl_boundary_included(analyses_dir):
    """Analysis exactly at TTL boundary (age == ttl_seconds) should be included."""
    _write_summary(analyses_dir, "NVDA_2026-04-14", {"boundary": True})
    # 7 days = 604800 seconds
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=604800, analyses_dir=analyses_dir)
    assert result == {"boundary": True}


def test_different_tickers_isolated(analyses_dir):
    _write_summary(analyses_dir, "NVDA_2026-04-20", {"ticker": "NVDA"})
    _write_summary(analyses_dir, "AAPL_2026-04-20", {"ticker": "AAPL"})
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 7, analyses_dir=analyses_dir)
    assert result == {"ticker": "NVDA"}


# ------------------------------------------------------------------
# --reanalyze-pct threshold simulation
# ------------------------------------------------------------------
# The reanalyze_pct check happens at the caller level (trading_graph.py),
# but we verify the TTL scan returns data that the caller can then check
# against the price delta threshold.

def test_reuse_metadata_available_for_price_delta_check(analyses_dir):
    """TTL scan returns summary with cached price so caller can compute delta."""
    _write_summary(analyses_dir, "NVDA_2026-04-20", {"price": 850.0, "decision": "buy"})
    result = _find_recent_analysis("NVDA", "2026-04-21", ttl_seconds=86400 * 7, analyses_dir=analyses_dir)
    assert result is not None
    assert result["price"] == 850.0
