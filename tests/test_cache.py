"""Unit tests for tradingagents/cache.py AnalysisCache class."""

import json
from unittest.mock import MagicMock, patch


@patch("redis.Redis")
@patch("redis.ConnectionPool.from_url")
def _make_cache(mock_pool, mock_redis_cls, redis_instance=None):
    """Helper to create AnalysisCache with mocked redis."""
    instance = redis_instance or MagicMock()
    mock_redis_cls.return_value = instance
    from tradingagents.cache import AnalysisCache
    cache = AnalysisCache()
    return cache, instance


def test_set_calls_redis_with_correct_args():
    cache, r = _make_cache()
    data = {"price": 100}
    cache.set("NVDA", data, namespace="market")
    r.set.assert_called_once_with("ta:market:NVDA", json.dumps(data, default=str), ex=3600)


def test_get_returns_deserialized_json():
    cache, r = _make_cache()
    data = {"score": 0.8}
    r.get.return_value = json.dumps(data)
    result = cache.get("sent1", namespace="analysis")
    r.get.assert_called_once_with("ta:analysis:sent1")
    assert result == data


def test_get_returns_none_on_miss():
    cache, r = _make_cache()
    r.get.return_value = None
    assert cache.get("missing", namespace="market") is None


def test_key_format():
    cache, _ = _make_cache()
    assert cache._key("news", "headline1") == "ta:news:headline1"


def test_default_ttls():
    cache, r = _make_cache()
    expected = {"market": 3600, "analysis": 86400, "news": 14400, "fundamentals": 43200, "indicators": 14400}
    for ns, ttl in expected.items():
        r.set.reset_mock()
        cache.set("k", "v", namespace=ns)
        r.set.assert_called_once_with(f"ta:{ns}:k", json.dumps("v", default=str), ex=ttl)


def test_custom_ttl_override():
    cache, r = _make_cache()
    cache.set("k", "v", namespace="market", ttl=999)
    r.set.assert_called_once_with("ta:market:k", json.dumps("v", default=str), ex=999)


@patch("redis.Redis")
@patch("redis.ConnectionPool.from_url")
def test_fallback_when_ping_fails(mock_pool, mock_redis_cls):
    instance = MagicMock()
    instance.ping.side_effect = ConnectionError("refused")
    mock_redis_cls.return_value = instance
    from tradingagents.cache import AnalysisCache
    cache = AnalysisCache()
    assert cache.available is False
    assert cache.get("x") is None
    assert cache.set("x", "y") is False


def test_stats_tracks_hits_and_misses():
    cache, r = _make_cache()
    r.get.return_value = json.dumps("data")
    cache.get("a", namespace="market")
    cache.get("b", namespace="market")
    r.get.return_value = None
    cache.get("c", namespace="market")
    cache.get("d", namespace="news")
    s = cache.stats()
    assert s["market"] == {"hits": 2, "misses": 1}
    assert s["news"] == {"hits": 0, "misses": 1}
