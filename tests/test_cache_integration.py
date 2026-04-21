"""Integration tests for AnalysisCache against a real Redis instance.

Run with: pytest -m redis
Requires Redis on localhost:6379.
"""

import time
import uuid

import pytest

redis = pytest.importorskip("redis")

PREFIX = f"ta:test:{uuid.uuid4().hex[:8]}"


def _redis_available():
    try:
        r = redis.Redis()
        r.ping()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.redis,
    pytest.mark.skipif(not _redis_available(), reason="Redis not available"),
]


@pytest.fixture()
def cache():
    from tradingagents.cache import AnalysisCache

    c = AnalysisCache()
    yield c
    # cleanup test keys
    r = redis.Redis()
    for key in r.scan_iter(f"ta:*:{PREFIX}:*"):
        r.delete(key)


def _key(name):
    return f"{PREFIX}:{name}"


def test_set_and_get(cache):
    cache.set(_key("k1"), {"v": 1}, namespace="market")
    assert cache.get(_key("k1"), namespace="market") == {"v": 1}


def test_namespace_isolation(cache):
    cache.set(_key("iso"), "market_val", namespace="market")
    cache.set(_key("iso"), "news_val", namespace="news")
    assert cache.get(_key("iso"), namespace="market") == "market_val"
    assert cache.get(_key("iso"), namespace="news") == "news_val"


def test_ttl_expiry(cache):
    cache.set(_key("exp"), "temp", namespace="market", ttl=1)
    assert cache.get(_key("exp"), namespace="market") == "temp"
    time.sleep(1.1)
    assert cache.get(_key("exp"), namespace="market") is None


def test_stats(cache):
    cache.set(_key("s1"), "x", namespace="analysis")
    cache.get(_key("s1"), namespace="analysis")
    cache.get(_key("missing"), namespace="analysis")
    s = cache.stats()
    assert s["analysis"]["hits"] >= 1
    assert s["analysis"]["misses"] >= 1
