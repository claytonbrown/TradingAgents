"""Fetch indicator values using the existing stockstats pipeline.

Thin wrapper around ``StockstatsUtils.get_stock_stats`` — no duplicate
computation.  If any indicator fetch fails the key is simply omitted.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Indicators to pre-fetch, keyed by the name interpret.py expects.
# Values are the stockstats indicator names (same ones the LLM would request).
_INDICATORS = {
    "rsi": {"stats": ["rsi_14"], "build": "_build_rsi"},
    "macd": {"stats": ["macd", "macds", "macdh"], "build": "_build_macd"},
    "bollinger": {"stats": ["boll_ub", "boll_lb", "boll"], "build": "_build_bollinger"},
    "sma_crossover": {"stats": ["close_50_sma", "close_200_sma"], "build": "_build_sma"},
}


def fetch_indicators(symbol: str, curr_date: str) -> dict[str, dict[str, Any]]:
    """Fetch indicator values via stockstats for *symbol* as of *curr_date*.

    Returns a dict keyed by indicator group name with sub-dicts matching
    what ``interpret.py`` expects.  Missing/failed indicators are omitted.
    """
    try:
        from tradingagents.dataflows.stockstats_utils import StockstatsUtils
    except ImportError:
        logger.warning("indicators: stockstats_utils not available")
        return {}

    results: dict[str, dict[str, Any]] = {}

    for group, spec in _INDICATORS.items():
        vals: dict[str, Any] = {}
        try:
            for stat_name in spec["stats"]:
                v = StockstatsUtils.get_stock_stats(symbol, stat_name, curr_date)
                if v == "N/A: Not a trading day (weekend or holiday)":
                    v = None
                vals[stat_name] = v
        except Exception as exc:
            logger.warning("indicators: failed fetching %s for %s: %s", group, symbol, exc)
            continue

        builder = globals().get(spec["build"])
        if builder and any(v is not None for v in vals.values()):
            built = builder(vals)
            if built:
                results[group] = built

    return results


def _build_rsi(vals: dict) -> dict[str, Any] | None:
    v = vals.get("rsi_14")
    if v is None:
        return None
    return {"value": float(v), "period": 14}


def _build_macd(vals: dict) -> dict[str, Any] | None:
    macd_val = vals.get("macd")
    sig_val = vals.get("macds")
    hist_val = vals.get("macdh")
    if macd_val is None:
        return None
    return {
        "value": _float(macd_val),
        "signal": _float(sig_val),
        "histogram": _float(hist_val),
    }


def _build_bollinger(vals: dict) -> dict[str, Any] | None:
    upper = vals.get("boll_ub")
    lower = vals.get("boll_lb")
    mid = vals.get("boll")
    if upper is None or lower is None:
        return None
    return {
        "value": _float(mid),
        "upper": _float(upper),
        "lower": _float(lower),
    }


def _build_sma(vals: dict) -> dict[str, Any] | None:
    sma50 = vals.get("close_50_sma")
    sma200 = vals.get("close_200_sma")
    if sma50 is None or sma200 is None:
        return None
    return {
        "sma50": _float(sma50),
        "sma200": _float(sma200),
        "crossover": None,  # stockstats doesn't provide this directly
    }


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
