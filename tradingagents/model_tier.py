"""Model tiering for cost-optimized LLM allocation.

Assigns tickers to one of three model tiers (deep, standard, light) based on
position value and P&L magnitude. Higher-value or higher-volatility positions
get more capable (and expensive) models.

Tier thresholds are configurable via DEFAULT_CONFIG keys:
    model_tier_thresholds:
        deep:     {"min_value": 20000, "min_abs_pnl_pct": 15}
        standard: {"min_value": 5000,  "min_abs_pnl_pct": 5}
        light:    {}  # everything else
"""

from __future__ import annotations

from typing import Any

# Default thresholds — override via config["model_tier_thresholds"]
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "deep": {"min_value": 20_000, "min_abs_pnl_pct": 15},
    "standard": {"min_value": 5_000, "min_abs_pnl_pct": 5},
}

TIERS = ("deep", "standard", "light")

# Maps tier → config key for the LLM model name
TIER_MODEL_KEYS: dict[str, str] = {
    "deep": "deep_think_llm",
    "standard": "deep_think_llm",
    "light": "quick_think_llm",
}


def assign_tier(
    ticker: str,
    *,
    position_value: float | None = None,
    pnl_pct: float | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Return the model tier for *ticker*.

    Parameters
    ----------
    ticker:
        Equity symbol (used for per-ticker overrides).
    position_value:
        Current market value of the position in USD.
    pnl_pct:
        Unrealised P&L percentage (can be negative).
    config:
        Full config dict. Checked for ``model_tier_overrides`` (per-ticker)
        and ``model_tier_thresholds`` (global thresholds).

    Returns
    -------
    One of ``"deep"``, ``"standard"``, or ``"light"``.
    """
    config = config or {}

    # Per-ticker override takes absolute precedence
    overrides: dict[str, str] = config.get("model_tier_overrides", {})
    if ticker in overrides and overrides[ticker] in TIERS:
        return overrides[ticker]

    thresholds = config.get("model_tier_thresholds", DEFAULT_THRESHOLDS)
    abs_pnl = abs(pnl_pct) if pnl_pct is not None else 0

    for tier in ("deep", "standard"):
        t = thresholds.get(tier, {})
        min_val = t.get("min_value", 0)
        min_pnl = t.get("min_abs_pnl_pct", 0)
        if (position_value is not None and position_value >= min_val) or abs_pnl >= min_pnl:
            return tier

    return "light"


def resolve_models(tier: str, config: dict[str, Any]) -> dict[str, str]:
    """Return ``{"deep_think_llm": ..., "quick_think_llm": ...}`` for *tier*.

    ``deep`` and ``standard`` tiers use ``deep_think_llm`` for both keys.
    ``light`` tier uses ``quick_think_llm`` for both keys.
    """
    key = TIER_MODEL_KEYS.get(tier, "quick_think_llm")
    model = config.get(key, config.get("quick_think_llm", ""))
    if tier == "light":
        return {"deep_think_llm": model, "quick_think_llm": model}
    return {
        "deep_think_llm": config.get("deep_think_llm", model),
        "quick_think_llm": config.get("quick_think_llm", model),
    }
