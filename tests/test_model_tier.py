"""Verify model tiering: correct tier assignment and model resolution per tier.

Tests assign_tier(), resolve_models(), threshold config, and per-ticker overrides.
"""

import unittest

from tradingagents.model_tier import (
    DEFAULT_THRESHOLDS,
    TIER_MODEL_KEYS,
    TIERS,
    assign_tier,
    resolve_models,
)


class TestAssignTier(unittest.TestCase):
    """assign_tier() returns correct tier based on position value and P&L."""

    def test_large_value_returns_deep(self):
        self.assertEqual(assign_tier("NVDA", position_value=25_000), "deep")

    def test_high_pnl_returns_deep(self):
        self.assertEqual(assign_tier("NVDA", pnl_pct=-18), "deep")

    def test_mid_value_returns_standard(self):
        self.assertEqual(assign_tier("CRM", position_value=8_000), "standard")

    def test_mid_pnl_returns_standard(self):
        self.assertEqual(assign_tier("CRM", pnl_pct=7), "standard")

    def test_small_value_returns_light(self):
        self.assertEqual(assign_tier("TINY", position_value=1_000), "light")

    def test_no_data_returns_light(self):
        self.assertEqual(assign_tier("UNKNOWN"), "light")

    def test_zero_value_returns_light(self):
        self.assertEqual(assign_tier("VTI", position_value=0, pnl_pct=0), "light")

    def test_negative_pnl_uses_absolute(self):
        # abs(-18) = 18 >= 15 → deep
        self.assertEqual(assign_tier("X", pnl_pct=-18), "deep")

    def test_per_ticker_override(self):
        cfg = {"model_tier_overrides": {"NVDA": "light"}}
        self.assertEqual(assign_tier("NVDA", position_value=100_000, config=cfg), "light")

    def test_override_invalid_tier_ignored(self):
        cfg = {"model_tier_overrides": {"NVDA": "ultra"}}
        # "ultra" not in TIERS → falls through to threshold logic
        self.assertEqual(assign_tier("NVDA", position_value=25_000, config=cfg), "deep")

    def test_custom_thresholds(self):
        cfg = {"model_tier_thresholds": {
            "deep": {"min_value": 100_000, "min_abs_pnl_pct": 50},
            "standard": {"min_value": 50_000, "min_abs_pnl_pct": 25},
        }}
        # 25k is below custom deep (100k) and standard (50k) → light
        self.assertEqual(assign_tier("CRM", position_value=25_000, config=cfg), "light")


class TestResolveModels(unittest.TestCase):
    """resolve_models() maps tier to correct LLM config keys."""

    def _cfg(self):
        return {"deep_think_llm": "gpt-5.4", "quick_think_llm": "gpt-5.4-mini"}

    def test_deep_uses_deep_model(self):
        result = resolve_models("deep", self._cfg())
        self.assertEqual(result["deep_think_llm"], "gpt-5.4")
        self.assertEqual(result["quick_think_llm"], "gpt-5.4-mini")

    def test_standard_uses_deep_model(self):
        result = resolve_models("standard", self._cfg())
        self.assertEqual(result["deep_think_llm"], "gpt-5.4")

    def test_light_uses_quick_model_for_both(self):
        result = resolve_models("light", self._cfg())
        self.assertEqual(result["deep_think_llm"], "gpt-5.4-mini")
        self.assertEqual(result["quick_think_llm"], "gpt-5.4-mini")


class TestConstants(unittest.TestCase):
    """Verify module-level constants."""

    def test_all_three_tiers_defined(self):
        self.assertEqual(set(TIERS), {"deep", "standard", "light"})

    def test_default_thresholds_have_deep_and_standard(self):
        self.assertIn("deep", DEFAULT_THRESHOLDS)
        self.assertIn("standard", DEFAULT_THRESHOLDS)

    def test_tier_model_keys_cover_all_tiers(self):
        for tier in TIERS:
            self.assertIn(tier, TIER_MODEL_KEYS)


if __name__ == "__main__":
    unittest.main()
