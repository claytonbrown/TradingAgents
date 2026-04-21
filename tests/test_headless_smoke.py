"""Smoke test: single ticker end-to-end via run_headless.main() (pytest -m smoke)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run_headless import main


@pytest.mark.smoke
def test_single_ticker_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Full pipeline: parse args → analyse → JSON output + file output."""
    fake_state = {
        "final_trade_decision": "BUY — strong momentum and earnings beat",
        "market_report": "Market is bullish",
        "sentiment_report": "Positive sentiment",
        "news_report": "Earnings beat expectations",
        "fundamentals_report": "Strong balance sheet",
        "cost_usd": 0.05,
        "tokens_in": 1200,
        "tokens_out": 300,
    }

    mock_graph = MagicMock()
    mock_graph.return_value.propagate.return_value = (fake_state, "BUY")

    with (
        patch("run_headless.HeadlessRunner._fetch_price", return_value=198.50),
        patch("tradingagents.graph.trading_graph.TradingAgentsGraph", mock_graph),
    ):
        exit_code = main([
            "--ticker", "NVDA",
            "--date", "2026-04-21",
            "--depth", "shallow",
            "--json",
            "--output-dir", str(tmp_path),
        ])

    assert exit_code == 0

    # Verify JSON stdout
    captured = capsys.readouterr()
    results = json.loads(captured.out)
    assert isinstance(results, list)
    assert len(results) == 1

    r = results[0]
    assert r["ticker"] == "NVDA"
    assert r["date"] == "2026-04-21"
    assert r["decision"] == "BUY"
    assert r["current_price"] == 198.50
    assert r["thesis"] == fake_state["final_trade_decision"]
    assert r["reports"]["market"] == "Market is bullish"
    assert r["metadata"]["debate_rounds"] == 1  # shallow
    assert r["metadata"]["cost_usd"] == 0.05

    # Verify output files
    assert (tmp_path / "NVDA.json").exists()
    assert (tmp_path / "NVDA.md").exists()

    file_result = json.loads((tmp_path / "NVDA.json").read_text())
    assert file_result["ticker"] == "NVDA"

    md = (tmp_path / "NVDA.md").read_text()
    assert "# NVDA" in md
    assert "BUY" in md
