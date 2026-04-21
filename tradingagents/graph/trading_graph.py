# TradingAgents/graph/trading_graph.py

import logging
import os
import time
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news
)

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    # ------------------------------------------------------------------
    # Analysis-level cache: Redis exact-date → filesystem TTL scan → LLM
    # ------------------------------------------------------------------

    @property
    def _analyses_dir(self) -> Path:
        return Path(self.config["results_dir"]) / "analyses"

    def _get_analysis_cache(self):
        """Return AnalysisCache instance or None."""
        url = self.config.get("cache_url", "")
        if not url:
            return None
        try:
            from tradingagents.cache import AnalysisCache
            if not hasattr(self, "_analysis_cache"):
                self._analysis_cache = AnalysisCache(url=url, config=self.config)
            return self._analysis_cache if self._analysis_cache.available else None
        except Exception:
            return None

    def _get_current_price(self, ticker: str) -> float | None:
        """Fetch latest closing price via yfinance (best-effort)."""
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return None

    def _check_analysis_cache(self, ticker: str, trade_date: str) -> Dict[str, Any] | None:
        """Check Redis (exact date) then filesystem TTL scan for a reusable analysis.

        Tier-aware: each tier reads only its own key. Platinum (TTL=0) always
        skips cache reads.

        Returns the cached summary dict or None.
        """
        from tradingagents.cache import TIER_TTL_ZERO

        tier = self.config.get("cache_tier")
        # Platinum always skips cache reads
        if tier == TIER_TTL_ZERO:
            logger.info("[ANALYSIS CACHE] %s %s: platinum tier — skipping cache read", ticker, trade_date)
            return None

        analysis_ttl = self.config.get("cache_ttl_overrides", {}).get(
            "analysis", self.config.get("cache_ttl_seconds", 86400)
        )

        # 1. Redis exact-date check (tier-suffixed key when tier is set)
        cache = self._get_analysis_cache()
        if cache:
            redis_key = f"{ticker}:{trade_date}:{tier}" if tier else f"{ticker}:{trade_date}"
            hit = cache.get(redis_key, namespace="analysis", ticker=ticker)
            if hit is not None:
                logger.info("[ANALYSIS CACHE] %s %s: Redis exact-date hit (tier=%s)", ticker, trade_date, tier or "none")
                return hit

        # 2. Filesystem TTL scan (filtered by tier when set)
        from tradingagents.cache import _find_recent_analysis
        summary = _find_recent_analysis(ticker, trade_date, analysis_ttl, self._analyses_dir, tier=tier)
        if summary is not None:
            logger.info("[ANALYSIS CACHE] %s %s: filesystem TTL scan hit (tier=%s)", ticker, trade_date, tier or "none")
        return summary

    def _price_delta_pct(self, cached_summary: Dict[str, Any], ticker: str) -> float | None:
        """Compute price change % between cached analysis and now."""
        cached_price = cached_summary.get("_cached_price")
        if cached_price is None or cached_price == 0:
            return None
        current = self._get_current_price(ticker)
        if current is None:
            return None
        return (current - cached_price) / cached_price * 100

    def _confirm_thesis(self, cached_summary: Dict[str, Any], ticker: str, delta_pct: float) -> bool:
        """Quick LLM check: does the cached thesis still hold given the price move?"""
        decision = cached_summary.get("final_trade_decision", "")
        if not decision:
            return False
        prompt = (
            f"A previous analysis of {ticker} concluded:\n\n{decision[:2000]}\n\n"
            f"Since then the price has moved {delta_pct:+.1f}%. "
            "Does this thesis still hold? Answer YES or NO only."
        )
        try:
            resp = self.quick_thinking_llm.invoke([("human", prompt)]).content.strip().upper()
            confirmed = resp.startswith("YES")
            logger.info("[THESIS CHECK] %s: %s (delta %.1f%%)", ticker, "confirmed" if confirmed else "invalidated", delta_pct)
            return confirmed
        except Exception:
            return False

    def _save_analysis_summary(self, ticker: str, trade_date: str, final_state: Dict[str, Any]) -> None:
        """Write summary.json for future TTL scans and cache in Redis.

        Tier-aware: filesystem path includes ``_{tier}`` suffix when a tier is
        configured.  Redis uses ``set_tiered`` to populate the current tier key
        and all higher-tier keys.
        """
        tier = self.config.get("cache_tier")

        summary = {
            "company_of_interest": ticker,
            "trade_date": trade_date,
            "final_trade_decision": final_state.get("final_trade_decision", ""),
            "investment_plan": final_state.get("investment_plan", ""),
            "_cached_at": time.time(),
        }
        if tier:
            summary["_cache_tier"] = tier
        # Add current price for future delta checks
        price = self._get_current_price(ticker)
        if price is not None:
            summary["_cached_price"] = price

        # Filesystem — include tier suffix when configured
        dir_name = f"{ticker}_{trade_date}_{tier}" if tier else f"{ticker}_{trade_date}"
        out_dir = self._analyses_dir / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Redis — tiered write populates current + higher tier keys
        cache = self._get_analysis_cache()
        if cache:
            base_key = f"{ticker}:{trade_date}"
            if tier:
                cache.set_tiered(base_key, summary, tier, namespace="analysis", price=price)
            else:
                cache.set(base_key, summary, namespace="analysis", price=price)

    # ------------------------------------------------------------------

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        Returns (raw_return, alpha_return, actual_holding_days) or
        (None, None, None) if price data is unavailable (too recent, delisted,
        or network error).
        """
        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
            spy = yf.Ticker("SPY").history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(spy) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(spy) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            spy_ret = float(
                (spy["Close"].iloc[actual_days] - spy["Close"].iloc[0])
                / spy["Close"].iloc[0]
            )
            alpha = raw - spy_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s (will retry next run): %s",
                ticker, trade_date, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(ticker, entry["date"])
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date.

        When ``checkpoint_enabled`` is set in config, the graph is recompiled
        with a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name
        reanalyze_pct = self.config.get("reanalyze_pct", 5.0)

        # --- Analysis-level cache check (before LLM calls) ---
        cached = self._check_analysis_cache(company_name, str(trade_date))
        if cached is not None:
            delta = self._price_delta_pct(cached, company_name)
            skip = False
            if delta is not None and abs(delta) < reanalyze_pct:
                if self._confirm_thesis(cached, company_name, delta):
                    skip = True
                    logger.info(
                        "%s: %+.1f%% < %.0f%% threshold, reusing %s analysis",
                        company_name, delta, reanalyze_pct, cached.get("trade_date", "?"),
                    )
            elif delta is None:
                skip = True
                logger.info("%s: no price delta available, reusing cached analysis", company_name)

            if skip:
                decision = cached.get("final_trade_decision", "")
                final_state = self.propagator.create_initial_state(company_name, str(trade_date))
                final_state["final_trade_decision"] = decision
                final_state["investment_plan"] = cached.get("investment_plan", "")
                final_state["trader_investment_plan"] = ""
                self.curr_state = final_state
                return final_state, self.process_signal(decision)

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name)

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def _run_graph(self, company_name, trade_date):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Initialize state — inject memory log context for PM.
        past_context = self.memory_log.get_past_context(company_name)
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, past_context=past_context
        )
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date resumes, different date starts fresh.
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)
            final_state = trace[-1]
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection.
        self.curr_state = final_state

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )

        # Save analysis summary for future TTL reuse
        self._save_analysis_summary(company_name, str(trade_date), final_state)

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
