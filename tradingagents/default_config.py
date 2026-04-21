import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Redis cache (optional). Set URL to enable, empty string to disable.
    # Checks TRADINGAGENTS_CACHE_URL first, then REDIS_URL (standard hosting env var).
    "cache_url": os.getenv("TRADINGAGENTS_CACHE_URL", os.getenv("REDIS_URL", "")),
    # Default TTL for all cached entries (seconds). 86400 = 1 day.
    "cache_ttl_seconds": 86400,
    # Per-namespace TTL overrides (seconds). Namespaces not listed use cache_ttl_seconds.
    "cache_ttl_overrides": {
        "market_data": 3600,
        "analysis": 604800,
        "news": 43200,
        "fundamentals": 86400,
        "indicators": 14400,
    },
    # Per-ticker analysis TTL overrides (seconds). Takes precedence over the
    # "analysis" namespace TTL from cache_ttl_overrides. Useful for volatile
    # tickers (short TTL) vs stable ETFs (long TTL).
    # Example: {"NVDA": 86400, "VTI": 2592000}  # volatile=1d, stable=30d
    "analysis_ttl_overrides": {},
    # When True, bypass all cache reads (Redis + filesystem TTL). Cache writes
    # still happen so results are available for subsequent runs.
    "no_cache": False,
    # Price-delta threshold (%) for reusing cached analysis. If the ticker price
    # moved less than this since the last cached analysis, reuse it (with quick
    # LLM thesis confirmation). Set to 0 to always re-analyze.
    "reanalyze_pct": 5.0,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
