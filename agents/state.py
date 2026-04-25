from typing import TypedDict, Optional


class PipelineState(TypedDict):
    """
    The single shared state object passed through the entire pipeline.
    
    Initialised in pipeline.py with run metadata only.
    Each agent adds its own output fields and leaves everything else untouched.
    Fields are never deleted — only added to.
    """

    # ── Run metadata (set at initialisation, never changed) ──────
    user_name:   str
    user_path:   str        # absolute path to this user's folder
    run_date:    str        # ISO format: "2025-04-21"
    tickers:     list[str]  # from portfolio.csv

    # ── Agent 1 outputs ──────────────────────────────────────────
    raw_text:        Optional[dict]  # {ticker: raw scraped text}
    summaries:       Optional[dict]  # {ticker: summary paragraph}
    aaii_sentiment:  Optional[dict]  # {bullish, bearish, neutral}
    fear_greed:      Optional[dict]  # {score, rating}
    earnings_dates:  Optional[dict]  # {ticker: next_earnings_date}
    earnings_data:   Optional[dict]  # {ticker: {recent_quarters, avg_surprise, consecutive_beats}}

    # ── Agent 2 outputs ──────────────────────────────────────────
    factor_results:    Optional[dict]  # {ticker: {betas, alpha, R2}}
    garch_results:     Optional[dict]  # {ticker: {omega, alpha, beta, sigma}}
    valuation:         Optional[dict]  # {ticker: {fwd_pe, ttm_pe, peg, ev_ebitda}}
    quant_commentary:  Optional[str]

    # ── Agent 3 outputs ──────────────────────────────────────────
    mc_current:       Optional[dict]  # Monte Carlo stats for current weights
    mc_rebalanced:    Optional[dict]  # Monte Carlo stats for suggested weights
    risk_commentary:  Optional[str]

    # ── Agent 4 outputs ──────────────────────────────────────────
    bl_views:              Optional[dict]  # {ticker: {view_return, confidence, reasoning}}
    recommended_weights:   Optional[dict]  # {ticker: weight}
    recommendation_table:  Optional[list]  # list of dicts for report rendering
    advisory_commentary:   Optional[str]

    # ── Pipeline metadata ────────────────────────────────────────
    errors:       Optional[list]   # non-fatal errors accumulate here
    report_path:  Optional[str]
    email_sent:   Optional[bool]