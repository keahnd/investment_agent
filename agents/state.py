from typing import TypedDict, Optional
import pandas as pd


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
    current_weights:   Optional[dict]  # {ticker: weight} computed from portfolio.csv
    cad_usd_rate:      Optional[float] # Current USD-CAD exchange rate
    total_portfolio_value: Optional[float] # Current portfolio value
    strategies:        Optional[list] # Rebalancing strategies
    

    # ── Agent 1 outputs ──────────────────────────────────────────
    raw_text:        Optional[dict]  # {ticker: raw scraped text}
    summaries:       Optional[dict]  # {ticker: summary paragraph}
    aaii_sentiment:  Optional[dict]  # {bullish, bearish, neutral}
    fear_greed:      Optional[dict]  # {score, rating}

    # ── Agent 2 outputs ──────────────────────────────────────────
    factor_results:    Optional[dict]  # {ticker: {alpha, b_mkt, b_smb, b_hml, r2, r2_adj, p_vals}}
    garch_results:     Optional[dict]  # {ticker: {omega, alpha, beta, persist, long_run_vol, sigma_current}}
    mu_sigma:          Optional[dict]  # {ticker: {mu_annual, sigma_annual}}
    valuation:         Optional[dict]  # {ticker: {fwd_pe, ttm_pe, peg, ev_ebitda}}
    financial_health:  Optional[dict]  # {ticker: {revenue_growth, fcf_margin, roic, ...}}
    earnings_data:     Optional[dict]  # {ticker: {recent_quarters, avg_surprise, consecutive_beats}}
    earnings_dates:    Optional[dict]  # {ticker: next_earnings_date}
    covariance_matrix: Optional[dict] # Covariance Matrix for the universe of stocks
    quant_commentary:  Optional[str]   # LLM anomaly flags and interpretation

    # ── Agent 3 outputs ──────────────────────────────────────────
    mc_current:       Optional[dict]  # Monte Carlo stats per-ticker
    mc_port_current:  Optional[dict]  # Monte Carlo stats for current weights
    mc_port_rebalanced:    Optional[dict]  # Monte Carlo stats for suggested weights
    sim_commentary:  Optional[str]

    # ── Agent 4 outputs ──────────────────────────────────────────
    bl_views:              Optional[dict]  # {ticker: {view_return, confidence, reasoning}}
    recommended_weights:   Optional[dict]  # {ticker: weight}
    recommendation_table:  Optional[list]  # list of dicts for report rendering
    advisory_commentary:   Optional[str]

    # ── Pipeline metadata ────────────────────────────────────────
    errors:       Optional[list]   # non-fatal errors accumulate here
    report_path:  Optional[str]
    email_sent:   Optional[bool]