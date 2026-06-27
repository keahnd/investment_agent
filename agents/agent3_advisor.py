from pathlib import Path
import json
import pandas as pd
import numpy as np
import cvxpy as cp
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.objective_functions import L2_reg
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import HRPOpt, objective_functions

from agents.state import PipelineState
from agents.llm import get_llm

UNCERTAINTY_SCALE = 0.05   # from config — tune this over time


def load_constraints(user_path: Path) -> dict:
    """
    Loads the user constriants from the config file
    
    Args:
        user_path: Path to config file
        
    Rerturns:
        dict of contraints
    """
    config_path = user_path / "config.json"
    constr = {}
    if config_path.exists():
        with open(config_path) as f:
            constr = {
                k: v for k, v in json.load(f).items()
                if k in ("risk_aversion", "investment_horizon_years", "action_threshold",
                    "max_allocation_per_asset", "max_allocation_per_sector", "min_cash_buffer")
            }
    return constr


def confidence_to_omega(confidence, asset_variance, uncertainty_scale):
    """
    Converts a 1-5 confidence score to an omega uncertainty value.
    Confidence 5 = low uncertainty = small omega = view pulls posterior strongly
    Confidence 1 = high uncertainty = large omega = view barely moves posterior
    """
    # Invert confidence so high confidence = small uncertainty
    uncertainty_factor = (6 - confidence) / 5.0
    # Scale 1.0 at confidence 1 down to 0.2 at confidence 5
    return uncertainty_factor * uncertainty_scale * asset_variance


def build_sector_constraints(tickers, valuation, constraints):
    """
    Groups tickers by sector and returns sector mapper dict
    for PyPortfolioOpt's sector constraint.
    """
    max_sector = constraints["max_allocation_per_sector"]
    
    sector_mapper = {}
    for ticker in tickers:
        sector = valuation.get(ticker, {}).get("sector") or "Unknown"
        sector_mapper[ticker] = sector
    
    sectors = set(sector_mapper.values())
    # sector_upper is the same limit applied to every sector
    sector_upper = {sector: max_sector for sector in sectors}
    sector_lower = {sector: 0.0 for sector in sectors}
    
    return sector_mapper, sector_lower, sector_upper


def compute_value_recommendation(ticker: str, current_weight: float,
								bl_view: dict, valuation_result: dict,
								quality_metrics: dict) -> dict:
	"""
	Determines BUY / HOLD / SELL / TRIM based on combined signals.
	Properly weights quality businesses at historically cheap valuations.
	"""
	view_delta   = bl_view.get("view_return_delta", 0.0)
	val_signal   = valuation_result.get("valuation_signal", "fair")
	pe_pctile    = valuation_result.get("pe_percentile_vs_history")
	peg          = quality_metrics.get("peg")
	consec_beats = quality_metrics.get("consecutive_beats", 0)
	gross_margin = quality_metrics.get("gross_margin", 0)

	# ── Quality score (0–3) ──
	quality_score = 0
	if gross_margin > 0.50:   quality_score += 1
	if consec_beats >= 4:     quality_score += 1
	if peg is not None and peg < 1.5: quality_score += 1

	# ── Signal strength ──
	historically_cheap = (
		val_signal == "cheap" or
		(pe_pctile is not None and pe_pctile < 30)
	)
	historically_expensive = (
		val_signal == "expensive" and
		(pe_pctile is None or pe_pctile > 70)
	)

	# ── Decision matrix ──
	if historically_cheap and quality_score >= 2 and view_delta > 0:
		action = "BUY"
		note = (
			f"High-quality business at historically cheap valuation "
			f"(PE at {pe_pctile:.0f}th percentile of own history). "
			f"PEG {peg:.2f}, {consec_beats} consecutive earnings beats."
		)
	elif historically_cheap and quality_score >= 1:
		action = "BUY"
		note = "Historically cheap valuation with solid quality metrics."
	elif historically_expensive and view_delta < -0.02:
		action = "SELL"
		note = "Trading above historical PE range with negative forward view."
	elif historically_expensive and current_weight > 0.10:
		action = "TRIM"
		note = "Position size elevated; valuation historically stretched."
	elif view_delta > 0.03 and not historically_expensive:
		action = "BUY"
		note = "Positive BL view with fair-to-cheap valuation."
	else:
		action = "HOLD"
		note = "No compelling action signal at current valuation."

	return {
		"ticker":          ticker,
		"action":          action,
		"view_delta":      view_delta,
		"valuation_signal": val_signal,
		"pe_percentile":   pe_pctile,
		"quality_score":   quality_score,
		"note":            note
	}

# ───────────────────────────────────────────────────────────────────────────
# STAGE 1 — composite score
# ───────────────────────────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    "bl_view_delta":    0.40,   # forward return vs market — highest weight
    "pe_percentile":    0.25,   # cheapness vs own history and sector
    "valuation_signal": 0.20,   # combined cheap/fair/expensive signal
    "quality":          0.15,   # gross margin, earnings beats, PEG
}


def compute_composite_score(
    bl_view:           dict,
    valuation_result:  dict,
    quality_metrics:   dict,
) -> float:
    """
    Returns a single score in the range [-1, +1].
    Positive = opportunity, negative = overvalued / unattractive.

    Each component is normalised to [-1, +1] before weighting so
    that no single metric dominates due to scale differences.
    """
    components = {}

    # ── BL view delta → [-1, +1] ──────────────────────────────────────────
    # Typical range is roughly -0.15 to +0.15 (annualised excess return).
    # Clip and scale so ±15% maps to ±1.
    view_delta = bl_view.get("view_return_delta", 0.0) or 0.0
    components["bl_view_delta"] = float(np.clip(view_delta / 0.15, -1.0, 1.0))

    # ── PE percentile → [-1, +1] ──────────────────────────────────────────
    # pe_percentile is 0–100 where LOW = historically cheap = GOOD.
    # Invert and centre: 0th pct → +1.0, 50th → 0.0, 100th → -1.0
    pe_pctile = valuation_result.get("pe_vs_history_pctile")
    if pe_pctile is not None:
        components["pe_percentile"] = 1.0 - (pe_pctile / 50.0)   # 0→+1, 50→0, 100→-1
        components["pe_percentile"] = float(np.clip(components["pe_percentile"], -1.0, 1.0))
    else:
        # No history available — fall back to sector comparison
        sector_prem = valuation_result.get("pe_vs_sector_premium_pct", 0.0) or 0.0
        # -30% discount vs sector → +0.6, +30% premium → -0.6
        components["pe_percentile"] = float(np.clip(-sector_prem / 50.0, -1.0, 1.0))

    # ── Valuation signal → [-1, +1] ───────────────────────────────────────
    # Weighted vote of all sub-signals already computed in valuation_result.
    signal_map  = {"cheap": 1.0, "fair": 0.0, "expensive": -1.0, "unknown": 0.0}
    val_signal  = valuation_result.get("valuation_signal", "fair")
    confidence  = valuation_result.get("signal_confidence", 0.5)
    # Scale by confidence: a high-confidence cheap signal scores higher
    # than a low-confidence cheap signal
    components["valuation_signal"] = signal_map.get(val_signal, 0.0) * confidence

    # ── Quality → [0, +1] ─────────────────────────────────────────────────
    # Quality only adds to score, never subtracts — a low-quality stock
    # at a cheap valuation still gets a positive valuation signal,
    # it just doesn't get the quality bonus.
    quality_score = quality_metrics.get("quality_score", 0)   # 0–3 from existing function
    components["quality"] = float(np.clip(quality_score / 3.0, 0.0, 1.0))

    # ── Weighted sum ───────────────────────────────────────────────────────
    total = sum(
        components[k] * SCORE_WEIGHTS[k]
        for k in SCORE_WEIGHTS
        if k in components
    )

    return float(np.clip(total, -1.0, 1.0))


# ───────────────────────────────────────────────────────────────────────────
# STAGE 2 — rank and rebalance
# ───────────────────────────────────────────────────────────────────────────

def rank_and_rebalance(
    ticker_analyses:  list[dict],
    current_weights:  dict[str, float],
    max_position:     float = 0.25,    # hard ceiling per ticker
    min_position:     float = 0.02,    # hard floor (set to 0 to allow full exit)
    max_adjust:       float = 0.05,    # maximum weight change in a single rebalance
    sell_threshold:   float = -0.40,   # composite score below this → force to min
    buy_threshold:    float = 0.40,    # composite score above this → eligible for max
) -> list[dict]:
    """
    Converts per-ticker composite scores into new portfolio weights.

    Parameters
    ----------
    ticker_analyses : list of dicts, each containing:
        {
          "ticker":            str,
          "composite_score":   float,   from compute_composite_score()
          "action":            str,     from compute_value_recommendation()
          "current_weight":    float,
          "bl_view":           dict,
          "valuation_result":  dict,
          "quality_metrics":   dict,
        }
    current_weights : {ticker: weight}, must sum to ~1.0
    max_adjust      : largest single rebalance step. 0.05 = 5% max shift per run.
                      Prevents the system from making large sudden moves on one
                      week's signal — value rebalancing should be gradual.

    Returns
    -------
    list of dicts with ticker, composite_score, rank, current_weight,
    raw_adjustment, new_weight, action, note
    """

    if not ticker_analyses:
        return []

    tickers = [t["ticker"] for t in ticker_analyses]
    scores  = [t["composite_score"] for t in ticker_analyses]
    n       = len(tickers)

    # ── Rank (1 = best opportunity, N = worst) ────────────────────────────
    # argsort descending: index of highest score first
    rank_order  = np.argsort(scores)[::-1]
    ranks       = np.empty(n, dtype=int)
    for rank_pos, ticker_idx in enumerate(rank_order):
        ranks[ticker_idx] = rank_pos + 1   # 1-indexed

    # ── Score-proportional adjustments (sum to zero) ──────────────────────
    scores_arr  = np.array(scores, dtype=float)
    mean_score  = scores_arr.mean()
    deviations  = scores_arr - mean_score      # centred: sum is 0

    max_dev = np.abs(deviations).max()
    if max_dev > 0:
        # Scale so the largest deviation maps to max_adjust
        raw_adjustments = (deviations / max_dev) * max_adjust
    else:
        # All scores identical — no rebalancing warranted
        raw_adjustments = np.zeros(n)

    # ── Apply action constraints ───────────────────────────────────────────
    # The action label from compute_value_recommendation() acts as a hard
    # constraint that overrides the rank-based adjustment when the signal
    # is strong enough.
    constrained_adjustments = raw_adjustments.copy()

    for i, analysis in enumerate(ticker_analyses):
        action = analysis.get("action", "HOLD")
        score  = scores[i]
        cw     = current_weights.get(tickers[i], 0.0)

        # SELL: force adjustment to be negative (trim toward min_position)
        if action == "SELL":
            constrained_adjustments[i] = min(raw_adjustments[i], -max_adjust * 0.5)

        # TRIM: cap adjustment at 0 (can only reduce or hold, not add)
        elif action == "TRIM":
            constrained_adjustments[i] = min(raw_adjustments[i], 0.0)

        # HOLD: dampen adjustment — don't make large moves on hold signals
        elif action == "HOLD":
            constrained_adjustments[i] = raw_adjustments[i] * 0.3

        # BUY: allow full adjustment, but only if not already at max
        elif action == "BUY":
            if cw >= max_position:
                constrained_adjustments[i] = 0.0   # already maxed out

        # Score threshold overrides — independent of action label
        if score < sell_threshold:
            constrained_adjustments[i] = min(constrained_adjustments[i], -max_adjust * 0.5)
        elif score > buy_threshold and action == "BUY":
            # Strong signal: allow up to the full max_adjust
            constrained_adjustments[i] = max(constrained_adjustments[i], max_adjust * 0.5)

    # ── Re-centre adjustments so they sum to zero ─────────────────────────
    # Constraints above may have broken the zero-sum property.
    # Redistribute the residual proportionally across unconstrained tickers.
    _recentre_adjustments(constrained_adjustments, ticker_analyses)

    # ── Compute new weights ────────────────────────────────────────────────
    new_weights = np.array([
        current_weights.get(t, 0.0) for t in tickers
    ]) + constrained_adjustments

    # Clip to [min_position, max_position]
    new_weights = np.clip(new_weights, min_position, max_position)

    # Renormalise to sum to 1.0
    total = new_weights.sum()
    if total > 0:
        new_weights = new_weights / total

    # ── Build output ───────────────────────────────────────────────────────
    results = []
    for i, analysis in enumerate(ticker_analyses):
        ticker = tickers[i]
        cw     = current_weights.get(ticker, 0.0)
        nw     = float(new_weights[i])
        adj    = nw - cw

        results.append({
            "ticker":          ticker,
            "rank":            int(ranks[i]),
            "composite_score": round(float(scores[i]), 3),
            "current_weight":  round(cw, 4),
            "adjustment":      round(adj, 4),
            "new_weight":      round(nw, 4),
            "action":          analysis.get("action", "HOLD"),
            "note":            analysis.get("note", ""),
            "score_components": {
                "bl_view_delta":    round(float(scores_arr[i]), 3),
                "pe_percentile":    analysis.get("valuation_result", {}).get("pe_vs_history_pctile"),
                "valuation_signal": analysis.get("valuation_result", {}).get("valuation_signal"),
                "quality_score":    analysis.get("quality_metrics", {}).get("quality_score"),
            }
        })

    # Sort output by rank (best opportunity first)
    results.sort(key=lambda x: x["rank"])
    return results


def _recentre_adjustments(adjustments: np.ndarray, analyses: list[dict]):
    """
    After applying action constraints, adjustments may no longer sum to zero.
    Redistribute the residual across tickers whose action is HOLD or BUY
    and which have room to absorb it (not already at bounds).
    Modifies adjustments in place.
    """
    residual = adjustments.sum()
    if abs(residual) < 1e-6:
        return

    # Tickers eligible to absorb redistribution
    eligible = [
        i for i, a in enumerate(analyses)
        if a.get("action") in ("HOLD", "BUY")
    ]

    if not eligible:
        # Force-distribute evenly if nothing eligible
        adjustments -= residual / len(adjustments)
        return

    per_ticker = residual / len(eligible)
    for i in eligible:
        adjustments[i] -= per_ticker


def run_robust_mean_variance(posterior_mu, posterior_cov, tickers, constraints):
    """
    Robust Mean-Variance optimisation.
    Maximises worst-case return within an ellipsoidal uncertainty set
    around the posterior mu estimates.
    
    Equivalent to maximising:
        mu.T @ w - epsilon * sqrt(w.T @ Sigma @ w) - lambda * w.T @ Sigma @ w
    
    Where epsilon controls robustness and lambda controls risk aversion.
    """
    n       = len(tickers)
    mu      = posterior_mu.values if hasattr(posterior_mu, 'values') else np.array(posterior_mu)
    sigma   = posterior_cov.values if hasattr(posterior_cov, 'values') else np.array(posterior_cov)
    
    epsilon        = 0.05
    risk_aversion  = constraints["risk_aversion"]
    min_w          = 0
    max_w          = constraints["max_allocation_per_asset"]

    # Guard against NaN (e.g. ticker with no price history) and floating-point non-PD noise.
    sigma = np.nan_to_num(sigma, nan=0.0)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, 1e-8)
    sigma = (eigvecs @ np.diag(eigvals) @ eigvecs.T)
    sigma = (sigma + sigma.T) / 2

    L = np.linalg.cholesky(sigma)
    w = cp.Variable(n)

    port_variance = cp.quad_form(w, sigma)
    # cp.norm(L.T @ w, 2) == sqrt(w.T Sigma w) but is DCP-compliant as a 2-norm
    robustness_penalty = epsilon * cp.norm(L.T @ w, 2)
    
    # Objective: maximise risk-adjusted return minus robustness penalty
    objective = cp.Maximize(
        mu @ w 
        - robustness_penalty
        - 0.5 * risk_aversion * port_variance
    )
    
    constraints = [
        cp.sum(w) == 1,          # weights sum to 1
        w >= min_w,              # minimum position size
        w <= max_w,              # maximum position size
    ]
    
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.CLARABEL)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver status: {problem.status}")
        
        raw_weights = w.value
        
        # Clean small numerical noise
        raw_weights = np.clip(raw_weights, min_w, max_w)
        raw_weights = raw_weights / raw_weights.sum()
        
        return {t: round(float(raw_weights[i]), 6) for i, t in enumerate(tickers)}, None
        
    except Exception as e:
        print(f"    [warn] Robust MV failed: {e}")
        return None, str(e)


def run_optimisation(posterior_mu, posterior_cov, tickers, valuation, constraints):
    """
    Runs all optimisation strategies and returns results dict.
    All strategy weights stored for report comparison.
    """
    results = {}
    errors  = []
    
    mu_series  = pd.Series(posterior_mu)
    cov_df     = pd.DataFrame(posterior_cov) if isinstance(posterior_cov, dict) else posterior_cov
    
    bounds  = (0, constraints["max_allocation_per_asset"])
    sector_mapper, sector_lower, sector_upper = build_sector_constraints(tickers, valuation, constraints)
    
    # ── Max Sharpe ────────────────────────────────────────────────
    try:
        ef = EfficientFrontier(mu_series, cov_df, weight_bounds=bounds)
        ef.add_sector_constraints(sector_mapper, sector_lower=sector_lower, sector_upper=sector_upper)
        ef.max_sharpe()
        results["max_sharpe"] = dict(ef.clean_weights())
    except Exception as e:
        errors.append(f"Max Sharpe failed: {e}")
        results["max_sharpe"] = None
        
    # ── Minimum Variance ─────────────────────────────────────────
    try:
        ef = EfficientFrontier(mu_series, cov_df, weight_bounds=bounds)
        ef.add_sector_constraints(sector_mapper, sector_lower=sector_lower, sector_upper=sector_upper)
        ef.min_volatility()
        results["min_variance"] = dict(ef.clean_weights())
    except Exception as e:
        errors.append(f"Min Variance failed: {e}")
        results["min_variance"] = None
        
    # ── Risk Parity ───────────────────────────────────────────────
    try:
        hrp = HRPOpt(returns=None, cov_matrix=cov_df)
        hrp.optimize()
        results["risk_parity"] = dict(hrp.clean_weights())
    except Exception as e:
        errors.append(f"Risk Parity failed: {e}")
        results["risk_parity"] = None
        
    # ── Target Return ─────────────────────────────────────────────
    try:
        target = 0.08 # Default target return
        ef = EfficientFrontier(mu_series, cov_df, weight_bounds=bounds)
        ef.add_sector_constraints(sector_mapper, sector_lower=sector_lower, sector_upper=sector_upper)
        ef.efficient_return(target_return=target)
        results["target_return"] = dict(ef.clean_weights())
    except Exception as e:
        errors.append(f"Target Return failed: {e}")
        results["target_return"] = None
        
    # ── Robust Mean-Variance ──────────────────────────────────────
    try:
        rmv_weights, rmv_error = run_robust_mean_variance(
            mu_series, cov_df, tickers, constraints
        )
        if rmv_error:
            raise ValueError(rmv_error)
        results["robust_mv"] = rmv_weights
    except Exception as e:
        errors.append(f"Robust MV failed: {e}")
        results["robust_mv"] = None
        
	# ── Value Based Investing ───────────────────────────────────────
    # try:
    #     results["value_invest"] = dict(ef.clean_weights())
    # except Exception as e:
    #     errors.append(f"Value Investing failed: {e}")
    #     results["value_invest"] = None

    return results, errors


def _action_label(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "BUY"
    elif delta < -threshold:
        return "SELL"
    else:
        return "HOLD"


def _consensus_action(actions: list[str]) -> str:
    """
    Returns the action the majority of strategies agree on.
    If no majority, returns HOLD as the conservative default.
    """
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    
    majority = len(actions) // 2 + 1
    for action, count in counts.items():
        if count >= majority:
            return action
    
    return "HOLD"


def build_recommendation_table(
    tickers: list,
    current_weights: dict,
    recommended_weights: dict,
    bl_views: dict,
    constraints: dict,
) -> list[dict]:
    """
    Builds the recommendation table for the report and Agent 4 (simulator).
    One row per ticker containing current weight, all strategy weights,
    deltas, action labels, and BL view summary.
    
    Returns a list of dicts — one per ticker.
    """
    threshold = constraints["action_threshold"]
    strategies = list(recommended_weights.keys())
    
    table = []
    
    for ticker in tickers:
        current_w = current_weights.get(ticker, 0.0)
        
        row = {
            "ticker":          ticker,
            "current_weight":  round(current_w, 4),
        }
        
        # Add weight and delta per strategy
        strategy_deltas = {}
        for strategy in strategies:
            weights = recommended_weights.get(strategy) or {}
            rec_w   = weights.get(ticker, 0.0)
            delta   = rec_w - current_w
            
            row[f"{strategy}_weight"] = round(rec_w, 4)
            row[f"{strategy}_delta"]  = round(delta, 4)
            row[f"{strategy}_action"] = _action_label(delta, threshold)
            
            strategy_deltas[strategy] = delta
        
        # Consensus action — what do most strategies agree on
        actions = [row[f"{strategy}_action"] for strategy in strategies]
        row["consensus_action"] = _consensus_action(actions)
        
        # BL view summary for this ticker
        view = bl_views.get(ticker, {})
        row["view_return"]         = view.get("view_return")
        row["confidence"]          = view.get("confidence")
        row["sentiment_direction"] = view.get("sentiment_direction")
        row["valuation_signal"]    = view.get("valuation_signal")
        row["conflict"]            = view.get("conflict")
        row["reasoning"]           = view.get("reasoning")
        
        table.append(row)
    
    return table


# LLM Sentiment Analysis
SENTIMENT_PROMPT = """You are a quantitative analyst reviewing news, valuations and sentiment data for assets.

For each ticker below, analyse the summary data, the valuations, the finances and earnings, and, the next earnings call 
and the macro economic data of the shiller cape value, the aaii sentiment (weekly investor sentiment survey), CNN's fear/greed index.

Data:
Macro context:
  CAPE: 
  Fear & Greed:
  AAII Sentiment:

Company context:
  Sector:
  Industry:
  Exchange:

  Sentiment summary: summary

  Quantitative signals: Valuation data, financial health data and earnings data

  Earnings proximity:  Next earnings
    Note: if earnings are within 7 days, reduce confidence by 1 and explicitly indicate earnings are upcoming.
    
{data}

Important context to apply:
  - If the valuation signal is "cheap" based on the stock's own PE history AND the 
    PEG is below 1.5, this is a POSITIVE view signal, not a conflict.
  - A high-quality business (gross margin > 50%, consistent earnings beats) trading 
    at the bottom quartile of its own PE history is a textbook value opportunity.
  - Only flag a conflict when sentiment and valuation genuinely point in opposite 
    directions — not when absolute PE looks high but relative PE is historically low.
  - The "conflict" field should be True only when: valuation is historically expensive 
    AND sentiment is positive, OR valuation is historically cheap AND there is a 
    confirmed negative fundamental catalyst (earnings miss, guidance cut).
  - If sentiment and valuation conflict, note this explicitly and explain your reasoning
  - Be willing to assign negative view_return when sentiment is weak and/or valuation is stretched, 
    or there are material headwinds. Neutral or negative views are expected and appropriate.
  - if Fear&Greed is at 80 (extreme greed) and AAII is bearish, the macro context should pull 
    individual views down even when company-specific news is positive
  - Focus on valuations and quality of business (growth prospects and earnings/revenue).
  - If sentiment is negative but valuation signal is neutral or cheap, explain the divergence — do not restate valuation as expensive.
    
Calibration guide for view_return:
  Strongly positive sentiment + cheap valuation  → +0.04 to +0.06
  Mild positive sentiment, fair valuation        → +0.01 to +0.02
  Neutral sentiment                              → -0.01 to +0.01  (default to 0.0 if truly no signal)
  Mild negative sentiment or stretched valuation → -0.02 to -0.03
  Strongly negative sentiment + expensive        → -0.04 to -0.06

  A neutral sentiment_direction must produce a view_return near 0.0.
  Do not default to positive — negative and zero views are correct and expected
  when the evidence does not support outperformance.
  
Produce a single JSON object where each key is a ticker symbol and 
each value contains:
  view_return:         float  (annual return relative to equilibrium, e.g. 0.03 = 3 percent above)
  confidence:          integer 1-5
  sentiment_direction: string  (positive / negative / neutral)
  valuation_signal:    string  (cheap / expensive / neutral / not applicable)
  conflict:            boolean (true if sentiment and valuation disagree)
  reasoning:           string  (2-3 sentences explaining the view)
  
Return only valid JSON. No preamble, no markdown code fences.
Ticker keys must exactly match the ticker values passed in.
"""


def _parse_views(raw: str, tickers: list[str]) -> dict:
    """
    Parses and validates the LLM JSON response for sentiment views.
    Strips markdown fences, clips numeric fields, and fills neutral fallbacks
    for any tickers missing from the response.
    """
    def _neutral() -> dict:
        return {
            "view_return": 0.0,
            "confidence": 1,
            "sentiment_direction": "neutral",
            "valuation_signal": "neutral",
            "conflict": False,
            "reasoning": "No view generated — using neutral fallback.",
        }

    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        views = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    [warn] LLM returned invalid JSON: {e}")
        return {ticker: _neutral() for ticker in tickers}

    for ticker in tickers:
        if ticker not in views:
            views[ticker] = _neutral()
        else:
            views[ticker]["view_return"] = max(-0.30, min(0.30, float(views[ticker].get("view_return", 0.0))))
            views[ticker]["confidence"]  = max(1, min(5, int(views[ticker].get("confidence", 1))))

    return views


def generate_sentiment_views(
        tickers: list[str],
        summaries: dict,
        earnings_dates: dict,
        valuations: dict,
        financial_health: dict, 
    	earnings_data: dict,
        aaii_sentiment: dict,
        fear_greed: dict,
        cape: float,
        file = None,
    ) -> dict:
    """
    Calls the LLM to interpret sentiment data per ticker and returns a parsed
    dict keyed by ticker. Falls back to neutral values if the call fails.
    """
    data_lines = []

    data_lines.append(f"""Market Data:
        Fear/Greed: {fear_greed}
        aaii_sentiment: {aaii_sentiment}
        Shiller Cape: {cape}""")

    for ticker in tickers:
        sent          = summaries.get(ticker, {})
        earnings_date = earnings_dates.get(ticker, {})
        valuation     = valuations.get(ticker, {})
        finances      = financial_health.get(ticker, {})
        earnings      = earnings_data.get(ticker, {})

        data_lines.append(f"""Ticker = {ticker}:
        News Summary: {sent}, \
        Next Earnings Date: {earnings_date}
        Valuation Data: {valuation}
		Financial Data: {finances}
		Earnings Data: {earnings}""")

    prompt = SENTIMENT_PROMPT.format(
        data="\n".join(data_lines),
    )

    try:
        llm      = get_llm()
        response = llm.invoke(prompt)
        raw = response.content.strip()
        return _parse_views(raw, tickers)
    except Exception as e:
        print(f"    [warn] LLM call failed: {e}")
        return _parse_views("", tickers)


def agent3_advisor(state: PipelineState) -> dict:
    """
    Agent 3 — Portfolio Advisor
    Generates Black-Litterman views, runs optimisation,
    produces recommendation table and advisory commentary.
    """
    print(f"  [Agent 3] Advisor running")
    
    user_name = state["user_name"]
    user_path = Path(state["user_path"])
    tickers   = state["tickers"]
    run_date  = state["run_date"]
    errors    = list(state.get("errors") or [])
    existing_errors = len(errors)
    prior_mu = [state["mu_sigma"][t]["mu_annual"] for t in tickers]
    cov_df = pd.DataFrame(state["covariance_matrix"])
    cov_matrix = cov_df.loc[tickers, tickers]
    
    constraints = load_constraints(user_path)
    
    bl_views = generate_sentiment_views(tickers, state["summaries"], state["earnings_dates"],
                    state["valuation"], state["financial_health"], state["earnings_data"],
                    state["aaii_sentiment"], state["fear_greed"], state["cape"])
    
    views = {
        ticker: prior_mu[i] + bl_views[ticker]["view_return"]
        for i, ticker in enumerate(tickers)
    }
    variances = np.diag(cov_matrix.values)   # diagonal of covariance matrix
    omega_diag = np.array([
        confidence_to_omega(
            bl_views[t]["confidence"],
            variances[i],
            UNCERTAINTY_SCALE
        )
        for i, t in enumerate(tickers)
    ])
    omega = np.diag(omega_diag)
    bl = BlackLittermanModel(
        cov_matrix   = cov_matrix,
        pi           = np.array(prior_mu),          # factor model mu as prior
        absolute_views = views,           # {ticker: view_return}
        omega        = omega,             # uncertainty matrix
    )
    
    posterior_mu  = bl.bl_returns().to_dict()    # pandas Series, one value per ticker
    posterior_cov = bl.bl_cov()        # DataFrame, posterior covariance matrix  
    
    
    sum_dir = Path(state["user_path"]) / "data" / state["run_date"] / "summaries"
    sum_dir.mkdir(parents=True, exist_ok=True)
    for i, ticker in enumerate(tickers):
        dir_name = ticker.removesuffix(".TO")
        (sum_dir / dir_name).mkdir(parents=True, exist_ok=True)
        with open(sum_dir / dir_name / "advisor.txt", "w", encoding="utf-8") as advisor_file:
            print(f"Historical Returns Estimate: {prior_mu[i]}", file=advisor_file)
            print(f"Views: {bl_views[ticker]}", file=advisor_file)
            print(f"posterior returns: {posterior_mu[ticker]}", file=advisor_file)

    recommended_weights, opt_error = run_optimisation(
        pd.Series(posterior_mu),
        pd.DataFrame(state["covariance_matrix"]).loc[tickers, tickers],
        tickers,
        state["valuation"],
        constraints,
    )

    errors.extend(opt_error)
    
    recommendation_table = build_recommendation_table(
        tickers             = tickers,
        current_weights     = state["current_weights"],
        recommended_weights = recommended_weights,
        bl_views            = bl_views,
        constraints         = constraints,
    )
    
    print(f"\n  [Agent 3] Complete. New Errors: {len(errors) - existing_errors}")

    return {
        "bl_views": bl_views,
        "posterior_mu": posterior_mu,
        "recommended_weights": recommended_weights,
        "recommendation_table": recommendation_table,
        "advisory_commentary": "[STUB] Advisory commentary not yet implemented.",
        "errors": errors
    }