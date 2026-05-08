import json
import numpy as np
from datetime import date

np.random.seed(42)

TICKERS = ["AAPL", "MSFT", "TD.TO", "XEQT.TO", "GOOGL"]

def make_paths(s_current, mu, sigma, n_paths=100, n_steps=252):
    dt = 1/252
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    shocks = np.random.normal(drift, vol, size=(n_steps, n_paths))
    cum_log = np.vstack([np.zeros(n_paths), np.cumsum(shocks, axis=0)])
    paths = s_current * np.exp(cum_log)
    return paths.tolist()

def make_mc_stats(paths_list, s_current):
    paths = np.array(paths_list)
    S_T = paths[-1, :]
    losses = s_current - S_T
    var_95 = float(np.percentile(losses, 95))
    cvar_95 = float(losses[losses >= var_95].mean())
    return {
        "p5":            float(np.percentile(S_T, 5)),
        "p25":           float(np.percentile(S_T, 25)),
        "p50":           float(np.percentile(S_T, 50)),
        "p75":           float(np.percentile(S_T, 75)),
        "p95":           float(np.percentile(S_T, 95)),
        "var_95":        var_95,
        "cvar_95":       cvar_95,
        "prob_up":     float(np.mean(S_T < s_current)),
        "expected_value":float(S_T.mean()),
        "paths":         paths_list,
    }

# ── Ticker data ───────────────────────────────────────────────────
ticker_configs = {
    "AAPL":   {"s": 189.50, "mu": 0.092, "sigma": 0.24, "w": 0.283, "shares": 10,  "mktval": 1895.0,  "currency": "USD"},
    "MSFT":   {"s": 415.20, "mu": 0.108, "sigma": 0.22, "w": 0.221, "shares": 5,   "mktval": 2076.0,  "currency": "USD"},
    "TD.TO":  {"s": 81.40,  "mu": 0.061, "sigma": 0.18, "w": 0.196, "shares": 20,  "mktval": 1628.0,  "currency": "CAD"},
    "XEQT.TO":{"s": 27.50,  "mu": 0.072, "sigma": 0.15, "w": 0.142, "shares": 50,  "mktval": 1375.0,  "currency": "CAD"},
    "GOOGL":  {"s": 175.30, "mu": 0.085, "sigma": 0.26, "w": 0.158, "shares": 8,   "mktval": 1402.4,  "currency": "USD"},
}

# ── Build mc_current (per-ticker) ─────────────────────────────────
mc_current = {}
for t, cfg in ticker_configs.items():
    paths = make_paths(cfg["s"], cfg["mu"], cfg["sigma"])
    mc_current[t] = make_mc_stats(paths, cfg["s"])

# ── Build portfolio-level simulations ─────────────────────────────
total_value = 8376.4   # CAD

def make_portfolio_paths(weights, total_value, n_paths=100, n_steps=252):
    dt = 1/252
    port_mu    = sum(cfg["mu"] * weights.get(t, 0) for t, cfg in ticker_configs.items())
    port_sigma = sum(cfg["sigma"] * weights.get(t, 0) for t, cfg in ticker_configs.items()) * 0.85
    drift = (port_mu - 0.5 * port_sigma**2) * dt
    vol = port_sigma * np.sqrt(dt)
    shocks = np.random.normal(drift, vol, size=(n_steps, n_paths))
    cum_log = np.vstack([np.zeros(n_paths), np.cumsum(shocks, axis=0)])
    paths = total_value * np.exp(cum_log)
    return paths.tolist()

current_weights = {t: cfg["w"] for t, cfg in ticker_configs.items()}

curr_port_paths = make_portfolio_paths(current_weights, total_value)
mc_portfolio_current = make_mc_stats(curr_port_paths, total_value)

strategies = {
    "max_sharpe":  {"AAPL": 0.312, "MSFT": 0.184, "TD.TO": 0.201, "XEQT.TO": 0.120, "GOOGL": 0.183},
    "min_variance":{"AAPL": 0.271, "MSFT": 0.236, "TD.TO": 0.214, "XEQT.TO": 0.180, "GOOGL": 0.099},
    "risk_parity": {"AAPL": 0.248, "MSFT": 0.212, "TD.TO": 0.228, "XEQT.TO": 0.192, "GOOGL": 0.120},
    "target_return":{"AAPL": 0.294, "MSFT": 0.208, "TD.TO": 0.201, "XEQT.TO": 0.130, "GOOGL": 0.167},
    "robust_mv":   {"AAPL": 0.301, "MSFT": 0.199, "TD.TO": 0.208, "XEQT.TO": 0.142, "GOOGL": 0.150},
}

mc_portfolio_recommended = {}
for strat, weights in strategies.items():
    paths = make_portfolio_paths(weights, total_value)
    mc_portfolio_recommended[strat] = make_mc_stats(paths, total_value)

# ── Build full state ──────────────────────────────────────────────
state = {
    "user_name":             "test_user",
    "user_path":             "users/test_user",
    "run_date":              str(date.today()),
    "tickers":               TICKERS,
    "current_weights":       current_weights,
    "total_portfolio_value": total_value,
    "usd_cad_rate":          1.362,
    "excluded_tickers":      [],

    "portfolio_rows": [
        {"ticker": t, "shares": cfg["shares"], "avg_cost": round(cfg["s"] * 0.85, 2),
         "market_value": cfg["mktval"], "type": "etf" if ".TO" in t and "XEQT" in t else "stock",
         "currency": cfg["currency"]}
        for t, cfg in ticker_configs.items()
    ],

    # ── Agent 1 outputs ───────────────────────────────────────────
    "summaries": {
        "AAPL":    "Sentiment is broadly positive driven by strong services revenue growth and analyst upgrades. Multiple credible sources highlight iPhone 16 cycle momentum. Some concern around China exposure and FX headwinds. Consensus price target implies 12% upside from current levels.",
        "MSFT":    "Azure cloud growth re-accelerating according to multiple analyst reports. Copilot AI integration receiving positive early enterprise feedback. Valuation appears stretched at current forward P/E. No major negative catalysts identified this week.",
        "TD.TO":   "Mixed sentiment following Q2 earnings miss on provisions. Regulatory scrutiny in the US market remains an overhang. Dividend yield attractive at current price. Long-term franchise value intact according to most Canadian financial media.",
        "XEQT.TO": "Broad market ETF with no company-specific sentiment. Macro commentary generally constructive for equities medium-term despite elevated CAPE.",
        "GOOGL":   "Search revenue resilient despite AI competition concerns. YouTube advertising showing recovery. Regulatory risks in EU elevated. Waymo progress cited positively by multiple sources.",
    },
    "aaii_sentiment": {
        "date":             str(date.today()),
        "bullish":          "38.5%",
        "neutral":          "29.2%",
        "bearish":          "32.3%",
        "bull_bear_spread": "+6.2%",
    },
    "fear_greed": {
        "score":  58,
        "rating": "Greed",
    },

    # ── Agent 2 outputs ───────────────────────────────────────────
    "factor_results": {
        "AAPL":    {"alpha_daily": 0.000182, "b_mkt": 1.21, "b_smb": -0.18, "b_hml": -0.42, "r2": 0.68, "r2_adj": 0.677, "p_val_alpha": 0.031, "mu_annual": 0.092, "rf_annual": 0.052},
        "MSFT":    {"alpha_daily": 0.000241, "b_mkt": 1.15, "b_smb": -0.22, "b_hml": -0.51, "r2": 0.71, "r2_adj": 0.707, "p_val_alpha": 0.018, "mu_annual": 0.108, "rf_annual": 0.052},
        "TD.TO":   {"alpha_daily": -0.000041,"b_mkt": 0.88, "b_smb": 0.14,  "b_hml": 0.62,  "r2": 0.55, "r2_adj": 0.546, "p_val_alpha": 0.412, "mu_annual": 0.061, "rf_annual": 0.048},
        "XEQT.TO": {"alpha_daily": 0.000012, "b_mkt": 0.95, "b_smb": 0.08,  "b_hml": 0.11,  "r2": 0.92, "r2_adj": 0.919, "p_val_alpha": 0.721, "mu_annual": 0.072, "rf_annual": 0.048},
        "GOOGL":   {"alpha_daily": 0.000158, "b_mkt": 1.18, "b_smb": -0.15, "b_hml": -0.38, "r2": 0.64, "r2_adj": 0.636, "p_val_alpha": 0.044, "mu_annual": 0.085, "rf_annual": 0.052},
    },
    "garch_results": {
        "AAPL":    {"omega_garch": 0.0000012, "alpha_garch": 0.091, "beta_garch": 0.892, "garch_persist": 0.983, "garch_long_run_vol": 0.228, "sigma_current_vol": 0.0148, "sigma_total_annual": 0.241, "vol_regime": "elevated"},
        "MSFT":    {"omega_garch": 0.0000008, "alpha_garch": 0.078, "beta_garch": 0.901, "garch_persist": 0.979, "garch_long_run_vol": 0.198, "sigma_current_vol": 0.0132, "sigma_total_annual": 0.219, "vol_regime": "elevated"},
        "TD.TO":   {"omega_garch": 0.0000015, "alpha_garch": 0.082, "beta_garch": 0.878, "garch_persist": 0.960, "garch_long_run_vol": 0.182, "sigma_current_vol": 0.0114, "sigma_total_annual": 0.181, "vol_regime": "normal"},
        "XEQT.TO": {"omega_garch": 0.0000006, "alpha_garch": 0.065, "beta_garch": 0.912, "garch_persist": 0.977, "garch_long_run_vol": 0.148, "sigma_current_vol": 0.0094, "sigma_total_annual": 0.149, "vol_regime": "elevated"},
        "GOOGL":   {"omega_garch": 0.0000018, "alpha_garch": 0.098, "beta_garch": 0.871, "garch_persist": 0.969, "garch_long_run_vol": 0.251, "sigma_current_vol": 0.0162, "sigma_total_annual": 0.258, "vol_regime": "normal"},
    },
    "mu_sigma": {
        "AAPL":    {"mu_annual": 0.092, "sigma_annual": 0.241, "s_current": 189.50, "is_fallback": False, "excluded_from_sim": False},
        "MSFT":    {"mu_annual": 0.108, "sigma_annual": 0.219, "s_current": 415.20, "is_fallback": False, "excluded_from_sim": False},
        "TD.TO":   {"mu_annual": 0.061, "sigma_annual": 0.181, "s_current": 81.40,  "is_fallback": False, "excluded_from_sim": False},
        "XEQT.TO": {"mu_annual": 0.072, "sigma_annual": 0.149, "s_current": 27.50,  "is_fallback": False, "excluded_from_sim": False},
        "GOOGL":   {"mu_annual": 0.085, "sigma_annual": 0.258, "s_current": 175.30, "is_fallback": False, "excluded_from_sim": False},
    },
    "valuation": {
        "AAPL":    {"fwd_pe": 28.4, "ttm_pe": 31.2, "peg": 2.8,  "ev_ebitda": 22.1, "target_price": 212.0, "analyst_rec": "Buy",  "200MA": 181.2, "50MA": 186.4, "sector": "Technology",  "industry": "Consumer Electronics", "earnings_growth": 0.082, "revenue_growth": 0.062},
        "MSFT":    {"fwd_pe": 32.1, "ttm_pe": 35.8, "peg": 2.2,  "ev_ebitda": 24.8, "target_price": 465.0, "analyst_rec": "Strong Buy", "200MA": 388.1, "50MA": 408.2, "sector": "Technology", "industry": "Software", "earnings_growth": 0.142, "revenue_growth": 0.158},
        "TD.TO":   {"fwd_pe": 10.2, "ttm_pe": 11.8, "peg": 1.4,  "ev_ebitda": 11.2, "target_price": 88.0,  "analyst_rec": "Hold", "200MA": 79.8,  "50MA": 80.1,  "sector": "Financials",  "industry": "Banks", "earnings_growth": -0.042, "revenue_growth": 0.028},
        "XEQT.TO": {"fwd_pe": None, "ttm_pe": None, "peg": None, "ev_ebitda": None,  "target_price": None,  "analyst_rec": None,  "200MA": 26.8,  "50MA": 27.1,  "sector": None, "industry": "ETF", "earnings_growth": None, "revenue_growth": None},
        "GOOGL":   {"fwd_pe": 21.8, "ttm_pe": 24.1, "peg": 1.2,  "ev_ebitda": 16.4, "target_price": 198.0, "analyst_rec": "Buy",  "200MA": 162.4, "50MA": 170.8, "sector": "Communication Services", "industry": "Internet Content", "earnings_growth": 0.118, "revenue_growth": 0.134},
    },
    "financial_health": {
        "AAPL":    {"revenue_growth_1yr": 0.062, "revenue_growth_3yr": 0.058, "gross_margin_current": 0.461, "gross_margin_expanding": True,  "gross_margin_trend": "expanding",   "fcf_current": 99580000000, "fcf_margin": 0.261, "earnings_quality": 1.18, "debt_to_fcf": 0.8,  "roic": 0.542},
        "MSFT":    {"revenue_growth_1yr": 0.158, "revenue_growth_3yr": 0.148, "gross_margin_current": 0.698, "gross_margin_expanding": True,  "gross_margin_trend": "expanding",   "fcf_current": 63280000000, "fcf_margin": 0.342, "earnings_quality": 1.24, "debt_to_fcf": 0.5,  "roic": 0.318},
        "TD.TO":   {"revenue_growth_1yr": 0.028, "revenue_growth_3yr": 0.041, "gross_margin_current": 0.512, "gross_margin_expanding": False, "gross_margin_trend": "contracting", "fcf_current": 8420000000,  "fcf_margin": 0.188, "earnings_quality": 0.94, "debt_to_fcf": 3.2,  "roic": 0.082},
        "XEQT.TO": {},
        "GOOGL":   {"revenue_growth_1yr": 0.134, "revenue_growth_3yr": 0.118, "gross_margin_current": 0.561, "gross_margin_expanding": True,  "gross_margin_trend": "mixed",       "fcf_current": 52640000000, "fcf_margin": 0.218, "earnings_quality": 1.31, "debt_to_fcf": 0.2,  "roic": 0.241},
    },
    "earnings_data": {
        "AAPL":    {"next_earnings_date": "2025-07-29", "recent_quarters": [{"date": "2025-05-01", "eps_estimate": 1.61, "eps_actual": 1.65, "surprise_pct": 2.48}, {"date": "2025-01-30", "eps_estimate": 2.35, "eps_actual": 2.40, "surprise_pct": 2.13}, {"date": "2024-10-31", "eps_estimate": 1.60, "eps_actual": 1.64, "surprise_pct": 2.50}, {"date": "2024-08-01", "eps_estimate": 1.35, "eps_actual": 1.40, "surprise_pct": 3.70}], "avg_eps_surprise_pct": 2.70, "consecutive_beats": 4},
        "MSFT":    {"next_earnings_date": "2025-07-23", "recent_quarters": [{"date": "2025-04-30", "eps_estimate": 3.22, "eps_actual": 3.46, "surprise_pct": 7.45}, {"date": "2025-01-29", "eps_estimate": 3.11, "eps_actual": 3.23, "surprise_pct": 3.86}, {"date": "2024-10-30", "eps_estimate": 3.10, "eps_actual": 3.30, "surprise_pct": 6.45}, {"date": "2024-07-30", "eps_estimate": 2.94, "eps_actual": 2.95, "surprise_pct": 0.34}], "avg_eps_surprise_pct": 4.53, "consecutive_beats": 4},
        "TD.TO":   {"next_earnings_date": "2025-08-21", "recent_quarters": [{"date": "2025-05-22", "eps_estimate": 1.98, "eps_actual": 1.87, "surprise_pct": -5.56}, {"date": "2025-02-27", "eps_estimate": 2.01, "eps_actual": 1.95, "surprise_pct": -2.99}, {"date": "2024-11-28", "eps_estimate": 2.05, "eps_actual": 2.08, "surprise_pct": 1.46}, {"date": "2024-08-22", "eps_estimate": 1.96, "eps_actual": 2.04, "surprise_pct": 4.08}], "avg_eps_surprise_pct": -0.75, "consecutive_beats": 0},
        "XEQT.TO": {"next_earnings_date": None, "recent_quarters": [], "avg_eps_surprise_pct": None, "consecutive_beats": 0},
        "GOOGL":   {"next_earnings_date": "2025-07-29", "recent_quarters": [{"date": "2025-04-29", "eps_estimate": 2.01, "eps_actual": 2.81, "surprise_pct": 39.80}, {"date": "2025-02-04", "eps_estimate": 2.13, "eps_actual": 2.15, "surprise_pct": 0.94}, {"date": "2024-10-29", "eps_estimate": 1.85, "eps_actual": 2.12, "surprise_pct": 14.59}, {"date": "2024-07-30", "eps_estimate": 1.84, "eps_actual": 1.89, "surprise_pct": 2.72}], "avg_eps_surprise_pct": 14.51, "consecutive_beats": 4},
    },
    "earnings_dates": {
        "AAPL":    "2025-07-29",
        "MSFT":    "2025-07-23",
        "TD.TO":   "2025-08-21",
        "XEQT.TO": None,
        "GOOGL":   "2025-07-29",
    },
    "cape": 34.8,
    "covariance_matrix": {
        "AAPL":    {"AAPL": 0.0581, "MSFT": 0.0312, "TD.TO": 0.0148, "XEQT.TO": 0.0201, "GOOGL": 0.0298},
        "MSFT":    {"AAPL": 0.0312, "MSFT": 0.0480, "TD.TO": 0.0122, "XEQT.TO": 0.0168, "GOOGL": 0.0264},
        "TD.TO":   {"AAPL": 0.0148, "MSFT": 0.0122, "TD.TO": 0.0328, "XEQT.TO": 0.0148, "GOOGL": 0.0112},
        "XEQT.TO": {"AAPL": 0.0201, "MSFT": 0.0168, "TD.TO": 0.0148, "XEQT.TO": 0.0222, "GOOGL": 0.0158},
        "GOOGL":   {"AAPL": 0.0298, "MSFT": 0.0264, "TD.TO": 0.0112, "XEQT.TO": 0.0158, "GOOGL": 0.0665},
    },
    "quant_commentary": "AAPL shows elevated GARCH persistence at 0.983, suggesting current volatility regime will be slow to mean-revert — heightened short-term uncertainty relative to historical norms. MSFT demonstrates strong alpha of 0.024% daily (p=0.018) with expanding gross margins and a consecutive beat streak of four quarters, supporting above-market return expectations. TD.TO alpha is statistically insignificant (p=0.412) and the most recent two quarters show EPS misses, warranting caution on near-term return estimates. XEQT.TO has high R² of 0.92 as expected for a broad market ETF — factor model fits well but alpha is trivially zero by construction. GOOGL's PEG of 1.2 is attractive relative to its growth profile, and the Q1 2025 EPS surprise of 39.8% was exceptional, though partially driven by one-time items.",

    # ── Agent 3 advisor outputs ───────────────────────────────────
    "bl_views": {
        "AAPL":    {"view_return": 0.018, "confidence": 3, "sentiment_direction": "positive", "valuation_signal": "expensive", "conflict": True,  "reasoning": "Sentiment is positive on services momentum but forward P/E of 28.4x and PEG of 2.8 suggest the market has priced in the optimism. Moderately positive view with reduced confidence given valuation stretch."},
        "MSFT":    {"view_return": 0.031, "confidence": 4, "sentiment_direction": "positive", "valuation_signal": "expensive", "conflict": True,  "reasoning": "Azure re-acceleration and Copilot traction are compelling catalysts. Valuation is elevated but justified by superior growth and margin profile. Strong confidence in positive view despite premium multiple."},
        "TD.TO":   {"view_return": -0.018,"confidence": 2, "sentiment_direction": "negative", "valuation_signal": "cheap",     "conflict": True,  "reasoning": "Sentiment is cautious on US regulatory overhang and recent EPS misses. However P/E of 10.2x and EV/EBITDA well below bank sector median suggest valuation has overshot to the downside. Conflict between cheap valuation and negative near-term sentiment."},
        "XEQT.TO": {"view_return": 0.005, "confidence": 1, "sentiment_direction": "neutral",  "valuation_signal": "neutral",   "conflict": False, "reasoning": "Broad market ETF with no company-specific view. Macro environment modestly constructive. Minimal view applied given CAPE of 34.8 suggesting index-level overvaluation."},
        "GOOGL":   {"view_return": 0.028, "confidence": 4, "sentiment_direction": "positive", "valuation_signal": "neutral",   "conflict": False, "reasoning": "Search resilience and YouTube recovery are well-documented across multiple credible sources. PEG of 1.2 is attractive for a company growing earnings at this rate. Sentiment and valuation aligned — high confidence positive view."},
    },
    "posterior_mu": {
        "AAPL":    0.096,
        "MSFT":    0.118,
        "TD.TO":   0.052,
        "XEQT.TO": 0.074,
        "GOOGL":   0.102,
    },
    "recommended_weights": {s: w for s, w in strategies.items()},
    "recommendation_table": [
        {
            "ticker": "AAPL", "current_weight": 0.283,
            "max_sharpe_weight": 0.312, "max_sharpe_delta": 0.029,  "max_sharpe_action": "BUY",
            "min_variance_weight": 0.271, "min_variance_delta": -0.012, "min_variance_action": "HOLD",
            "risk_parity_weight": 0.248, "risk_parity_delta": -0.035, "risk_parity_action": "SELL",
            "target_return_weight": 0.294, "target_return_delta": 0.011, "target_return_action": "HOLD",
            "robust_mv_weight": 0.301, "robust_mv_delta": 0.018, "robust_mv_action": "HOLD",
            "consensus_action": "HOLD",
            "view_return": 0.018, "confidence": 3, "sentiment_direction": "positive",
            "valuation_signal": "expensive", "conflict": True,
            "reasoning": "Sentiment is positive on services momentum but forward P/E of 28.4x and PEG of 2.8 suggest the market has priced in the optimism.",
        },
        {
            "ticker": "MSFT", "current_weight": 0.221,
            "max_sharpe_weight": 0.184, "max_sharpe_delta": -0.037, "max_sharpe_action": "SELL",
            "min_variance_weight": 0.236, "min_variance_delta": 0.015, "min_variance_action": "HOLD",
            "risk_parity_weight": 0.212, "risk_parity_delta": -0.009, "risk_parity_action": "HOLD",
            "target_return_weight": 0.208, "target_return_delta": -0.013, "target_return_action": "HOLD",
            "robust_mv_weight": 0.199, "robust_mv_delta": -0.022, "robust_mv_action": "SELL",
            "consensus_action": "HOLD",
            "view_return": 0.031, "confidence": 4, "sentiment_direction": "positive",
            "valuation_signal": "expensive", "conflict": True,
            "reasoning": "Azure re-acceleration and Copilot traction are compelling catalysts.",
        },
        {
            "ticker": "TD.TO", "current_weight": 0.196,
            "max_sharpe_weight": 0.201, "max_sharpe_delta": 0.005,  "max_sharpe_action": "HOLD",
            "min_variance_weight": 0.214, "min_variance_delta": 0.018, "min_variance_action": "HOLD",
            "risk_parity_weight": 0.228, "risk_parity_delta": 0.032, "risk_parity_action": "BUY",
            "target_return_weight": 0.201, "target_return_delta": 0.005, "target_return_action": "HOLD",
            "robust_mv_weight": 0.208, "robust_mv_delta": 0.012, "robust_mv_action": "HOLD",
            "consensus_action": "HOLD",
            "view_return": -0.018, "confidence": 2, "sentiment_direction": "negative",
            "valuation_signal": "cheap", "conflict": True,
            "reasoning": "Sentiment cautious on US regulatory overhang but valuation at 10.2x P/E looks overdone.",
        },
        {
            "ticker": "XEQT.TO", "current_weight": 0.142,
            "max_sharpe_weight": 0.120, "max_sharpe_delta": -0.022, "max_sharpe_action": "SELL",
            "min_variance_weight": 0.180, "min_variance_delta": 0.038, "min_variance_action": "BUY",
            "risk_parity_weight": 0.192, "risk_parity_delta": 0.050, "risk_parity_action": "BUY",
            "target_return_weight": 0.130, "target_return_delta": -0.012, "target_return_action": "HOLD",
            "robust_mv_weight": 0.142, "robust_mv_delta": 0.000, "robust_mv_action": "HOLD",
            "consensus_action": "HOLD",
            "view_return": 0.005, "confidence": 1, "sentiment_direction": "neutral",
            "valuation_signal": "neutral", "conflict": False,
            "reasoning": "Broad market ETF. Macro modestly constructive. Minimal view given CAPE of 34.8.",
        },
        {
            "ticker": "GOOGL", "current_weight": 0.158,
            "max_sharpe_weight": 0.183, "max_sharpe_delta": 0.025,  "max_sharpe_action": "BUY",
            "min_variance_weight": 0.099, "min_variance_delta": -0.059, "min_variance_action": "SELL",
            "risk_parity_weight": 0.120, "risk_parity_delta": -0.038, "risk_parity_action": "SELL",
            "target_return_weight": 0.167, "target_return_delta": 0.009, "target_return_action": "HOLD",
            "robust_mv_weight": 0.150, "robust_mv_delta": -0.008, "robust_mv_action": "HOLD",
            "consensus_action": "HOLD",
            "view_return": 0.028, "confidence": 4, "sentiment_direction": "positive",
            "valuation_signal": "neutral", "conflict": False,
            "reasoning": "Search resilience and YouTube recovery well-documented. PEG of 1.2 attractive.",
        },
    ],

    # ── Agent 4 simulator outputs ─────────────────────────────────
    "mc_current":               mc_current,
    "mc_portfolio_current":     mc_portfolio_current,
    "mc_portfolio_recommended": mc_portfolio_recommended,

    "risk_commentary": "The current portfolio carries a 95% VaR of approximately $1,240 CAD over a one-year horizon, with conditional expected loss in tail scenarios of $1,680 CAD. The primary risk contributors are GOOGL and AAPL, which account for the two highest individual volatilities at 25.8% and 24.1% annualised respectively. Both AAPL and MSFT are currently in elevated GARCH volatility regimes with persistence above 0.97, indicating that the current elevated volatility is unlikely to mean-revert quickly — this widens the simulation distribution meaningfully relative to long-run averages. The probability of overall portfolio loss over a one-year horizon stands at 28% under current weights, which compares favourably to the broad equity market given the diversification benefit of TD.TO and XEQT.TO.",

    "advisory_commentary": "The portfolio enters this week with a moderately constructive but cautious posture, consistent with the macro backdrop of a CAPE at 34.8 and a Fear and Greed reading of 58. At this valuation level, the CAPE-adjusted equity risk premium implies forward returns below historical averages, and the portfolio's factor model expected returns already reflect this compression through the blended ERP calculation. AAII sentiment shows a modest bull-bear spread of 6.2% — not extreme in either direction — suggesting the market is not pricing in either euphoria or crisis, which is broadly neutral for near-term positioning. The single most actionable signal this week is GOOGL, where sentiment and valuation are aligned — a PEG of 1.2, four consecutive EPS beats averaging 14.5% surprise, and positive coverage across multiple credible sources all support a higher allocation. Max Sharpe recommends increasing GOOGL to 18.3%, and this is the clearest high-conviction call in the portfolio. The complication is that Min Variance and Risk Parity both recommend reducing GOOGL due to its higher standalone volatility of 25.8%, illustrating the tension between return-seeking and risk-minimising frameworks for a volatile but well-positioned stock. TD.TO presents the most interesting conflict: the valuation case is compelling at 10.2x forward P/E and EV/EBITDA well below the bank sector median, but two consecutive EPS misses and ongoing US regulatory overhang have suppressed the BL confidence score to 2 out of 5. This is a classic cheap-for-a-reason situation — the recommended posture is to hold current weight and revisit when the regulatory picture clarifies, rather than adding on valuation alone. Of the five strategies, Robust Mean-Variance offers the most balanced risk-adjusted profile this week — it captures the GOOGL and MSFT upside while maintaining meaningful diversification through TD.TO and XEQT.TO, and its VaR of $1,140 CAD is the lowest among the return-seeking strategies. This is the recommended strategy for investors who want to act on this week's signals without concentrating in the highest-volatility names.",

    "divergence_summary": None,
    "report_path":        None,
    "email_sent":         None,
    "errors":             [],
}

with open("tests/fixtures/sample_state.json", "w") as f:
    json.dump(state, f, indent=2, default=str)

print(f"Sample state written successfully")
print(f"Tickers: {TICKERS}")
print(f"Total portfolio value: ${total_value:,.2f} CAD")
print(f"MC current keys: {list(mc_current['AAPL'].keys())}")
print(f"Strategies: {list(strategies.keys())}")
