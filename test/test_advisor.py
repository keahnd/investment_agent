import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent3_advisor import (
    agent3_advisor
)

# Test 3 — Advisor Agent LLM Call
print("\nTesting Advisor Agent LLM Call...")

test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "HD", "JPM", "BRK-B", "JNJ", "PG", "XOM"]

# Build 10x10 covariance matrix: σ=0.20 for all, intra-sector ρ=0.40, cross-sector ρ=0.15
_sectors_map = {
    "AAPL": "Technology",           "MSFT": "Technology",       "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "JPM":  "Financials",           "BRK-B": "Financials",
    "JNJ":  "Healthcare",
    "PG":   "Consumer Staples",
    "XOM":  "Energy",
}
_VAR, _INTRA, _CROSS = 0.04, 0.016, 0.006
_cov = {}
for t1 in test_tickers:
    _cov[t1] = {}
    for t2 in test_tickers:
        if t1 == t2:
            _cov[t1][t2] = _VAR
        elif _sectors_map[t1] == _sectors_map[t2]:
            _cov[t1][t2] = _INTRA
        else:
            _cov[t1][t2] = _CROSS

_mu = {
    "AAPL": 0.12, "MSFT": 0.14, "GOOGL": 0.13, "AMZN": 0.15, "HD":    0.10,
    "JPM":  0.11, "BRK-B": 0.09, "JNJ":  0.08, "PG":   0.08, "XOM":   0.10,
}

test_state = {
    "user_name": "test_user",
    "user_path": "users/keahn_div",
    "run_date":  "2025-05-01",
    "tickers":   test_tickers,
    "errors":    [],

    "current_weights": {t: 0.10 for t in test_tickers},
    "posterior_mu":    _mu,

    # Agent 1 outputs consumed by Agent 3
    "summaries": {
        "AAPL":  "Apple reported strong iPhone demand and better-than-expected services revenue.",
        "MSFT":  "Microsoft Azure grew 31% YoY. AI Copilot adoption is accelerating.",
        "GOOGL": "Alphabet Search revenue beat estimates; YouTube ad growth reaccelerated.",
        "AMZN":  "AWS grew 17% YoY; retail margins improved on cost discipline.",
        "HD":    "Home Depot missed on same-store sales amid softer housing market.",
        "JPM":   "JPMorgan posted record revenue; net interest income guidance raised.",
        "BRK-B": "Berkshire added to equity positions; cash pile near record highs.",
        "JNJ":   "Johnson & Johnson MedTech segment grew 8%; pharma pipeline strong.",
        "PG":    "Procter & Gamble raised guidance on pricing power and volume recovery.",
        "XOM":   "ExxonMobil beat on refining margins; Guyana output ahead of schedule.",
    },
    "aaii_sentiment": {"bullish": 38.5, "bearish": 29.2, "neutral": 32.3},
    "fear_greed":     {"score": 62, "rating": "Greed"},

    # Agent 2 outputs consumed by Agent 3
    "mu_sigma": {
        "AAPL":  {"mu_annual": 0.12, "sigma_annual": 0.22, "s_current": 185.5, "is_fallback": False},
        "MSFT":  {"mu_annual": 0.14, "sigma_annual": 0.20, "s_current": 415.0, "is_fallback": False},
        "GOOGL": {"mu_annual": 0.13, "sigma_annual": 0.23, "s_current": 175.0, "is_fallback": False},
        "AMZN":  {"mu_annual": 0.15, "sigma_annual": 0.25, "s_current": 195.0, "is_fallback": False},
        "HD":    {"mu_annual": 0.10, "sigma_annual": 0.20, "s_current": 380.0, "is_fallback": False},
        "JPM":   {"mu_annual": 0.11, "sigma_annual": 0.22, "s_current": 195.0, "is_fallback": False},
        "BRK-B": {"mu_annual": 0.09, "sigma_annual": 0.18, "s_current": 390.0, "is_fallback": False},
        "JNJ":   {"mu_annual": 0.08, "sigma_annual": 0.16, "s_current": 158.0, "is_fallback": False},
        "PG":    {"mu_annual": 0.08, "sigma_annual": 0.15, "s_current": 165.0, "is_fallback": False},
        "XOM":   {"mu_annual": 0.10, "sigma_annual": 0.24, "s_current": 112.0, "is_fallback": False},
    },
    "valuation": {
        "AAPL":  {"fwd_pe": 28.5, "ttm_pe": 30.1, "peg": 2.1, "ev_ebitda": 22.0,
                  "sector": "Technology",              "industry": "Consumer Electronics",         "exchange": "NMS"},
        "MSFT":  {"fwd_pe": 32.0, "ttm_pe": 35.2, "peg": 2.3, "ev_ebitda": 24.0,
                  "sector": "Technology",              "industry": "Software—Infrastructure",      "exchange": "NMS"},
        "GOOGL": {"fwd_pe": 22.0, "ttm_pe": 24.0, "peg": 1.5, "ev_ebitda": 16.0,
                  "sector": "Technology",              "industry": "Internet Content & Information","exchange": "NMS"},
        "AMZN":  {"fwd_pe": 40.0, "ttm_pe": 50.0, "peg": 2.8, "ev_ebitda": 18.0,
                  "sector": "Consumer Discretionary",  "industry": "Internet Retail",              "exchange": "NMS"},
        "HD":    {"fwd_pe": 22.0, "ttm_pe": 23.5, "peg": 2.0, "ev_ebitda": 15.0,
                  "sector": "Consumer Discretionary",  "industry": "Home Improvement Retail",      "exchange": "NYQ"},
        "JPM":   {"fwd_pe": 12.0, "ttm_pe": 12.5, "peg": 1.1, "ev_ebitda":  9.0,
                  "sector": "Financials",              "industry": "Banks—Diversified",            "exchange": "NYQ"},
        "BRK-B": {"fwd_pe": 21.0, "ttm_pe": 22.0, "peg": 1.8, "ev_ebitda": 11.0,
                  "sector": "Financials",              "industry": "Insurance—Diversified",        "exchange": "NYQ"},
        "JNJ":   {"fwd_pe": 15.0, "ttm_pe": 16.0, "peg": 2.2, "ev_ebitda": 13.0,
                  "sector": "Healthcare",              "industry": "Drug Manufacturers—General",   "exchange": "NYQ"},
        "PG":    {"fwd_pe": 24.0, "ttm_pe": 25.0, "peg": 3.0, "ev_ebitda": 18.0,
                  "sector": "Consumer Staples",        "industry": "Household & Personal Products","exchange": "NYQ"},
        "XOM":   {"fwd_pe": 13.0, "ttm_pe": 14.0, "peg": 1.3, "ev_ebitda":  7.0,
                  "sector": "Energy",                  "industry": "Oil & Gas Integrated",         "exchange": "NYQ"},
    },
    "earnings_dates": {
        "AAPL":  "2025-07-31", "MSFT":  "2025-07-23", "GOOGL": "2025-07-25",
        "AMZN":  "2025-08-01", "HD":    "2025-08-20", "JPM":   "2025-07-11",
        "BRK-B": "2025-08-02", "JNJ":   "2025-07-15", "PG":    "2025-07-29",
        "XOM":   "2025-08-01",
    },
    "cape": 36.4,
    "covariance_matrix": _cov,
}

result = agent3_advisor(test_state)

print("\n=== bl_views ===")
for ticker, view in result["bl_views"].items():
    print(f"  {ticker}: view={view.get('view_return'):.2%}, "
          f"conf={view.get('confidence')}, dir={view.get('sentiment_direction')}")
    print(f"    {view.get('reasoning')}")

print("\n=== recommended_weights ===")
for strategy, weights in result["recommended_weights"].items():
    if weights is None:
        print(f"  {strategy}: None (optimization failed)")
    else:
        print(f"  {strategy}:")
        for ticker, w in weights.items():
            print(f"    {ticker}: {w:.2%}")

print("\n=== errors ===")
for e in result.get("errors", []):
    print(f"  {e}")

# ── Strategy Validation ───────────────────────────────────────────────────────
print("\n=== Strategy Validation ===")

STRATEGIES = ["max_sharpe", "min_variance", "risk_parity", "target_return", "robust_mv"]
WEIGHT_SUM_TOL = 1e-3

recommended_weights = result["recommended_weights"]

assert "recommended_weights" in result, "FAIL: 'recommended_weights' key missing from result"
print("  PASS: 'recommended_weights' present")

for strategy in STRATEGIES:
    assert strategy in recommended_weights, f"FAIL: strategy '{strategy}' missing from recommended_weights"
print(f"  PASS: all 5 strategy keys present")

for strategy in STRATEGIES:
    weights = recommended_weights[strategy]
    if weights is None:
        print(f"  WARN: {strategy} returned None (optimization failed)")
        continue
    assert isinstance(weights, dict), f"FAIL: {strategy} weights is not a dict"
    for ticker in test_tickers:
        assert ticker in weights, f"FAIL: {strategy} missing ticker '{ticker}'"
    for ticker, w in weights.items():
        assert w >= -1e-6, f"FAIL: {strategy} negative weight for {ticker}: {w:.6f}"
    total = sum(weights.values())
    assert abs(total - 1.0) < WEIGHT_SUM_TOL, f"FAIL: {strategy} weights sum to {total:.6f}, expected ~1.0"
    print(f"  PASS: {strategy} — all tickers present, non-negative, sum={total:.6f}")

# ── Recommendation Table Validation ──────────────────────────────────────────
print("\n=== Recommendation Table Validation ===")

table = result["recommendation_table"]
assert "recommendation_table" in result, "FAIL: 'recommendation_table' key missing from result"
assert len(table) == len(test_tickers), f"FAIL: expected {len(test_tickers)} rows, got {len(table)}"
print(f"  PASS: table has {len(table)} rows")

for row in table:
    ticker = row.get("ticker")
    assert ticker in test_tickers, f"FAIL: unexpected ticker '{ticker}' in table"
    assert "current_weight" in row, f"FAIL: {ticker} missing 'current_weight'"
    assert row["consensus_action"] in ("BUY", "SELL", "HOLD"), \
        f"FAIL: {ticker} invalid consensus_action '{row['consensus_action']}'"
    for strategy in STRATEGIES:
        for suffix in ("_weight", "_delta", "_action"):
            col = f"{strategy}{suffix}"
            assert col in row, f"FAIL: {ticker} missing column '{col}'"
    print(f"  PASS: {ticker} — all strategy columns present, consensus={row['consensus_action']}")

print("\n=== All Tests Passed ===")