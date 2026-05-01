import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent3_advisor import (
    agent3_advisor
)

# Test 3 — Advisor Agent LLM Call
print("\nTesting Advisor Agent LLM Call...")

test_tickers = ["AAPL", "MSFT"]

test_state = {
    "user_name": "test_user",
    "user_path": "users/test_user",
    "run_date":  "2025-05-01",
    "tickers":   test_tickers,
    "errors":    [],

    # Agent 1 outputs consumed by Agent 3
    "summaries": {
        "AAPL": "Apple reported strong iPhone 16 demand and better-than-expected services revenue. Management guided conservatively but beat on EPS.",
        "MSFT": "Microsoft Azure grew 31% YoY. AI Copilot adoption is accelerating. Cloud margins improved sequentially.",
    },
    "aaii_sentiment": {"bullish": 38.5, "bearish": 29.2, "neutral": 32.3},
    "fear_greed":     {"score": 62, "rating": "Greed"},

    # Agent 2 outputs consumed by Agent 3
    "mu_sigma": {
        "AAPL": {"mu_annual": 0.12, "sigma_annual": 0.22, "s_current": 185.5, "is_fallback": False},
        "MSFT": {"mu_annual": 0.14, "sigma_annual": 0.20, "s_current": 415.0, "is_fallback": False},
    },
    "valuation": {
        "AAPL": {"fwd_pe": 28.5, "ttm_pe": 30.1, "peg": 2.1, "ev_ebitda": 22.0,
                 "sector": "Technology", "industry": "Consumer Electronics", "exchange": "NMS"},
        "MSFT": {"fwd_pe": 32.0, "ttm_pe": 35.2, "peg": 2.3, "ev_ebitda": 24.0,
                 "sector": "Technology", "industry": "Software—Infrastructure", "exchange": "NMS"},
    },
    "earnings_dates": {
        "AAPL": "2025-07-31",
        "MSFT": "2025-07-23",
    },
    "cape": 36.4,
}

result = agent3_advisor(test_state)

print("\n=== bl_views ===")
for ticker, view in result["bl_views"].items():
    print(f"  {ticker}: view={view.get('view_return'):.2%}, "
          f"conf={view.get('confidence')}, dir={view.get('sentiment_direction')}")
    print(f"    {view.get('reasoning')}")

print("\n=== recommended_weights ===")
for ticker, w in result["recommended_weights"].items():
    print(f"  {ticker}: {w:.2%}")

print("\n=== errors ===")
for e in result.get("errors", []):
    print(f"  {e}")