import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent2_quant import (
    fetch_financial_health,
    agent2_quant,
    fetch_real_rf
)

# Test 1 — Financial Health
import json

raw = response.content.strip()
# Strip markdown fences if model ignored instructions
raw = raw.replace("```json", "").replace("```", "").strip()

try:
    views = json.loads(raw)
except json.JSONDecodeError:
    # retry logic or full fallback
    views = {}

# Validate and fill missing tickers
for ticker in tickers:
    if ticker not in views:
        views[ticker] = {
            "view_return": 0.0,
            "confidence": 1,
            "sentiment_direction": "neutral",
            "valuation_signal": "neutral",
            "conflict": False,
            "reasoning": "No view generated — using neutral fallback."
        }
    else:
        # Clip view_return to valid range
        views[ticker]["view_return"] = max(-0.30, min(0.30, views[ticker]["view_return"]))
        # Clip confidence to 1-5
        views[ticker]["confidence"] = max(1, min(5, int(views[ticker]["confidence"])))

bl_views = views

# Test 2 - Real RF Fetch
print("Testing Real RF Fetch...")
results = fetch_real_rf()
print(f"{results}")

# Minimal state with only what Agent 2 needs
# test_state = {
#     "user_name":  "test_user",
#     "user_path":  "users/test_user",
#     "run_date":   "2025-04-29",
#     "tickers":    ["AAPL", "MSFT"],
#     "errors":     [],

#     # Agent 1 outputs — Agent 2 doesn't read these
#     # but they need to exist in state as None
#     "raw_text":        None,
#     "summaries":       None,
#     "aaii_sentiment":  None,
#     "fear_greed":      None,

#     # Agent 2 outputs — all None at start
#     "factor_results":   None,
#     "garch_results":    None,
#     "mu_sigma":         None,
#     "valuation":        None,
#     "financial_health": None,
#     "earnings_data":    None,
#     "earnings_dates":   None,
#     "quant_commentary": None,
# }

# result = agent2_quant(test_state)

# # Inspect outputs
# print("\n=== mu_sigma ===")
# for ticker, vals in result["mu_sigma"].items():
#     print(f"  {ticker}: mu={vals['mu_annual']:.2%}, "
#           f"sigma={vals['sigma_annual']:.2%}, "
#           f"fallback={vals['is_fallback']}")

# print("\n=== valuation ===")
# for ticker, vals in result["valuation"].items():
#     print(f"  {ticker}: fwd_pe={vals.get('fwd_pe')}, peg={vals.get('peg')}")

# print("\n=== errors ===")
# for e in result["errors"]:
#     print(f"  {e}")

# print("\n=== commentary ===")
# print(result["quant_commentary"])