import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent2_quant import (
    fetch_financial_health,
    agent2_quant
)

# Test 1 — Financial Health
# print("Testing Financial Health...")
# results = fetch_financial_health("AAPL")
# print(f"{results}")

# Minimal state with only what Agent 2 needs
test_state = {
    "user_name":  "test_user",
    "user_path":  "users/test_user",
    "run_date":   "2025-04-29",
    "tickers":    ["AAPL", "MSFT"],
    "errors":     [],

    # Agent 1 outputs — Agent 2 doesn't read these
    # but they need to exist in state as None
    "raw_text":        None,
    "summaries":       None,
    "aaii_sentiment":  None,
    "fear_greed":      None,

    # Agent 2 outputs — all None at start
    "factor_results":   None,
    "garch_results":    None,
    "mu_sigma":         None,
    "valuation":        None,
    "financial_health": None,
    "earnings_data":    None,
    "earnings_dates":   None,
    "quant_commentary": None,
}

result = agent2_quant(test_state)

# Inspect outputs
print("\n=== mu_sigma ===")
for ticker, vals in result["mu_sigma"].items():
    print(f"  {ticker}: mu={vals['mu_annual']:.2%}, "
          f"sigma={vals['sigma_annual']:.2%}, "
          f"fallback={vals['is_fallback']}")

print("\n=== valuation ===")
for ticker, vals in result["valuation"].items():
    print(f"  {ticker}: fwd_pe={vals.get('fwd_pe')}, peg={vals.get('peg')}")

print("\n=== errors ===")
for e in result["errors"]:
    print(f"  {e}")

print("\n=== commentary ===")
print(result["quant_commentary"])