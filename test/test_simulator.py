import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent3_simulator import (
    run_monte_carlo,
    port_monte_carlo,
    agent3_simulator
)

# Test 1 - Per-Ticker MC
# print("Testing per-ticker MC...")
# results = run_monte_carlo(0.2, 1, 100)
# print(f"{results}")

# Test 2 - Portfolio MC
# print("Testing portfolio MC...")
# results = port_monte_carlo(np.array([0.1, 0.2]), np.array([[0.2, 0.12], [0.12, 0.1]]), 100, [0.6, 0.4])
# print(f"{results}")

# Minimal state with only what Agent 3 needs
test_state = {
    "user_name":  "test_user",
    "user_path":  "users/test_user",
    "run_date":   "2025-04-29",
    "tickers":    ["AAPL", "LMT", "JNJ"],
    "current_weights":  {"AAPL": 0.50, "LMT": 0.30, "JNJ": 0.20},
    "cad_usd_rate":		1.35,
    "total_portfolio_value": 10000,
    "strategies":		["equal_weight"],
    "errors":     [],

    # Agent 1 and 2 outputs — Agent 3 doesn't read these
    # but they need to exist in state as None
    "raw_text":         None,
    "summaries":        None,
    "aaii_sentiment":   None,
    "fear_greed":       None,
    "factor_results":   None,
    "garch_results":    None,
    "mu_sigma": {
        "AAPL": {"mu_annual": 0.12,  "sigma_annual": 0.25, "s_current": 175.0},
        "LMT":  {"mu_annual": 0.09,  "sigma_annual": 0.20, "s_current": 450.0},
        "JNJ":  {"mu_annual": 0.07,  "sigma_annual": 0.15, "s_current": 155.0},
    },
    "valuation":        None,
    "financial_health": None,
    "earnings_data":    None,
    "earnings_dates":   None,
    "covariance_matrix": { 
        "AAPL": {"AAPL": 2.480e-4, "LMT": 5.96e-5, "JNJ": 2.98e-5},
        "LMT":  {"AAPL": 5.96e-5,  "LMT": 1.587e-4,"JNJ": 2.98e-5},
        "JNJ":  {"AAPL": 2.98e-5,  "LMT": 2.98e-5, "JNJ": 8.93e-5},
    },
    "quant_commentary": None,
    
    # Agent 3 outputs — all None at start
    "mc_current":           None,
    "mc_rebalanced":        None,
    "risk_commentary":      None,
}

result = agent3_simulator(test_state)

# Inspect outputs
print("\n=== mc_current ===")
print(result["mc_current"])

print("\n=== mc_rebalanced ===")
print(result["mc_current"])

print("\n=== errors ===")
for e in result["errors"]:
    print(f"  {e}")