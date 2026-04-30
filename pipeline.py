"""
Portfolio Pipeline — Main Entry Point
"""

import csv
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

from agents.graph import pipeline_graph

load_dotenv()


def load_tickers(user_path: Path) -> list[str]:
	tickers = []
	with open(user_path / "portfolio.csv", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			sym      = row["Symbol"].strip().upper()
			exchange = row.get("Exchange", "").strip()
			if exchange == "TSX":
				sym += ".TO"
			tickers.append(sym)
	return tickers


def run_user(user_path: Path) -> None:
	user_name = user_path.name
	print(f"\n{'='*60}")
	print(f"  Running pipeline for: {user_name}")
	print(f"{'='*60}")

	tickers = load_tickers(user_path)

	initial_state = {
		# Run metadata
		"user_name":  user_name,
		"user_path":  str(user_path.resolve()),
		"run_date":   date.today().isoformat(),
		"tickers":    tickers,

		# All agent outputs start as None
		"raw_text":             None,
		"summaries":            None,
		"aaii_sentiment":       None,
		"fear_greed":           None,
		"factor_results":    	None,
		"garch_results":     	None,
		"mu_sigma":          	None,
		"valuation":         	None,
		"financial_health":  	None,
		"earnings_data":     	None,
		"earnings_dates":    	None,
		"quant_commentary":  	None,
		"mc_current":           None,
		"mc_rebalanced":        None,
		"risk_commentary":      None,
		"bl_views":             None,
		"recommended_weights":  None,
		"recommendation_table": None,
		"advisory_commentary":  None,
		"errors":               [],
		"report_path":          None,
		"email_sent":           None,
	}

	final_state = pipeline_graph.invoke(initial_state)
	print(f"\n  Keys in final state: {list(final_state.keys())}")
	print(f"  Agent 1 summaries populated: {final_state['summaries'] is not None}")
	print(f"  Agent 4 commentary populated: {final_state['advisory_commentary'] is not None}")

    # Print results
	print(f"\n  Done. Errors: {final_state['errors']}")
	print(f"\n  Recommendation table:")
	for row in final_state["recommendation_table"]:
		print(f"    {row['ticker']:<8} → {row['recommended_weight']:.2%}  ({row['action']})")


def main():
    users_root = Path("users")
    user_dirs = [d for d in users_root.iterdir() if d.is_dir()]
    print(f"Found {len(user_dirs)} user(s): {[d.name for d in user_dirs]}")

    for user_path in user_dirs:
        try:
            run_user(user_path)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] Pipeline failed for {user_path.name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()