"""
Portfolio Pipeline — Main Entry Point
"""

import csv
import logging
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv
import yfinance as yf
import json
import traceback

from agents.graph 					import pipeline_graph
from database.schema				import init_database
from database.reconciler			import reconcile_virtual_portfolio, build_divergence_data
from database.persist    			import persist_to_database
from reports.report 				import generate_report
from reports.charts					import generate_all_charts
from reports.email 					import send_report_email, send_error_email
from scraper.wealthsimple_scraper 	import scrape_user_holdings

load_dotenv()

logger = logging.getLogger("investment_agent")


def _setup_logger(log_path: Path) -> None:
	logger.setLevel(logging.DEBUG)
	logger.handlers.clear()
	fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
	fh = logging.FileHandler(log_path, encoding="utf-8")
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(fmt)
	sh = logging.StreamHandler()
	sh.setLevel(logging.INFO)
	sh.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(sh)


def fetch_usd_cad_rate() -> float:
    """
    Fetches current USD/CAD exchange rate from yfinance.
    Returns how many CAD per 1 USD.
    Falls back to 1.36 if fetch fails.
    """
    try:
        rate = yf.Ticker("USDCAD=X").info.get("regularMarketPrice")
        if rate and 1.0 < rate < 2.0:   # sanity check
            return float(rate)
    except:
        pass
    logger.warning("USD/CAD fetch failed, using fallback rate of 1.36")
    return 1.36


def get_user_email(user_path: Path) -> str:
    config_path = user_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            return config["email"]
            


def load_portfolio(user_path: Path, usd_cad_rate: float) -> list[str]:
	tickers = []
	weights = {}
	total_value_cad = 0
	rows = []
	portfolio_rows = []

	with open(user_path / "portfolio.csv", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if not row.get("Symbol"):
				continue
			sym      = row["Symbol"].strip().upper()
			exchange = row.get("Exchange", "").strip()
			if exchange == "TSX":
				sym += ".TO"
			tickers.append(sym)
			market_value = float(row["Market Value"])
			currency = row.get("Market Value Currency", "CAD").strip().upper()
			# Convert to CAD
			if currency == "USD":
				market_value = market_value * usd_cad_rate

			market_price = float(row["Market Price"])
			price_currency = row.get("Market Price Currency", "CAD").strip().upper()
   
			if price_currency == "USD":
				market_price = market_price * usd_cad_rate


			rows.append((sym, market_value))
			total_value_cad += market_value

			portfolio_rows.append({
				"ticker":        sym,
				"shares":        float(row["Quantity"]),
				"avg_cost":      float(row["Book Value (CAD)"])/float(row["Quantity"]),
				"market_value":  market_value,
				"type":          row.get("Security Type", "EQUITY"),
				"currency":      "CAD",
				"market_price": market_price,
			})

	for sym, market_value in rows:
		weights[sym] = round(market_value / total_value_cad, 6)

	return tickers, weights, total_value_cad, portfolio_rows


def run_user(user_path: Path) -> None:
	user_name = user_path.name
	today = datetime.today().strftime("%Y-%m-%d")
	user_email = get_user_email(user_path)

	report_dir = user_path / "reports" / f"{today}"
	report_dir.mkdir(exist_ok=True)
	_setup_logger(report_dir / f"pipeline_{today}.log")
	logger.info(f"{'='*60}")
	logger.info(f"  Running pipeline for: {user_name}")
	logger.info(f"{'='*60}")
	output_path = report_dir / f"recommendation_{today}.txt"
	vp_output = report_dir / f"virtual_portfolio_{today}.txt"
	asset_analysis =  report_dir / "asset_analysis"
	asset_analysis.mkdir(exist_ok=True)

	try:
		# scrape wealthsimple for up to date holdings
		scrape_user_holdings(user_path)
	except Exception as e:
		logger.warning(f"Scraper failed for {user_name}: {e}")
		send_error_email(user_name, f"Scraper failed — using last CSV for {user_name}: {e}")
    	# pipeline continues with whatever portfolio.csv already exists
	
	try:
		# Initialise database — creates tables if they don't exist
		conn = init_database(user_path)

		usd_cad_rate = fetch_usd_cad_rate()

		tickers, weights, total_value, portfolio_rows = load_portfolio(user_path, usd_cad_rate)

		initial_state = {
			# Run metadata
			"user_name":  user_name,
			"user_path":  str(user_path.resolve()),
			"run_date":   today,
			"tickers":    tickers,
			"current_weights":  weights,
			"cad_usd_rate":		usd_cad_rate,
			"total_portfolio_value": total_value,
			"portfolio_rows": portfolio_rows,

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
			"covariance_matrix":	None,
			"cape":					None,
			"quant_commentary":  	None,
			"bl_views":             None,
			"posterior_mu":			None,
			"recommended_weights":  None,
			"recommendation_table": None,
			"advisory_commentary":  None,
			"mc_current":           None,
			"mc_port_current":		None,
			"mc_rebalanced":        None,
			"sim_commentary":       None,
			"errors":               [],
			"report_path":          None,
			"email_sent":           None,
		}

		final_state = pipeline_graph.invoke(initial_state)
		logger.info(f"Keys in final state: {list(final_state.keys())}")
		logger.info(f"Agent 1 summaries populated: {final_state['summaries'] is not None}")
		logger.info(f"Agent 3 commentary populated: {final_state['advisory_commentary'] is not None}")
		logger.info(f"Done. Errors: {final_state['errors']}")
	
		# Step 3 — persist
		db_errors = persist_to_database(final_state, conn)
		if db_errors:
			for e in db_errors:
				logger.error(f"[db error] {e}")

		# Step 4 — reconcile
		reconcile_result = None
		try:
			with open(vp_output, 'w', encoding='utf-8') as f:
				reconcile_result = reconcile_virtual_portfolio(conn, today, f)
		except Exception as e:
			logger.warning(f"Reconciler failed: {e}")

		# Step 5 — build divergence data for report
		try:
			divergence_data = build_divergence_data(conn, today, reconcile_result)
		except Exception as e:
			logger.warning(f"Divergence data build failed: {e}")
			divergence_data = None

		with open(output_path, 'w', encoding='utf-8') as f:
			strategies = ["max_sharpe", "min_variance", "risk_parity", "target_return", "robust_mv"]
			header = f"{'Ticker':<10} {'Current':>8} " + " ".join(f"{s[:10]:>10}" for s in strategies) + f"  {'Consensus':<10}"
			print(f"\n  Recommendation Table", file=f)
			print(f"  {header}", file=f)
			print(f"  {'-' * len(header)}", file=f)
			for row in final_state["recommendation_table"]:
				weights = " ".join(
					f"{row.get(f'{s}_weight', 0.0):>9.1%} " for s in strategies
				)
				print(f"  {row['ticker']:<10} {row['current_weight']:>8.1%} {weights} {row.get('consensus_action', 'N/A'):<10}", file=f)
	
		charts_dir  = user_path / "data" / f"{today}" / "raw" / "charts"
		generate_all_charts(final_state, charts_dir)

		pdf_path = generate_report(
			final_state     = final_state,
			report_dir      = report_dir,
			charts_dir      = charts_dir,
			divergence_data = divergence_data
		)
  
		send_report_email(final_state, pdf_path, user_email)
  
		conn.close()
	
	except Exception as e:
		tb = traceback.format_exc()
		logger.error(f"Pipeline failed for {user_name}: {e}")
		logger.error(tb)
		send_error_email(user_name, str(e), tb)
	finally:
		logger.handlers.clear()


def main():
    users_root = Path("users")
    user_dirs = [d for d in users_root.iterdir() if d.is_dir()]
    print(f"Found {len(user_dirs)} user(s): {[d.name for d in user_dirs]}")

    for user_path in user_dirs:
        try:
            run_user(user_path)
        except Exception as e:
            print(f"\n  [ERROR] Pipeline failed for {user_path.name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()