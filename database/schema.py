import sqlite3
from pathlib import Path

def init_database(user_path):
	"""
	Initialises the SQLite database and creates all tables if they don't exist.

	Args:
		user_path: Path to the user's directory where history.db will be created
	Returns:
		Active database connection
	"""
	db_path = user_path / "history.db"
	con = sqlite3.connect(db_path)
	cursor = con.cursor()
	cursor.execute("""
				CREATE TABLE IF NOT EXISTS portfolios (
					date			TEXT NOT NULL,
					ticker 			TEXT NOT NULL,
					quantity		REAL,
					avg_cost		REAL,
					asset_class 	TEXT,
					market_price	REAL,
					market_value 	REAL,
					weight 			REAL,
					PRIMARY KEY (date, ticker)
				)""")
	cursor.execute("""
				CREATE TABLE IF NOT EXISTS recommendations (
					date                TEXT NOT NULL,
					ticker              TEXT NOT NULL,
					strategy			TEXT NOT NULL,
					current_weight      REAL,
					recommended_weight  REAL,
					action              TEXT,
					mu_annual           REAL,
					sigma_annual        REAL,
					PRIMARY KEY (date, ticker, strategy)
				)""")
	cursor.execute("""
				CREATE TABLE IF NOT EXISTS model_outputs (
					date                TEXT NOT NULL,
					ticker              TEXT NOT NULL,
					
					alpha_daily         REAL,
					beta_mkt            REAL,
					beta_smb            REAL,
					beta_hml            REAL,
					r_squared           REAL,
					alpha_pval          REAL,
					
					garch_omega         REAL,
					garch_alpha         REAL,
					garch_beta          REAL,
					garch_persistence   REAL,
					garch_longrun_vol   REAL,
					garch_current_vol   REAL,
					
					mu_annual           REAL,
					sigma_annual        REAL,
					
					estimated_price     REAL,
					prob_up				REAL,
					mc_p25              REAL,
					mc_p50              REAL,
					mc_p75              REAL,
					mc_p95				REAL,
					mc_var95            REAL,
					mc_cvar_95          REAL,
					
					forward_pe          REAL,
					ttm_pe              REAL,
					peg_ratio           REAL,
					ev_ebitda           REAL,
					target_price		REAL,
					analyst_rec			TEXT,
					ma_200				REAL,
					ma_50				REAL,
					earnings_growth		REAL,
					revenue_growth		REAL,
					sector				TEXT,
					industry			TEXT,
					
					cape                REAL,
					PRIMARY KEY (date, ticker)
				)""")
	cursor.execute("""
				CREATE TABLE IF NOT EXISTS virtual_portfolio (
					date            TEXT NOT NULL,
					ticker          TEXT NOT NULL,
					strategy		TEXT NOT NULL,
					weight          REAL,
					price           REAL,
					quantity        REAL,
					market_value    REAL,
					total_value     REAL,
					divergence      REAL,
					PRIMARY KEY (date, ticker, strategy)
				)""")
	
	cursor.execute("""
		CREATE TABLE IF NOT EXISTS bl_views (
			date                TEXT NOT NULL,
			ticker              TEXT NOT NULL,
			view_return         REAL,
			confidence          INTEGER,
			sentiment_direction TEXT,
			valuation_signal    TEXT,
			conflict            INTEGER,
			reasoning           TEXT,
			posterior_mu        REAL,
			PRIMARY KEY (date, ticker)
		)""")

	cursor.execute("""
		CREATE TABLE IF NOT EXISTS portfolio_simulations (
			date			TEXT NOT NULL, 
   			strategy		TEXT NOT NULL, 
	  		p25				REAL, 
			p50				REAL, 
		 	p75				REAL,
			p95				REAL,
			var_95			REAL, 
   			cvar_95			REAL, 
	  		prob_up			REAL, 
			expected_value  REAL,
			PRIMARY KEY (date, strategy)
		)""")

	con.commit()
	return con

def insert_portfolio_row(conn, date, ticker, quantity, avg_cost, asset_class,
							market_price, market_value, weight):
	"""
	Inserts or replaces a single row in the portfolios table.

	Args:
		conn: Active database connection
		date: Date of the snapshot (YYYY-MM-DD string)
		ticker: Asset ticker symbol
		quantity: Number of shares held
		avg_cost: Average cost per share
		asset_class: Asset class label (e.g. 'Equity', 'Fixed Income')
		market_price: Latest market price
		market_value: Total market value (quantity * market_price)
		weight: Portfolio weight (0–1)
	"""
	# OR REPLACE IS FOR TESTING
	conn.execute("""
		INSERT OR REPLACE INTO portfolios						
			(date, ticker, quantity, avg_cost, asset_class,
			market_price, market_value, weight)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		""", (date, ticker, quantity, avg_cost, asset_class,
				market_price, market_value, weight))
	
	conn.commit()
	
def insert_model_output(conn, date, ticker, params: dict):
	"""
	Inserts or replaces a full model output row for a ticker.

	Args:
		conn: Active database connection
		date: Date of the run (YYYY-MM-DD string)
		ticker: Asset ticker symbol
		params: Dict of model outputs keyed by column name (factor model,
				GARCH, Monte Carlo, and valuation fields)
	"""
	# OR REPLACE IS FOR TESTING
	conn.execute("""
		INSERT OR REPLACE INTO model_outputs (date, ticker, alpha_daily, beta_mkt, beta_smb,
			beta_hml, r_squared, alpha_pval, garch_omega, garch_alpha, garch_beta,
			garch_persistence, garch_longrun_vol, garch_current_vol, mu_annual,
			sigma_annual, estimated_price, prob_up, mc_p25, mc_p50, mc_p75, mc_p95, mc_var95, mc_cvar_95,
			forward_pe, ttm_pe, peg_ratio, ev_ebitda, target_price,	analyst_rec, ma_200,
			ma_50, earnings_growth, revenue_growth, sector, industry, cape)
		VALUES (:date, :ticker, :alpha_daily, :beta_mkt, :beta_smb,
			:beta_hml, :r_squared, :alpha_pval, :garch_omega, :garch_alpha, :garch_beta,
			:garch_persistence, :garch_longrun_vol, :garch_current_vol, :mu_annual,
			:sigma_annual, :estimated_price, :prob_up, :mc_p25, :mc_p50, :mc_p75, :mc_p95, :mc_var95, :mc_cvar_95,
			:forward_pe, :ttm_pe, :peg_ratio, :ev_ebitda, :target_price,	:analyst_rec, :ma_200,
			:ma_50, :earnings_growth, :revenue_growth, :sector, :industry, :cape)
	""", {"date": date, "ticker": ticker, **params})
	
	conn.commit()
	
def insert_recommendation(conn, date, ticker, strategy, current_w,
						   recommended_w, action, mu, sigma):
	"""
	Inserts or replaces a rebalancing recommendation for a ticker and strategy.

	Args:
		conn: Active database connection
		date: Date of the recommendation (YYYY-MM-DD string)
		ticker: Asset ticker symbol
		strategy: Optimisation strategy name (e.g. 'Max Sharpe', 'Min Vol')
		current_w: Current portfolio weight (0–1)
		recommended_w: Target weight from optimiser (0–1)
		action: Signal string ('BUY', 'SELL', or 'HOLD')
		mu: Annualised expected return
		sigma: Annualised volatility
	"""
	# OR REPLACE IS FOR TESTING
	conn.execute("""
		INSERT OR REPLACE INTO recommendations
			(date, ticker, strategy, current_weight, recommended_weight,
			 action, mu_annual, sigma_annual)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	""", (date, ticker, strategy, current_w, recommended_w, action, mu, sigma))
	conn.commit()
	
def insert_virtual_portfolio(conn, date, positions, rp_value=0.0):
	"""
	Write the virtual portfolio positions to the database.

	Args:
		conn: Connection to database
		date: The date of the write
		positions: dictionary of portfolio positions
		rp_value: Real portfolio total value on this date (used to compute divergence)
	"""
	if hasattr(date, "strftime"):
		date = date.strftime("%Y-%m-%d")
	for strategy, tickers in positions.items():
		total_value = sum(p["market_value"] for p in tickers.values())
		divergence = total_value - rp_value
		for ticker, p in tickers.items():
			conn.execute("""
				INSERT OR REPLACE INTO virtual_portfolio
					(date, ticker, strategy, weight, price, quantity, market_value, total_value, divergence)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			""", (date, ticker, strategy, p["weight"], p["price"], p["shares"], p["market_value"], total_value, divergence))
	conn.commit()