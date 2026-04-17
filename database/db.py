import sqlite3
from pathlib import Path

def init_database(user_path):
    db_path = user_path / "history.db"
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    cursor.execute("""
				CREATE TABLE IF NOT EXISTS portfolios (
					date			TEXT NOT NULL,
					ticker 			TEXT NOT NULL,
					shares			REAL,
					avg_cost		REAL,
					asset_class 	TEXT,
					current_price	REAL,
					market_value 	REAL
					weight 			REAL,
					PRIMARY KEY (date, ticker)
				)""")
    cursor.execute("""
				CREATE TABLE IF NOT EXISTS recommendations (
					date                TEXT NOT NULL,
					ticker              TEXT NOT NULL,
					current_weight      REAL,
					recommended_weight  REAL,
					action              TEXT,
					mu_annual           REAL,
					sigma_annual        REAL,
					PRIMARY KEY (date, ticker)
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
					
					mc_p05              REAL,
					mc_p25              REAL,
					mc_p50              REAL,
					mc_p75              REAL,
					mc_p95              REAL,
					mc_var95            REAL,
					mc_cvar95           REAL,
					
					forward_pe          REAL,
					ttm_pe              REAL,
					peg_ratio           REAL,
					ev_ebitda           REAL,
     				target_price		REAL
					analyst_rec			TEXT,
					200MA				REAL,
					50MA				REAL,
					earnings_growth		REAL,
					revenue_growth		REAL,
					
					cape                REAL,
					PRIMARY KEY (date, ticker)
				)""")
    cursor.execute("""
				CREATE TABLE IF NOT EXISTS virtual_portfolio (
					date            TEXT NOT NULL,
					ticker          TEXT NOT NULL,
					weight          REAL,
					price           REAL,
					shares          REAL,
					market_value    REAL,
					total_value     REAL,
					PRIMARY KEY (date, ticker)
				)""")
    
    con.commit()
    return con