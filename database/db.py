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
					quantity		REAL,
					avg_cost		REAL,
					asset_class 	TEXT,
					current_price	REAL,
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
					weight          REAL,
					price           REAL,
					quantity          REAL,
					market_value    REAL,
					total_value     REAL,
					PRIMARY KEY (date, ticker)
				)""")
    
    con.commit()
    return con

def insert_portfolio_row(conn, date, ticker, quantity, avg_cost, asset_class,
                        	current_price, market_value, weight):
    # OR REPLACE IS FOR TESTING
    conn.execute("""
                	INSERT OR REPLACE INTO portfolios						
						(date, ticker, quantity, avg_cost, asset_class,
						current_price, market_value, weight)
                   	VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (date, ticker, quantity, avg_cost, asset_class,
                          current_price, market_value, weight))
    
    conn.commit()
    
def insert_model_output(conn, date, ticker, params: dict):
    # OR REPLACE IS FOR TESTING
    conn.execute("""
			INSERT OR REPLACE INTO model_outputs (date, ticker, alpha_daily, beta_mkt, beta_smb,
				beta_hml, r_squared, alpha_pval, garch_omega, garch_alpha, garch_beta,
				garch_persistence, garch_longrun_vol, garch_current_vol, mu_annual,
				sigma_annual, mc_p05, mc_p25, mc_p50, mc_p75, mc_p95, mc_var95, mc_cvar95,
				forward_pe, ttm_pe, peg_ratio, ev_ebitda, target_price,	analyst_rec, ma_200,
				ma_50, earnings_growth, revenue_growth, sector, industry, cape)
			VALUES (:date, :ticker, :alpha_daily, :beta_mkt, :beta_smb,
				:beta_hml, :r_squared, :alpha_pval, :garch_omega, :garch_alpha, :garch_beta,
				:garch_persistence, :garch_longrun_vol, :garch_current_vol, :mu_annual,
				:sigma_annual, :mc_p05, :mc_p25, :mc_p50, :mc_p75, :mc_p95, :mc_var95, :mc_cvar95,
				:forward_pe, :ttm_pe, :peg_ratio, :ev_ebitda, :target_price,	:analyst_rec, :ma_200,
				:ma_50, :earnings_growth, :revenue_growth, :sector, :industry, :cape)
		""", {"date": date, "ticker": ticker, **params})
    conn.commit()
    
def insert_recommendation(conn, date, ticker, strategy, current_w,
                           recommended_w, action, mu, sigma):
    # OR REPLACE IS FOR TESTING
    conn.execute("""
        INSERT OR REPLACE INTO recommendations
            (date, ticker, strategy, current_weight, recommended_weight,
             action, mu_annual, sigma_annual)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (date, ticker, strategy, current_w, recommended_w, action, mu, sigma))
    conn.commit()