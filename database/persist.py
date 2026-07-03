"""
database/persist.py
====================
Maps the final LangGraph pipeline state to SQLite database writes.
Called once per user after pipeline_graph.invoke() completes.
"""

import json
from database.schema import (
    insert_portfolio_row,
    insert_model_output,
    insert_recommendation,
)


def persist_to_database(final_state: dict, conn) -> list[str]:
    """
    Writes all pipeline outputs to SQLite.
    Returns a list of any errors encountered during writes.
    Non-fatal — a failed write logs an error but does not crash the pipeline.
    
    Args:
        final_state: Complete state dict returned by pipeline_graph.invoke()
        conn: Active database connection from init_database()
    
    Returns:
        List of error strings encountered during writes
    """
    errors = []
    date   = final_state["run_date"]

    # ── 1. portfolios table ───────────────────────────────────────
    # Read directly from portfolio.csv snapshot already in state
    # current_weights computed in pipeline.py from market values
    try:
        _write_portfolios(final_state, conn, date)
    except Exception as e:
        errors.append(f"portfolios write failed: {e}")

    # ── 2. model_outputs table ────────────────────────────────────
    try:
        _write_model_outputs(final_state, conn, date)
    except Exception as e:
        errors.append(f"model_outputs write failed: {e}")

    # ── 3. recommendations table ──────────────────────────────────
    try:
        _write_recommendations(final_state, conn, date)
    except Exception as e:
        errors.append(f"recommendations write failed: {e}")

    # ── 4. bl_views table ─────────────────────────────────────────
    try:
        _write_bl_views(final_state, conn, date)
    except Exception as e:
        errors.append(f"bl_views write failed: {e}")

    # ── 5. portfolio_simulations table ────────────────────────────
    try:
        _write_simulations(final_state, conn, date)
    except Exception as e:
        errors.append(f"portfolio_simulations write failed: {e}")

    conn.commit()
    return errors


def _write_portfolios(state, conn, date):
    """
    Writes current portfolio snapshot from portfolio_rows stored in state.
    portfolio_rows is populated in pipeline.py when the CSV is read.
    """
    for row in state.get("portfolio_rows") or []:
        insert_portfolio_row(
            conn         = conn,
            date         = date,
            ticker       = row["ticker"],
            quantity     = row["shares"],
            avg_cost     = row["avg_cost"],
            asset_class  = row["type"],
            market_price = row.get("market_price"),
            market_value = row["market_value"],
            weight       = state["current_weights"].get(row["ticker"], 0.0),
        )


def _write_model_outputs(state, conn, date):
    """
    Writes factor model, GARCH, valuation, and per-ticker MC outputs.
    Maps state field names to database column names.
    """
    tickers = state["tickers"]
    cape    = state.get("cape") or 25.0

    for ticker in tickers:
        fr  = state.get("factor_results",   {}).get(ticker, {})
        gr  = state.get("garch_results",    {}).get(ticker, {})
        val = state.get("valuation",        {}).get(ticker, {})
        mc  = state.get("mc_current",       {}).get(ticker, {})
        ms  = state.get("mu_sigma",         {}).get(ticker, {})
        fh  = state.get("financial_health", {}).get(ticker, {})
        ed  = state.get("earnings_data",    {}).get(ticker, {})

        params = {
            # Factor model
            "alpha_daily":       fr.get("alpha_daily"),
            "beta_mkt":          fr.get("b_MKT"),
            "beta_smb":          fr.get("b_SMB"),
            "beta_hml":          fr.get("b_HML"),
            "r_squared":         fr.get("r2"),
            "alpha_pval":        fr.get("p_val_alpha"),

            # GARCH
            "garch_omega":       gr.get("omega_garch"),
            "garch_alpha":       gr.get("alpha_garch"),
            "garch_beta":        gr.get("beta_garch"),
            "garch_persistence": gr.get("garch_persist"),
            "garch_longrun_vol": gr.get("garch_long_run_vol"),
            "garch_current_vol": gr.get("sigma_current_vol"),

            # mu and sigma
            "mu_annual":         ms.get("mu_annual"),
            "sigma_annual":      ms.get("sigma_annual"),

            # Per-ticker Monte Carlo
            "estimated_price":   mc.get("E_ST"),
            "prob_up":			 mc.get("prob_up"),
            "mc_p25":            mc.get("pct")[2],
            "mc_p50":            mc.get("pct")[3],
            "mc_p75":            mc.get("pct")[4],
            "mc_p95":			 mc.get("pct")[5],	
            "mc_var95":          mc.get("VaR_95"),
            "mc_cvar_95":          mc.get("CVaR_95"),

            # Valuation
            "forward_pe":        val.get("fwd_pe"),
            "ttm_pe":            val.get("ttm_pe"),
            "peg_ratio":         val.get("peg"),
            "ev_ebitda":         val.get("ev_ebitda"),
            "target_price":      val.get("target_price"),
            "analyst_rec":       val.get("analyst_rec"),
            "ma_200":            val.get("200MA"),
            "ma_50":             val.get("50MA"),
            "sector":            val.get("sector"),
            "industry":          val.get("industry"),
			"earnings_growth":   val.get("earnings_growth"),
			"revenue_growth":    val.get("revenue_growth"),

            # Macro
            "cape":              cape,
        }

        insert_model_output(conn, date, ticker, params)


def _write_recommendations(state, conn, date):
    """
    Writes one row per ticker per strategy from the recommendation table.
    """
    rec_table    = state.get("recommendation_table") or []
    posterior_mu = state.get("posterior_mu") or {}
    ms           = state.get("mu_sigma") or {}

    for row in rec_table:
        ticker = row["ticker"]
        mu     = posterior_mu.get(ticker)
        sigma  = ms.get(ticker, {}).get("sigma_annual")

        # Write one row per strategy
        for strategy in (state.get("recommended_weights") or {}).keys():
            insert_recommendation(
                conn          = conn,
                date          = date,
                ticker        = ticker,
                strategy      = strategy,
                current_w     = row["current_weight"],
                recommended_w = row.get(f"{strategy}_weight", 0.0),
                action        = row.get(f"{strategy}_action", "HOLD"),
                mu            = mu,
                sigma         = sigma,
            )


def _write_bl_views(state, conn, date):
    """
    Writes Black-Litterman views and posterior mu per ticker.
    """
    bl_views     = state.get("bl_views") or {}
    posterior_mu = state.get("posterior_mu") or {}

    for ticker, view in bl_views.items():
        conn.execute("""
            INSERT OR REPLACE INTO bl_views
                (date, ticker, view_return, confidence, sentiment_direction,
                 valuation_signal, conflict, reasoning, posterior_mu)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            ticker,
            view.get("view_return"),
            view.get("confidence"),
            view.get("sentiment_direction"),
            view.get("valuation_signal"),
            1 if view.get("conflict") else 0,
            view.get("reasoning"),
            posterior_mu.get(ticker),
        ))


def _write_simulations(state, conn, date):
    """
    Writes portfolio-level simulation results.
    Current portfolio uses strategy='current', is_current=1.
    Each strategy simulation uses strategy name, is_current=0.
    """
    # Current portfolio simulation
    curr = state.get("mc_port_current") or {}
    if curr:
        conn.execute("""
            INSERT OR REPLACE INTO portfolio_simulations
                (date, strategy, p25, p50, p75, p95,
                 var_95, cvar_95, prob_up, expected_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, "current",
            curr.get("pct")[2],  curr.get("pct")[3], curr.get("pct")[4],
			curr.get("pct")[5],  curr.get("VaR_95"), curr.get("CVaR_95"),
			curr.get("prob_up"), curr.get("E_ST")
        ))

    # Per-strategy recommended portfolio simulations
    for strategy, sim in (state.get("mc_port_rebalanced") or {}).items():
        if sim:
            conn.execute("""
                INSERT OR REPLACE INTO portfolio_simulations
                    (date, strategy, p25, p50, p75, p95,
                     var_95, cvar_95, prob_up, expected_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date, strategy,
                sim.get("pct")[2],  sim.get("pct")[3], sim.get("pct")[4],
                sim.get("pct")[5],  sim.get("VaR_95"), sim.get("CVaR_95"),
                sim.get("prob_up"), sim.get("E_ST")
            ))