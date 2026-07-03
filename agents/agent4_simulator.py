import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

from agents.state import PipelineState
from agents.llm import get_llm

logger = logging.getLogger("investment_agent")


def get_mu_for_tickers(tickers: list, state: PipelineState) -> list:
    """
    Uses BL posterior mu if available, falls back to factor model mu.
    Sigma always comes from GARCH via mu_sigma regardless.
    
    Args:
        tickers: list of tickers in portfolio
        state: pipeline state object
        
    Returns:
        list of returns for each ticker
    """
    returns = []
    posterior = state.get("posterior_mu") or {}

    for ticker in tickers:
        if ticker in posterior:
            returns.append(posterior[ticker])
        else:
            returns.append(state["mu_sigma"][ticker]["mu_annual"])

    return np.array(returns)


def run_monte_carlo(mu: float, sigma: float, S0: float, file=None):
    """
    Simulates 10,000 GBM paths with Student-t innovations over 252 trading days.

    Args:
        mu: Annualised expected return
        sigma: Annualised total volatility
        S0: Current (last historical) price

    Returns:
        dict with paths, S_T, E_ST, prob_up, VaR_95, CVaR_95, pct,
              N_paths, N_steps, S_current, df_t, t_scale
    """
    N_paths = 10_000
    N_steps = 252
    S_current = S0
    df_t = 6
    t_scale = np.sqrt((df_t - 2) / df_t)
    
    # mu    = np.clip(mu,    -2.0,  2.0)   # cap at ±200% annual return
    # sigma = np.clip(sigma,  0.01, 2.0)   # cap at 1%–200% annual vol

    print(f"\n[Monte Carlo Simulation]", file=file)
    print(f"  Paths      : {N_paths:,}", file=file)
    print(f"  Steps      : {N_steps} days", file=file)
    print(f"  mu         : {mu:.2%} p.a.", file=file)
    print(f"  sigma      : {sigma:.2%} p.a.", file=file)
    print(f"  Innovation : Student-t (df={df_t})", file=file)
    print(f"  S_current  : {S_current:.2f}", file=file)

    dt_sim = 1 / 252
    drift = (mu - 0.5 * sigma ** 2) * dt_sim
    vol_step = sigma * np.sqrt(dt_sim)

    Z = stats.t.rvs(df=df_t, size=(N_steps, N_paths)) / t_scale
    shocks = drift + vol_step * Z

    cum_log = np.cumsum(shocks, axis=0)
    cum_log = np.vstack([np.zeros(N_paths), cum_log])
    paths = S_current * np.exp(cum_log)

    S_T = paths[-1, :]
    pct = np.percentile(S_T, [1, 5, 25, 50, 75, 95, 99])
    E_ST = S_T.mean()
    prob_up = np.mean(S_T > S_current)

    losses = S_current - S_T
    VaR_95 = np.percentile(losses, 95)
    CVaR_95 = losses[losses >= np.percentile(losses, 95, method='lower')].mean()

    print(f"\n  ── Terminal Price Distribution (S_T) ──", file=file)
    print(f"  Current Price       : {S_current:.2f}", file=file)
    print(f"  E[S_T]              : {E_ST:.2f}", file=file)
    print(f"  Median              : {pct[3]:.2f}", file=file)
    print(f"  Std dev             : {S_T.std():.2f}", file=file)
    print(f"  1st  percentile     : {pct[0]:.2f}", file=file)
    print(f"  5th  percentile     : {pct[1]:.2f}", file=file)
    print(f"  25th percentile     : {pct[2]:.2f}", file=file)
    print(f"  75th percentile     : {pct[4]:.2f}", file=file)
    print(f"  95th percentile     : {pct[5]:.2f}", file=file)
    print(f"  99th percentile     : {pct[6]:.2f}", file=file)
    print(f"\n  Prob(S_T > S_current): {prob_up:.2%}", file=file)
    print(f"  1-year 95% VaR      : {VaR_95:.2f}  ({VaR_95 / S_current:.2%} of price)", file=file)
    print(f"  1-year 95% CVaR     : {CVaR_95:.2f}  ({CVaR_95 / S_current:.2%} of price)", file=file)

    return {
        "sample_paths": paths[:, :100],
        "S_T": S_T,
        "E_ST": E_ST,
        "prob_up": prob_up,
        "VaR_95": S_current - pct[1],
        "CVaR_95": CVaR_95,
        "pct": pct,
        "N_paths": N_paths,
        "N_steps": N_steps,
        "S_current": S_current,
        "df_t": df_t,
        "t_scale": t_scale,
    }
    

def port_monte_carlo(mu: np.ndarray, covar_ann: np.ndarray, init_port_value: float, weights: list, file = None):
    """
    Performs a Monte Carlo simulation at portfolio level
    
    Args:
        mu: dict of each ticker estimated returns
        covar: Dataframe of the covariance matrix of tickers
        
    Returns:

    """
    N_paths = 10_000
    N_steps = 252
    n_assets = mu.shape[0]
    weights = np.array(weights)
    df_ch = 6
    ch_scale = np.sqrt((df_ch - 2) / df_ch)
    # Replace any NaN (e.g. from a ticker with no price history) with 0 before decomposition.
    # eigh produces NaN eigenvalues from NaN input, so nan_to_num must come first.
    covar_ann = np.nan_to_num(covar_ann, nan=0.0)
    # Clip negative eigenvalues — safeguard against floating-point noise from dict round-trip.
    eigvals, eigvecs = np.linalg.eigh(covar_ann)
    eigvals = np.maximum(eigvals, 1e-8)
    covar_ann = (eigvecs @ np.diag(eigvals) @ eigvecs.T)
    covar_ann = (covar_ann + covar_ann.T) / 2
    L = np.linalg.cholesky(covar_ann)
    dt_sim = 1 / 252
    

    print(f"\n[Monte Carlo Simulation]", file=file)
    print(f"  Paths      : {N_paths:,}", file=file)
    print(f"  Steps      : {N_steps} days", file=file)
    print(f"  Innovation : Chi-squared (df={df_ch})", file=file)
    
    # Generate random standard normals
    Z = np.random.standard_normal((N_steps, N_paths, n_assets))
    
    # scale by chi-squared to get fat tails
    chi = np.random.chisquare(df_ch, size=(N_steps, N_paths))
    shocks = Z / (np.sqrt(chi/df_ch) / ch_scale)[:, :, np.newaxis]
    
    # Generate correlated shocks
    corr_shocks = (shocks @ L.T) * np.sqrt(dt_sim)
    
    drift_vec = (mu - 0.5 * np.diag(covar_ann)) * dt_sim
    log_rets = drift_vec + corr_shocks
    port_log_rets = log_rets @ weights

    cum_log = np.cumsum(port_log_rets, axis=0)          # (N_steps, N_paths)
    cum_log = np.vstack([np.zeros((1, N_paths)), cum_log])  # prepend t=0 row
    paths = init_port_value * np.exp(cum_log)            # (N_steps+1, N_paths)


    S_T = paths[-1, :]
    pct = np.percentile(S_T, [1, 5, 25, 50, 75, 95, 99])
    E_ST = S_T.mean()
    prob_up = np.mean(S_T > init_port_value)

    losses = init_port_value - S_T
    VaR_95 = np.percentile(losses, 95)
    CVaR_95 = losses[losses >= np.percentile(losses, 95, method='lower')].mean()

    print(f"\n  ── Terminal Price Distribution (S_T) ──", file=file)
    print(f"  Current Price       : {init_port_value:.2f}", file=file)
    print(f"  E[S_T]              : {E_ST:.2f}", file=file)
    print(f"  Median              : {pct[3]:.2f}", file=file)
    print(f"  Std dev             : {S_T.std():.2f}", file=file)
    print(f"  1st  percentile     : {pct[0]:.2f}", file=file)
    print(f"  5th  percentile     : {pct[1]:.2f}", file=file)
    print(f"  25th percentile     : {pct[2]:.2f}", file=file)
    print(f"  75th percentile     : {pct[4]:.2f}", file=file)
    print(f"  95th percentile     : {pct[5]:.2f}", file=file)
    print(f"  99th percentile     : {pct[6]:.2f}", file=file)
    print(f"\n  Prob(S_T > S_current): {prob_up:.2%}", file=file)
    print(f"  1-year 95% VaR      : {VaR_95:.2f}  ({VaR_95 / init_port_value:.2%} of price)", file=file)
    print(f"  1-year 95% CVaR     : {CVaR_95:.2f}  ({CVaR_95 / init_port_value:.2%} of price)", file=file)

    return {
        "sample_paths": paths[:, :100],
        "S_T": S_T,
        "E_ST": E_ST,
        "prob_up": prob_up,
        "VaR_95": init_port_value - pct[1],
        "CVaR_95": CVaR_95,
        "pct": pct,
        "N_paths": N_paths,
        "N_steps": N_steps,
        "S_current": init_port_value,
        "df_ch": df_ch,
        "ch_scale": ch_scale,
    }


def _fmt(val: float | None, spec: str) -> str:
    return format(val, spec) if val is not None else "N/A"

# LLM SIM SUMMARY
SIMULATOR_PROMPT = """You are a quantitative analyst reviewing simulation outputs for a portfolio.

For each monte carlo simulation, done per ticker, the current portfolio and the rebalanced portfolios

Address these specific items:
- Tail risk in dollar terms.
- Which individual tickers drive the most risk and the most returns.
- The probability of gains from prob_up
- Fallback acknowledgement, if any tickers have fallback_present as true then the simulation was run on default assumptions
and should be treated as such. Dont't mention fallback unless its present.

Data:
{data}

For each ticker and portfolio write 2-3 sentences maximum summarising the simulation results.
Compare the upside and expected returns to the downside risk.
Compare the different portfolios to each other in terms of expected returns, probability of
gains and downside risk measures."""


def generate_sim_commentary(
        tickers: list[str],
        strategies: list[str],
        mc_tickers: dict,
        mc_curr_port: dict,
        mc_rebal_port: dict,
        fallback_present: dict,
        file = None
    ) -> str:
    """
    Calls the LLM to interpret simulation outputs per ticker/portfolio.

    Builds a structured simulation summary for each ticker and portfolio

    Args:
        tickers: List of ticker symbols to include in the commentary.
        mc_tickers: Dict of per-ticker sim results
        mc_curr_port: Dict of current protfolio sim results
        mc_rebal_port: Dict of rebalanced portfolio sim results
        fallback_present: Dict of whether ticker has real data or assumed data

    Returns:
        Plain prose commentary, one paragraph per ticker labelled with the
        ticker symbol. Returns a fallback string if the LLM call fails.
    """
    # Build a structured summary of all metrics per ticker
    data_lines = []
    for ticker in tickers:        
        fr  = mc_tickers.get(ticker, {})

        pct = fr.get('pct', [None]*7)
        data_lines.append(f"""{ticker}:
        E[S_T]={_fmt(fr.get('E_ST'), '.2f')}, median={_fmt(pct[3], '.2f')}, \
        S_current={_fmt(fr.get('S_current'), '.2f')}
        Prob(up)={_fmt(fr.get('prob_up'), '.2%')}
        VaR_95={_fmt(fr.get('VaR_95'), '.2f')}, CVaR_95={_fmt(fr.get('CVaR_95'), '.2f')}
        Percentiles: 1%={_fmt(pct[0], '.2f')}, 5%={_fmt(pct[1], '.2f')}, \
            25%={_fmt(pct[2], '.2f')}, 75%={_fmt(pct[4], '.2f')}, \
            95%={_fmt(pct[5], '.2f')}, 99%={_fmt(pct[6], '.2f')},
        Fallback: {fallback_present.get(ticker, True)}
        """)
    
    fr = mc_curr_port
    pct = fr.get('pct', [None]*7)
    data_lines.append(f"""Current Portfolio:
        E[S_T]={_fmt(fr.get('E_ST'), '.2f')}, median={_fmt(pct[3], '.2f')}, \
            S_current={_fmt(fr.get('S_current'), '.2f')}
        Prob(up)={_fmt(fr.get('prob_up'), '.2%')}
        VaR_95={_fmt(fr.get('VaR_95'), '.2f')}, CVaR_95={_fmt(fr.get('CVaR_95'), '.2f')}
        Percentiles: 1%={_fmt(pct[0], '.2f')}, 5%={_fmt(pct[1], '.2f')}, \
            25%={_fmt(pct[2], '.2f')}, 75%={_fmt(pct[4], '.2f')}, \
            95%={_fmt(pct[5], '.2f')}, 99%={_fmt(pct[6], '.2f')}
        """)
    
    for strat in strategies:
        if strat not in mc_rebal_port:
            continue
        fr = mc_rebal_port[strat]
        pct = fr.get('pct', [None]*7)
        data_lines.append(f"""{strat}:
        E[S_T]={_fmt(fr.get('E_ST'), '.2f')}, median={_fmt(pct[3], '.2f')}, \
            S_current={_fmt(fr.get('S_current'), '.2f')}
        Prob(up)={_fmt(fr.get('prob_up'), '.2%')}
        VaR_95={_fmt(fr.get('VaR_95'), '.2f')}, CVaR_95={_fmt(fr.get('CVaR_95'), '.2f')}
        Percentiles: 1%={_fmt(pct[0], '.2f')}, 5%={_fmt(pct[1], '.2f')}, \
            25%={_fmt(pct[2], '.2f')}, 75%={_fmt(pct[4], '.2f')}, \
            95%={_fmt(pct[5], '.2f')}, 99%={_fmt(pct[6], '.2f')}
        """)

    prompt = SIMULATOR_PROMPT.format(data = "\n".join(data_lines))

    try:
        llm      = get_llm()
        response = llm.invoke(prompt)
        print(f"\n LLM Commentary:{response.content.strip()}", file=file)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"LLM sim commentary failed: {e}")
        return "Simulation commentary unavailable this run."
    
    
def build_advisory_prompt(
        tickers: list,
        recommendation_table: list,
        bl_views: dict,
        quant_commentary: str,
        risk_commentary: str,
        mc_portfolio_current: dict,
        mc_portfolio_recommended: dict,
        aaii_sentiment: dict,
        fear_greed: dict,
        cape: float,
        total_portfolio_value: float,
    ) -> str:

    # ── Recommendation table block ────────────────────────────────
    rec_lines = []
    for row in recommendation_table:
        rec_lines.append(
            f"{row['ticker']}: "
            f"current={row['current_weight']:.1%}  "
            f"consensus={row['consensus_action']}  "
            f"view_return={row['view_return']:+.1%}  "
            f"confidence={row['confidence']}  "
            f"conflict={row['conflict']}  "
            f"reasoning={row['reasoning']}"
        )

    # ── Strategy weight comparison block ─────────────────────────
    strategies = [s for s in mc_portfolio_recommended.keys()] if mc_portfolio_recommended else []
    weight_lines = []
    for row in recommendation_table:
        ticker = row["ticker"]
        parts  = [f"current={row['current_weight']:.1%}"]
        for s in strategies:
            w      = row.get(f"{s}_weight", 0)
            action = row.get(f"{s}_action", "N/A")
            parts.append(f"{s}={w:.1%}({action})")
        weight_lines.append(f"{ticker}: " + "  ".join(parts))

    # ── Simulation comparison block ───────────────────────────────
    sim_lines = []

    curr = mc_portfolio_current or {}
    sim_lines.append(
        f"Current portfolio:"
        f"  median_value=${curr.get('p50', 0):,.0f}CAD"
        f"  VaR95=${curr.get('var_95', 0):,.0f}CAD"
        f"  CVaR95=${curr.get('cvar_95', 0):,.0f}CAD"
        f"  prob_up={curr.get('prob_up', 0):.1%}"
        f"  expected_value=${curr.get('expected_value', 0):,.0f}CAD"
    )

    for strategy, sim in (mc_portfolio_recommended or {}).items():
        if sim:
            sim_lines.append(
                f"{strategy}:"
                f"  median_value=${sim.get('p50', 0):,.0f}CAD"
                f"  VaR95=${sim.get('var_95', 0):,.0f}CAD"
                f"  CVaR95=${sim.get('cvar_95', 0):,.0f}CAD"
                f"  prob_up={sim.get('prob_up', 0):.1%}"
                f"  expected_value=${sim.get('expected_value', 0):,.0f}CAD"
            )

    # ── Strategy disagreement detection ──────────────────────────
    disagreement_lines = []
    for row in recommendation_table:
        ticker  = row["ticker"]
        actions = [row.get(f"{s}_action", "HOLD") for s in strategies]
        unique  = set(actions)
        if len(unique) > 1:
            action_summary = "  ".join(
                f"{s}={row.get(f'{s}_action', 'N/A')}"
                for s in strategies
            )
            disagreement_lines.append(f"{ticker}: {action_summary}")

    disagreement_block = (
        "STRATEGY DISAGREEMENTS:\n" + "\n".join(disagreement_lines)
        if disagreement_lines
        else "All strategies are in consensus on all positions."
    )

    prompt = f"""You are a portfolio manager writing a weekly briefing for a long-term investor.
Total portfolio value: ${total_portfolio_value:,.0f} CAD

MACRO CONTEXT:
  Shiller CAPE         : {cape}
  CNN Fear & Greed     : {fear_greed.get('score')} ({fear_greed.get('rating')})
  AAII Bullish         : {aaii_sentiment.get('bullish')}
  AAII Bearish         : {aaii_sentiment.get('bearish')}
  AAII Neutral         : {aaii_sentiment.get('neutral')}
  AAII Bull-Bear Spread: {aaii_sentiment.get('bull_bear_spread')}

PORTFOLIO RECOMMENDATIONS:
{chr(10).join(rec_lines)}

STRATEGY WEIGHT COMPARISON:
{chr(10).join(weight_lines)}

{disagreement_block}

SIMULATION RESULTS:
{chr(10).join(sim_lines)}

RISK COMMENTARY:
{risk_commentary}

QUANTITATIVE MODEL COMMENTARY:
{quant_commentary}

Write a unified portfolio briefing of 4-6 paragraphs covering:

1. Macro environment — what the CAPE, Fear & Greed, and AAII readings mean 
   for the portfolio right now. If CAPE is above 30 address valuation risk 
   explicitly. If Fear & Greed is below 25 or above 75 note the contrarian signal.

2. Key BUY and SELL recommendations — for every position where consensus action 
   is BUY or SELL explain specifically what drove it. Name the metrics, name the 
   sentiment signals, name the sources that supported the view.

3. Sentiment vs fundamentals conflicts — for every position where conflict=True 
   explain the tension explicitly. A cheap stock with negative sentiment is a 
   different situation from a cheap stock with improving sentiment. Say which 
   it is and what would resolve the conflict.

4. Strategy disagreements — for every position where strategies disagree 
   explain why they diverge. Name the strategies on each side. Note whether 
   the disagreement reflects genuine uncertainty or a known model difference 
   such as risk parity ignoring return estimates.

5. Risk and simulation context — reference the VaR and CVaR numbers in dollar 
   terms. Compare the current portfolio simulation to the recommended strategy 
   simulations. Recommend which single strategy you find most compelling given 
   the current macro environment and simulation outcomes — give a specific 
   reason not just the best expected return.

Rules:
- Be specific — name metrics, name values, name tickers
- Never say "consider" or "may want to" — give a clear view
- Do not restate numbers already visible in the tables — interpret what they mean
- Write for an investor reading this at 7am on Monday morning
- 4-6 paragraphs, 4-6 sentences each
- Plain prose only — no headers, no bullet points, no markdown
- Focus more on the macro environment and portfolio recommendations.

Return only the commentary text."""

    return prompt


def generate_advisory_commentary(
        tickers: list,
        recommendation_table: list,
        bl_views: dict,
        quant_commentary: str,
        risk_commentary: str,
        mc_portfolio_current: dict,
        mc_portfolio_recommended: dict,
        aaii_sentiment: dict,
        fear_greed: dict,
        cape: float,
        total_portfolio_value: float,
    ) -> str:
    """
    Final LLM call in the pipeline. Produces the advisory commentary
    that appears as the main prose section of the weekly report.
    Temperature 0.3 — this is prose the user reads, not structured data.
    """
    prompt = build_advisory_prompt(
        tickers                  = tickers,
        recommendation_table     = recommendation_table,
        bl_views                 = bl_views,
        quant_commentary         = quant_commentary,
        risk_commentary          = risk_commentary,
        mc_portfolio_current     = mc_portfolio_current,
        mc_portfolio_recommended = mc_portfolio_recommended,
        aaii_sentiment           = aaii_sentiment,
        fear_greed               = fear_greed,
        cape                     = cape,
        total_portfolio_value    = total_portfolio_value,
    )

    try:
        llm      = get_llm(0.3)
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        logger.warning(f"Advisory commentary LLM call failed: {e}")
        return (
            "Advisory commentary unavailable this run. "
            "Review the recommendation table and simulation results directly."
        )
    


def agent4_simulator(state: PipelineState) -> dict:
    """
    Agent 4 — Monte Carlo Simulator
    Runs Monte Carlo simulations for current and candidate rebalanced portfolios.
    Uses mu and sigma estimates from agent 2

    Args:
        state: Pipeline state dict containing tickers, run_date, and errors.

    Returns:
        Partial state update dict with keys: mc_current, mc_rebalanced, risk_commentary, errors.
    """
    logger.info("[Agent 4] Simulator running")
    tickers  = state["tickers"]
    logger.info(f"  Tickers : {tickers}")
    errors   = list(state.get("errors") or [])
    existing_errors = len(errors)
    covar = pd.DataFrame(state["covariance_matrix"])
    covar = covar.loc[tickers, tickers]          # align to tickers order
    Sigma = covar.values * 252
    curr_weights = [state["current_weights"][t] for t in tickers]
    total_value = state["total_portfolio_value"]
    strategies = list(state["recommended_weights"].keys())
    port_info = state["mu_sigma"]
    mu_vec = get_mu_for_tickers(tickers, state)
    fallback_present = {t: port_info[t]["is_fallback"] for t in tickers}

    sum_dir = Path(state["user_path"]) / "data" / state["run_date"] / "summaries"

    mc_current = {}
    mc_port_current = {}
    mc_port_rebalanced = {}

    for ticker in tickers:
        logger.info(f"  Monte Carlo Sim: Ticker ({ticker})")
        state_info = port_info[ticker]
        dir_name = ticker.removesuffix(".TO")
        (sum_dir / dir_name).mkdir(parents=True, exist_ok=True)
        with open(sum_dir / dir_name / "sim.txt", "w", encoding="utf-8") as f:
            if not state_info.get("s_current"):
                logger.warning(f"No valid price for {ticker} — skipping simulation.")
                mc_current[ticker] = {}
                continue
            mc_current[ticker] = run_monte_carlo(
                state_info["mu_annual"], state_info["sigma_annual"], state_info["s_current"], file=f
            )

    port_sum_dir = sum_dir / "_portfolio"
    port_sum_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Monte Carlo Sim: current weights")
    with open(port_sum_dir / "mc_current.txt", "w", encoding="utf-8") as f:
        mc_port_current = port_monte_carlo(mu_vec, Sigma, total_value, curr_weights, file=f)

    for strategy in strategies:
        logger.info(f"  Monte Carlo Sim: Rebalanced ({strategy})")
        if state["recommended_weights"].get(strategy) is None:
            logger.warning(f"Recommendation failed for {strategy}")
            continue
        weights = [state["recommended_weights"][strategy][ticker] for ticker in tickers]
        with open(port_sum_dir / f"mc_{strategy}.txt", "w", encoding="utf-8") as f:
            for row in state["recommendation_table"]:
                print(row, file=f)
            mc_port_rebalanced[strategy] = port_monte_carlo(mu_vec, Sigma, total_value, weights, file=f)
    
    
    try:
        with open(port_sum_dir / "llm_commentary.txt", "w", encoding="utf-8") as f:
            sim_commentary = generate_sim_commentary(tickers, strategies, mc_current, mc_port_current, mc_port_rebalanced, fallback_present, file=f)
    except Exception as e:
            errors.append(f"Sim Commentary Failed: {e}")
            sim_commentary = None
            
    logger.info(f"[Agent 4] Complete. New Errors: {len(errors) - existing_errors}")
    
    advisory_commentary = generate_advisory_commentary(
        tickers                  = state["tickers"],
        recommendation_table     = state["recommendation_table"],
        bl_views                 = state["bl_views"],
        quant_commentary         = state["quant_commentary"],
        risk_commentary          = sim_commentary,
        mc_portfolio_current     = mc_port_current,
        mc_portfolio_recommended = mc_port_rebalanced,
        aaii_sentiment           = state["aaii_sentiment"] or {},
        fear_greed               = state["fear_greed"] or {},
        cape                     = state.get("cape") or 25.0,
        total_portfolio_value    = state["total_portfolio_value"],
    )

    return {
        "mc_current":     mc_current,
        "mc_port_current": mc_port_current,
        "mc_port_rebalanced":  mc_port_rebalanced,
        "sim_commentary": sim_commentary,
        "advisory_commentary" : advisory_commentary,
        "errors": errors,
    }