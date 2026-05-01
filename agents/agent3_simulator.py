import numpy as np
import pandas as pd
import scipy.stats as stats

from agents.state import PipelineState


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
        "VaR_95": VaR_95,
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
        "VaR_95": VaR_95,
        "CVaR_95": CVaR_95,
        "pct": pct,
        "N_paths": N_paths,
        "N_steps": N_steps,
        "S_current": init_port_value,
        "df_ch": df_ch,
        "ch_scale": ch_scale,
    }


def get_rebalanced_weights(tickers):
    """
    STUB for now
    """
    return [1/len(tickers)] * len(tickers)
    


def agent3_simulator(state: PipelineState) -> dict:
    """
    Agent 3 — Monte Carlo Simulator
    Runs Monte Carlo simulations for current and candidate rebalanced portfolios.
    Uses mu and sigma estimates from agent 2

    Args:
        state: Pipeline state dict containing tickers, run_date, and errors.

    Returns:
        Partial state update dict with keys: mc_current, mc_rebalanced, risk_commentary, errors.
    """
    print(f"  [Agent 3] Simulator running")
    tickers  = state["tickers"]
    print(f"            Tickers : {tickers}")
    errors   = list(state.get("errors") or [])
    existing_errors = len(errors)
    covar = pd.DataFrame(state["covariance_matrix"])
    covar = covar.loc[tickers, tickers]          # align to tickers order
    Sigma = covar.values * 252
    curr_weights = [state["current_weights"][t] for t in tickers]
    total_value = state["total_portfolio_value"]
    strategies = state["strategies"]
    port_info = state["mu_sigma"]
    mu_vec = np.array([port_info[t]["mu_annual"] for t in tickers])
    
    
    mc_current = {}
    mc_port_current = {}
    mc_port_rebalanced = {}
    
    for ticker in tickers:
        print(f"\nMonte Carlo Sim: Ticker ({ticker})")
        state_info = port_info[ticker]
        mc_current[ticker] = run_monte_carlo(state_info["mu_annual"], state_info["sigma_annual"], state_info["s_current"])
        
    print(f"\nMonte Carlo Sim: current weights")
    mc_port_current = port_monte_carlo(mu_vec, Sigma, total_value, curr_weights)
    
    for strategy in strategies:
        print(f"\nMonte Carlo Sim: Rebalanced ({strategy})")
        weights = get_rebalanced_weights(tickers)
        mc_port_rebalanced[strategy] = port_monte_carlo(mu_vec, Sigma, total_value, weights)
        
    print(f"\n  [Agent 3] Complete. New Errors: {len(errors) - existing_errors}")

    return {
        "mc_current":     mc_current,
        "mc_port_current": mc_port_current,
        "mc_port_rebalanced":  mc_port_rebalanced,
        "risk_commentary": "[STUB] Simulation not yet implemented.",
        "errors": errors,
    }