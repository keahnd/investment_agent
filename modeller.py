"""
Asset Price Modelling Pipeline
================================
Steps:
  1. Load historical price data from yfinance
  2. Compute log returns
  3. Fama-French style 3-factor OLS regression  -> mu, alpha, betas
  4. GARCH(1,1) volatility estimation           -> time-varying sigma
  5. Monte Carlo simulation (GBM + fat tails)   -> 1-year price distribution
  6. Portfolio optimisation (Max Sharpe + Min Vol) -> rebalancing signals
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import yfinance as yf
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier
from scipy.optimize import minimize

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

END_DATE = datetime.today()
LOOKBACK_DAYS = 5 * 365
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)


def load_portfolio(csv_path):
    """
    Fetches the current portfolio tickers and weights from the CSV file.
    Args:
        csv_path: Path to the csv file
    Returns:
        DataFrame object of current holdings
    """
    return pd.read_csv(csv_path)


def fetch_prices(tickers):
    """
    Fetches YFinance data for the tickers in the list.
    Asset  : TICKERS
    Market : SPY  (S&P 500 — MKT factor proxy)
    Size   : IWM - SPY  (Russell 2000 minus S&P 500 — SMB proxy)
    Value  : IVE - IVW  (S&P 500 Value minus Growth — HML proxy)
    Rf     : ^IRX (13-week T-bill annualised yield in %)

    Args:
        tickers: List of tickers to fetch
    Returns:
        Tuple of (price DataFrame, rf_ann float, rf_daily float)
    """
    all_tickers = list(tickers) + ['SPY', 'IWM', 'IVE', 'IVW', '^IRX']
    data = yf.download(
        all_tickers, start=START_DATE, end=END_DATE,
        auto_adjust=True, progress=False, threads=False
    )['Close'].dropna()
    dt = 1 / 252
    rf_ann = float(data['^IRX'].mean()) / 100
    rf_daily = rf_ann * dt
    return data, rf_ann, rf_daily


def compute_returns(raw, ticker, rf_ann):
    """
    Computes log returns, excess returns, and Fama-French factor series.

    Args:
        raw: Price DataFrame from fetch_prices
        ticker: Asset ticker string
        rf_ann: Annualised risk-free rate

    Returns:
        dict with log_ret, excess_ret, MKT, SMB, HML, factor_vols,
              prices, S0, T_hist, rf_daily
    """
    dt = 1 / 252
    rf_daily = rf_ann * dt

    prices = raw[ticker].values
    S0 = float(prices[0])

    log_ret = np.log(prices[1:] / prices[:-1])
    excess_ret = log_ret - rf_daily

    spy_rets = np.log(raw['SPY'].values[1:] / raw['SPY'].values[:-1])
    iwm_rets = np.log(raw['IWM'].values[1:] / raw['IWM'].values[:-1])
    ive_rets = np.log(raw['IVE'].values[1:] / raw['IVE'].values[:-1])
    ivw_rets = np.log(raw['IVW'].values[1:] / raw['IVW'].values[:-1])

    MKT = spy_rets - rf_daily
    SMB = iwm_rets - spy_rets
    HML = ive_rets - ivw_rets

    T_hist = len(log_ret)
    factor_vols = np.array([MKT.std(), SMB.std(), HML.std()])

    print(f"\n[Returns — {ticker}]")
    print(f"  Annualised mean return : {log_ret.mean() * 252:.2%}")
    print(f"  Annualised hist. vol   : {log_ret.std() * np.sqrt(252):.2%}")
    print(f"  Skewness               : {stats.skew(log_ret):.3f}")
    print(f"  Excess kurtosis        : {stats.kurtosis(log_ret):.3f}")
    print(f"  {T_hist} trading days  ({START_DATE:%Y-%m-%d} → {END_DATE:%Y-%m-%d})")
    print(f"  S0 = {S0:.2f},  S_final = {float(prices[-1]):.2f}")
    print(f"  Avg risk-free (ann.) = {rf_ann:.2%}")

    return {
        "log_ret": log_ret,
        "excess_ret": excess_ret,
        "MKT": MKT,
        "SMB": SMB,
        "HML": HML,
        "factor_vols": factor_vols,
        "prices": prices,
        "S0": S0,
        "T_hist": T_hist,
        "rf_daily": rf_daily,
    }


def run_factor_models(excess_ret, MKT, SMB, HML, rf_ann):
    """
    Runs 3-factor OLS regression on excess returns.

    Args:
        excess_ret: Daily excess return array
        MKT, SMB, HML: Fama-French factor arrays
        rf_ann: Annualised risk-free rate

    Returns:
        dict with alpha_daily, b_MKT, b_SMB, b_HML, mu_annual,
              residuals, r_squared, Y_hat
    """
    T_hist = len(excess_ret)
    X = np.column_stack([np.ones(T_hist), MKT, SMB, HML])
    Y = excess_ret

    XtX_inv = np.linalg.inv(X.T @ X)
    betas = XtX_inv @ X.T @ Y
    Y_hat = X @ betas
    residuals = Y - Y_hat

    n, k = T_hist, 4
    s2 = residuals @ residuals / (n - k)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    t_stats = betas / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    SS_tot = np.sum((Y - Y.mean()) ** 2)
    SS_res = residuals @ residuals
    R2 = 1 - SS_res / SS_tot
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - k)

    labels = ["Alpha (daily)", "Beta_MKT", "Beta_SMB", "Beta_HML"]
    print("\n[Factor Model Regression]")
    print(f"  {'Parameter':<18} {'Estimate':>10} {'Std Err':>10} {'t-stat':>8} {'p-value':>8}")
    print("  " + "-" * 60)
    for i, lbl in enumerate(labels):
        sig = "***" if p_vals[i] < 0.001 else "**" if p_vals[i] < 0.01 else "*" if p_vals[i] < 0.05 else ""
        print(f"  {lbl:<18} {betas[i]:>10.6f} {se[i]:>10.6f} {t_stats[i]:>8.2f} {p_vals[i]:>8.4f} {sig}")
    print(f"\n  R²      = {R2:.4f}")
    print(f"  Adj. R² = {R2_adj:.4f}")

    alpha_daily = betas[0]
    b_MKT, b_SMB, b_HML = betas[1], betas[2], betas[3]

    factor_annual_means = np.array([
        MKT.mean() * 252,
        SMB.mean() * 252,
        HML.mean() * 252,
    ])
    mu_annual = (
        rf_ann
        + alpha_daily * 252
        + b_MKT * factor_annual_means[0]
        + b_SMB * factor_annual_means[1]
        + b_HML * factor_annual_means[2]
    )
    print(f"\n  Estimated annualised mu  : {mu_annual:.2%}")

    return {
        "alpha_daily": alpha_daily,
        "b_MKT": b_MKT,
        "b_SMB": b_SMB,
        "b_HML": b_HML,
        "mu_annual": mu_annual,
        "residuals": residuals,
        "r_squared": R2,
        "Y_hat": Y_hat,
    }


def garch_neg_log_likelihood(params, returns):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    n = len(returns)
    h = np.zeros(n)
    h[0] = np.var(returns)
    ll = 0.0
    for t in range(1, n):
        h[t] = omega + alpha * returns[t - 1] ** 2 + beta * h[t - 1]
        if h[t] <= 0:
            return 1e10
        ll += 0.5 * (np.log(2 * np.pi) + np.log(h[t]) + returns[t] ** 2 / h[t])
    return ll


def run_garch(residuals, factor_vols, b_MKT, b_SMB, b_HML):
    """
    Fits GARCH(1,1) to OLS residuals and computes total asset volatility.

    Args:
        residuals: Regression residual array
        factor_vols: Array of [MKT, SMB, HML] daily std devs
        b_MKT, b_SMB, b_HML: Factor loadings from run_factor_models

    Returns:
        dict with h_garch, sigma_t, sigma_total_annual, garch_persist,
              garch_long_run_vol
    """
    T_hist = len(residuals)
    x0 = [1e-6, 0.08, 0.90]
    bounds = [(1e-9, 0.01), (1e-4, 0.49), (1e-4, 0.99)]
    result = minimize(
        garch_neg_log_likelihood, x0, args=(residuals,),
        method='L-BFGS-B', bounds=bounds
    )

    omega_g, alpha_g_est, beta_g_est = result.x
    garch_persist = alpha_g_est + beta_g_est
    garch_long_run_vol = np.sqrt(omega_g / (1 - garch_persist)) * np.sqrt(252)

    h_garch = np.zeros(T_hist)
    h_garch[0] = np.var(residuals)
    for t in range(1, T_hist):
        h_garch[t] = omega_g + alpha_g_est * residuals[t - 1] ** 2 + beta_g_est * h_garch[t - 1]

    sigma_t = np.sqrt(h_garch)
    sigma_current_daily = np.sqrt(h_garch[-1])
    sigma_total_annual = np.sqrt(
        (b_MKT ** 2 * factor_vols[0] ** 2 + b_SMB ** 2 * factor_vols[1] ** 2 + b_HML ** 2 * factor_vols[2] ** 2) * 252
        + h_garch[-1] * 252
    )

    print("\n[GARCH(1,1) on Idiosyncratic Residuals]")
    print(f"  omega               : {omega_g:.2e}")
    print(f"  alpha (ARCH)        : {alpha_g_est:.4f}")
    print(f"  beta  (GARCH)       : {beta_g_est:.4f}")
    print(f"  Persistence (α+β)   : {garch_persist:.4f}")
    print(f"  Long-run vol (ann.) : {garch_long_run_vol:.2%}")
    print(f"  Current cond. vol   : {sigma_current_daily * np.sqrt(252):.2%} (annualised idiosyncratic)")
    print(f"  Total asset vol     : {sigma_total_annual:.2%} (annualised)")

    return {
        "h_garch": h_garch,
        "sigma_t": sigma_t,
        "sigma_total_annual": sigma_total_annual,
        "garch_persist": garch_persist,
        "garch_long_run_vol": garch_long_run_vol,
    }


def run_monte_carlo(mu, sigma, S0):
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

    print(f"\n[Monte Carlo Simulation]")
    print(f"  Paths      : {N_paths:,}")
    print(f"  Steps      : {N_steps} days")
    print(f"  mu         : {mu:.2%} p.a.")
    print(f"  sigma      : {sigma:.2%} p.a.")
    print(f"  Innovation : Student-t (df={df_t})")
    print(f"  S_current  : {S_current:.2f}")

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
    CVaR_95 = losses[losses >= VaR_95].mean()

    print(f"\n  ── Terminal Price Distribution (S_T) ──")
    print(f"  E[S_T]              : {E_ST:.2f}")
    print(f"  Median              : {pct[3]:.2f}")
    print(f"  Std dev             : {S_T.std():.2f}")
    print(f"  1st  percentile     : {pct[0]:.2f}")
    print(f"  5th  percentile     : {pct[1]:.2f}")
    print(f"  25th percentile     : {pct[2]:.2f}")
    print(f"  75th percentile     : {pct[4]:.2f}")
    print(f"  95th percentile     : {pct[5]:.2f}")
    print(f"  99th percentile     : {pct[6]:.2f}")
    print(f"\n  Prob(S_T > S_current): {prob_up:.2%}")
    print(f"  1-year 95% VaR      : {VaR_95:.2f}  ({VaR_95 / S_current:.2%} of price)")
    print(f"  1-year 95% CVaR     : {CVaR_95:.2f}  ({CVaR_95 / S_current:.2%} of price)")

    return {
        "paths": paths,
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


def fetch_valuation_metrics(ticker: str) -> dict:
    """
    Fetches valuation metrics for ticker from yfinance.

    Args:
        ticker: Ticker whose information is required

    Returns:
        dict: Valuation metrics for ticker
    """
    info = yf.Ticker(ticker).info
    return {
        'peg': info.get('pegRatio'),
        'fwd_pe': info.get('forwardPE'),
        'ttm_pe': info.get('trailingPE'),
        'ev_ebitda': info.get('enterpriseToEbitda'),
    }


def build_covariance(returns_dict):
    """
    Builds an annualised covariance matrix from per-ticker log return Series.

    Args:
        returns_dict: {ticker: log_ret np.array}

    Returns:
        Annualised covariance matrix as DataFrame
    """
    df = pd.DataFrame(returns_dict).dropna()
    return df.cov() * 252


def run_optimisation(mu_dict, cov_matrix, rf_ann):
    """
    Runs Max Sharpe and Min Volatility portfolio optimisations.

    Args:
        mu_dict: {ticker: annualised expected return}
        cov_matrix: Annualised covariance matrix DataFrame
        rf_ann: Annualised risk-free rate

    Returns:
        dict with 'max_sharpe' and 'min_vol' weight dicts
    """
    mu_series = pd.Series(mu_dict)

    ef_sharpe = EfficientFrontier(mu_series, cov_matrix)
    ef_sharpe.max_sharpe(risk_free_rate=rf_ann)
    weights_sharpe = ef_sharpe.clean_weights()

    ef_minvol = EfficientFrontier(mu_series, cov_matrix)
    ef_minvol.min_volatility()
    weights_minvol = ef_minvol.clean_weights()

    return {"max_sharpe": weights_sharpe, "min_vol": weights_minvol}


def print_recommendation(portfolio_df, opt_weights, scenario_name):
    """
    Compares optimised weights to current holdings and prints BUY/SELL/HOLD signals.

    Args:
        portfolio_df: DataFrame with 'ticker' and 'weight' columns
        opt_weights: {ticker: target_weight} from run_optimisation
        scenario_name: Label string e.g. 'Max Sharpe'
    """
    current = dict(zip(portfolio_df['ticker'], portfolio_df['weight']))
    threshold = 0.02

    print(f"\n--- {scenario_name} Rebalancing ---")
    print(f"{'Ticker':<8} {'Current':>10} {'Target':>10} {'Delta':>10} {'Signal':>8}")
    for ticker, target in opt_weights.items():
        current_w = current.get(ticker, 0.0)
        delta = target - current_w
        if delta > threshold:
            signal = "BUY"
        elif delta < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        print(f"{ticker:<8} {current_w:>10.1%} {target:>10.1%} {delta:>+10.1%} {signal:>8}")


def compute_confidence_score(n_obs, r_squared, garch_persist):
    history_score = min(1.0, max(0.0, (n_obs - 252) / (1260 - 252)))
    r2_score = min(1.0, max(0.0, r_squared))
    garch_score = 1.0 - max(0.0, (garch_persist - 0.90) / 0.09)
    score = 0.50 * history_score + 0.30 * r2_score + 0.20 * garch_score

    if score >= 0.70:
        confidence = 'high'
    elif score >= 0.45:
        confidence = 'medium'
    else:
        confidence = 'low'

    return score, confidence


def plot_outputs(ticker, ret, fm, g, mc):
    """
    Generates a 7-panel analysis dashboard for a single ticker.

    Args:
        ticker: Ticker string (used in title and filename)
        ret: Output dict from compute_returns
        fm:  Output dict from run_factor_models
        g:   Output dict from run_garch
        mc:  Output dict from run_monte_carlo
    """
    prices = ret["prices"]
    excess_ret = ret["excess_ret"]
    sigma_t = g["sigma_t"]
    garch_long_run_vol = g["garch_long_run_vol"]
    N_paths = mc["N_paths"]
    N_steps = mc["N_steps"]
    paths = mc["paths"]
    S_current = mc["S_current"]
    S_T = mc["S_T"]
    E_ST = mc["E_ST"]
    pct = mc["pct"]
    Y_hat = fm["Y_hat"]
    R2 = fm["r_squared"]
    residuals = fm["residuals"]
    t_scale = mc["t_scale"]
    df_t = mc["df_t"]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    DARK = '#0f1117'
    PANEL = '#1a1d27'
    CYAN = '#00d4ff'
    ORANGE = '#ff6b35'
    GREEN = '#00ff9f'
    RED = '#ff4757'
    GREY = '#8892a4'
    WHITE = '#e8eaf0'

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors=GREY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2d3a')
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        ax.grid(True, color='#2a2d3a', linewidth=0.5, alpha=0.7)

    # ── (A) Historical price ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    days_hist = np.arange(len(prices))
    ax1.plot(days_hist, prices, color=CYAN, linewidth=1.0, alpha=0.9)
    ax1.fill_between(days_hist, prices.min(), prices, color=CYAN, alpha=0.08)
    style_ax(ax1, "A — Historical Price Path (5 Years)")
    ax1.set_xlabel("Trading Day")
    ax1.set_ylabel("Price ($)")

    # ── (B) GARCH conditional volatility ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    garch_vol_ann = sigma_t * np.sqrt(252)
    ax2.plot(garch_vol_ann, color=ORANGE, linewidth=0.8)
    ax2.axhline(garch_long_run_vol, color=GREY, linestyle='--', linewidth=1, label='Long-run vol')
    style_ax(ax2, "B — GARCH(1,1) Conditional Vol")
    ax2.set_xlabel("Trading Day")
    ax2.set_ylabel("Annualised Vol")
    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)

    # ── (C) Monte Carlo paths (sample 200) ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    sample_idx = np.random.choice(N_paths, 200, replace=False)
    t_axis = np.arange(N_steps + 1)
    for i in sample_idx:
        ax3.plot(t_axis, paths[:, i], alpha=0.04, linewidth=0.5, color=CYAN)

    p5 = np.percentile(paths, 5, axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    ax3.fill_between(t_axis, p5, p95, alpha=0.15, color=CYAN, label='5–95%')
    ax3.fill_between(t_axis, p25, p75, alpha=0.25, color=CYAN, label='25–75%')
    ax3.plot(t_axis, p50, color=GREEN, linewidth=1.5, label='Median')
    ax3.plot(t_axis, p5, color=RED, linewidth=0.8, linestyle='--', label='5th pct')
    ax3.plot(t_axis, p95, color=ORANGE, linewidth=0.8, linestyle='--', label='95th pct')
    ax3.axhline(S_current, color=GREY, linewidth=0.8, linestyle=':')
    style_ax(ax3, f"C — Monte Carlo Simulation ({N_paths:,} paths, 1 Year)")
    ax3.set_xlabel("Trading Day")
    ax3.set_ylabel("Price ($)")
    ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, ncol=3)

    # ── (D) Terminal price distribution ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(S_T, bins=80, color=CYAN, alpha=0.7, edgecolor='none', density=True)
    ax4.axvline(S_current, color=WHITE, linewidth=1.2, linestyle=':', label=f'Current ({S_current:.0f})')
    ax4.axvline(E_ST, color=GREEN, linewidth=1.5, linestyle='-', label=f'E[S_T] ({E_ST:.0f})')
    ax4.axvline(pct[1], color=RED, linewidth=1.2, linestyle='--', label=f'5th pct ({pct[1]:.0f})')
    ax4.axvline(pct[5], color=ORANGE, linewidth=1.2, linestyle='--', label=f'95th pct ({pct[5]:.0f})')
    style_ax(ax4, "D — Terminal Price Distribution")
    ax4.set_xlabel("S_T ($)")
    ax4.set_ylabel("Density")
    ax4.legend(fontsize=6.5, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)

    # ── (E) Factor model: actual vs fitted ───────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(Y_hat[:500], excess_ret[:500], alpha=0.3, s=4, color=CYAN)
    lims = [min(Y_hat.min(), excess_ret.min()), max(Y_hat.max(), excess_ret.max())]
    ax5.plot(lims, lims, color=ORANGE, linewidth=1.2, label=f'45° line  R²={R2:.3f}')
    style_ax(ax5, "E — Factor Model: Actual vs Fitted")
    ax5.set_xlabel("Fitted excess return")
    ax5.set_ylabel("Actual excess return")
    ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)

    # ── (F) Residuals distribution vs Normal ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(residuals, bins=60, color=CYAN, alpha=0.6, density=True, label='Residuals', edgecolor='none')
    x_range = np.linspace(residuals.min(), residuals.max(), 300)
    normal_pdf = stats.norm.pdf(x_range, 0, residuals.std())
    t_pdf = stats.t.pdf(x_range / (residuals.std() / t_scale), df_t) / (residuals.std() / t_scale)
    ax6.plot(x_range, normal_pdf, color=ORANGE, linewidth=1.5, label='Normal fit')
    ax6.plot(x_range, t_pdf, color=GREEN, linewidth=1.5, label=f'Student-t (df={df_t})')
    style_ax(ax6, "F — Residual Distribution vs. Parametric Fits")
    ax6.set_xlabel("Residual")
    ax6.set_ylabel("Density")
    ax6.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)

    # ── (G) QQ plot of residuals ──────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
    ax7.scatter(osm, osr, s=3, alpha=0.4, color=CYAN)
    qq_line = np.array([osm[0], osm[-1]])
    ax7.plot(qq_line, slope * qq_line + intercept, color=ORANGE, linewidth=1.5)
    style_ax(ax7, "G — Normal Q-Q Plot of Residuals")
    ax7.set_xlabel("Theoretical quantiles")
    ax7.set_ylabel("Sample quantiles")

    fig.suptitle(
        f"{ticker} — Factor Regression  ·  GARCH(1,1)  ·  Monte Carlo Simulation",
        color=WHITE, fontsize=13, fontweight='bold', y=0.98
    )

    filename = f"{ticker}_asset_price_model.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"\n[Done]  Plot saved to {filename}")


def main():
    portfolio_df = load_portfolio("portfolio.csv")
    tickers = portfolio_df['ticker'].tolist()

    raw_prices, rf_ann, rf_daily = fetch_prices(tickers)

    results = []
    returns_dict = {}   # {ticker: log_ret} for covariance matrix
    mu_dict = {}        # {ticker: mu_annual} for optimisation

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")
        try:
            ret = compute_returns(raw_prices, ticker, rf_ann)
            fm = run_factor_models(ret["excess_ret"], ret["MKT"], ret["SMB"], ret["HML"], rf_ann)
            g = run_garch(fm["residuals"], ret["factor_vols"], fm["b_MKT"], fm["b_SMB"], fm["b_HML"])
            mc = run_monte_carlo(fm["mu_annual"], g["sigma_total_annual"], float(raw_prices[ticker].iloc[-1]))
            val = fetch_valuation_metrics(ticker)
            score, conf = compute_confidence_score(ret["T_hist"], fm["r_squared"], g["garch_persist"])

            plot_outputs(ticker, ret, fm, g, mc)

            returns_dict[ticker] = ret["log_ret"]
            mu_dict[ticker] = fm["mu_annual"]

            results.append({
                "ticker": ticker,
                "mu_annual": fm["mu_annual"],
                "sigma_annual": g["sigma_total_annual"],
                "VaR_95": mc["VaR_95"],
                "CVaR_95": mc["CVaR_95"],
                "E_ST": mc["E_ST"],
                "prob_up": mc["prob_up"],
                "confidence": conf,
                **val,
            })

        except Exception as e:
            print(f"  [WARN] {ticker} failed: {e}")

    # ── Portfolio-level optimisation ─────────────────────────────────────────
    if len(returns_dict) >= 2:
        print(f"\n{'='*60}")
        print("  Portfolio Optimisation")
        print(f"{'='*60}")
        cov_matrix = build_covariance(returns_dict)
        opt_weights = run_optimisation(mu_dict, cov_matrix, rf_ann)
        print_recommendation(portfolio_df, opt_weights["max_sharpe"], "Max Sharpe")
        print_recommendation(portfolio_df, opt_weights["min_vol"], "Min Volatility")
    else:
        print("\n[WARN] Need at least 2 tickers for portfolio optimisation.")

    # ── Per-ticker summary table ──────────────────────────────────────────────
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print("  Per-Ticker Summary")
        print(f"{'='*60}")
        print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
