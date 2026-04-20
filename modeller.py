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
import pprint
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import yfinance as yf
import requests, zipfile, io
from datetime import datetime, timedelta, date
from pypfopt import EfficientFrontier
from arch import arch_model
import time
import os
import sys
import json
import traceback
from pathlib import Path
from curl_cffi import requests
from database.db import init_database, insert_portfolio_row


class _Tee:
    """Writes to both the real stdout and a report file simultaneously."""
    def __init__(self, file_obj):
        self._file = file_obj
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

np.random.seed(42)

END_DATE = datetime.today()
LOOKBACK_DAYS = 5 * 365
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)
CACHE_FILE = "prices_cache.parquet"
FF_CACHE_FILE = "ff_factors_cache.parquet"
VAL_CACHE_FILE = "valuation_cache.json"
USERS_DIR = Path("users")


def discover_users() -> list[Path]:
    """
    Returns a list of user directories
    
    Returns:
        List of user directories
    """
    return [
        p for p in USERS_DIR.iterdir()
        if p.is_dir() and (p / "config.json").exists() and (p / "portfolio.csv").exists()
    ]
    

def load_portfolio(user_path):
    """
    Fetches the current portfolio tickers and weights from the CSV file.
    Args:
        user_path: Path to the users folder
    Returns:
        DataFrame object of current holdings
    """
    user_dir = Path(user_path)
    return pd.read_csv(user_dir / "portfolio.csv")


def load_config(user_path):
    """
    Loads the config file for the specified user
    Args:
        user_path: Path to the users folder
    Returns:
        json object of user details
    """
    user_dir = Path(user_path)
    with open(user_dir / 'config.json') as f:
        return json.load(f)


def _download_with_retry(tickers, start, end):
    for attempt in range(5):
        wait = 30 * (attempt + 1)   # 30s, 60s, 90s, 120s, 150s
        try:
            session = requests.Session(impersonate="chrome")
            data = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=False, session=session
            )['Close']
            if data.empty:
                raise ValueError("Download returned empty DataFrame (likely rate limited).")
            return data
        except Exception as e:
            if attempt < 4:
                print(f"  [WARN] Download failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to download price data after 5 attempts: {e}")


def fetch_prices(tickers, force_refresh=False):
    """
    Fetches asset prices from yfinance for the given tickers only.
    Factor data (MKT, SMB, HML, RF) is now sourced from fetch_ff_factors().

    Results are cached to CACHE_FILE. On subsequent runs only the missing
    date window is downloaded and appended. Pass force_refresh=True or
    delete the cache file to re-download everything.

    Args:
        tickers: List of portfolio tickers to fetch
        force_refresh: If True, ignore cache and re-download all data
    Returns:
        Price DataFrame
    """
    all_tickers = list(tickers)

    if not force_refresh and os.path.exists(CACHE_FILE):
        cached = pd.read_parquet(CACHE_FILE)

        missing_tickers = [t for t in all_tickers if t not in cached.columns]
        if missing_tickers:
            print(f"  [Cache] New tickers {missing_tickers} — downloading full history...")
            new_cols = _download_with_retry(missing_tickers, START_DATE, END_DATE)
            cached = cached.join(new_cols, how='left')
            cached.to_parquet(CACHE_FILE)
            print(f"  [Cache] New tickers added and saved.")

        last_cached = cached.index[-1].date()
        today = END_DATE.date()

        if last_cached >= today - timedelta(days=1):
            print(f"  [Cache] Up to date ({last_cached}). Loading from {CACHE_FILE}")
            data = cached
        else:
            gap_start = last_cached + timedelta(days=1)
            print(f"  [Cache] Updating from {gap_start} to {today}...")
            new_data = _download_with_retry(all_tickers, gap_start, END_DATE)
            if not new_data.empty:
                data = pd.concat([cached, new_data]).drop_duplicates().sort_index().dropna()
                data.to_parquet(CACHE_FILE)
                print(f"  [Cache] Updated and saved to {CACHE_FILE}")
            else:
                print(f"  [Cache] No new rows available yet, using cached data.")
                data = cached
    else:
        print("  [Cache] No cache found. Downloading full history...")
        data = _download_with_retry(all_tickers, START_DATE, END_DATE)
        data.to_parquet(CACHE_FILE)
        print(f"  [Cache] Prices saved to {CACHE_FILE}")

    return data


def fetch_ff_factors(force_refresh=False):
    """
    Downloads the official Fama-French 3-factor daily data via pandas_datareader.
    Columns returned (all in decimal, not percent): Mkt-RF, SMB, HML, RF

    Results are cached to FF_CACHE_FILE. Re-downloads the full dataset whenever
    the cache is stale (the FF dataset is small, ~few MB).

    Args:
        force_refresh: If True, ignore cache and re-download
    Returns:
        DataFrame indexed by date with columns Mkt-RF, SMB, HML, RF
    """
    if not force_refresh and os.path.exists(FF_CACHE_FILE):
        cached = pd.read_parquet(FF_CACHE_FILE)
        last_cached = cached.index[-1].date()
        today = END_DATE.date()

        if last_cached >= (today - timedelta(days=5)):  # FF data has a few-day publishing lag
            print(f"  [FF Cache] Up to date ({last_cached}). Loading from {FF_CACHE_FILE}")
            return cached

        print(f"  [FF Cache] Stale ({last_cached}). Re-downloading...")

    print("  [FF Cache] Downloading Fama-French daily factors...")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
        with z.open(csv_name) as f:
            # Skip the header description rows until we hit the data
            raw = pd.read_csv(f, skiprows=3, index_col=0)

    # Drop the footer rows (non-date index values)
    raw = raw[pd.to_numeric(raw.index, errors='coerce').notna()]
    raw.index = pd.to_datetime(raw.index, format='%Y%m%d')
    raw.columns = raw.columns.str.strip()

    factors = raw / 100  # convert percent → decimal
    factors = factors.loc[START_DATE:END_DATE]
    factors.to_parquet(FF_CACHE_FILE)
    print(f"  [FF Cache] Saved to {FF_CACHE_FILE}")
    return factors


def compute_returns(raw, ticker, ff_factors, file=None):
    """
    Computes log returns, excess returns, and aligns official FF factor series.

    Args:
        raw: Price DataFrame from fetch_prices
        ticker: Asset ticker string
        ff_factors: DataFrame from fetch_ff_factors (Mkt-RF, SMB, HML, RF in decimal)

    Returns:
        dict with log_ret, excess_ret, MKT, SMB, HML, factor_vols,
              prices, S0, T_hist, rf_daily, rf_ann
    """
    price_series = raw[ticker].dropna()
    log_ret_series = np.log(price_series / price_series.shift(1)).dropna()

    # Align FF factors to the dates we have log returns for
    ff = ff_factors.reindex(log_ret_series.index).dropna()
    common_idx = log_ret_series.index.intersection(ff.index)

    log_ret = log_ret_series.loc[common_idx].values
    MKT = ff.loc[common_idx, 'Mkt-RF'].values
    SMB = ff.loc[common_idx, 'SMB'].values
    HML = ff.loc[common_idx, 'HML'].values
    rf_daily_series = ff.loc[common_idx, 'RF'].values

    rf_daily = rf_daily_series.mean()
    rf_ann = rf_daily * 252
    excess_ret = log_ret - rf_daily_series  # day-specific rf

    S0 = float(price_series.iloc[0])
    T_hist = len(log_ret)
    factor_vols = np.array([MKT.std(), SMB.std(), HML.std()])

    print(f"\n[Returns — {ticker}]", file=file)
    print(f"  Annualised mean return : {log_ret.mean() * 252:.2%}", file=file)
    print(f"  Annualised hist. vol   : {log_ret.std() * np.sqrt(252):.2%}", file=file)
    print(f"  Skewness               : {stats.skew(log_ret):.3f}", file=file)
    print(f"  Excess kurtosis        : {stats.kurtosis(log_ret):.3f}", file=file)
    print(f"  {T_hist} trading days  ({START_DATE:%Y-%m-%d} → {END_DATE:%Y-%m-%d})", file=file)
    print(f"  S0 = {S0:.2f},  S_final = {float(price_series.iloc[-1]):.2f}", file=file)
    print(f"  Avg risk-free (ann.) = {rf_ann:.2%}", file=file)

    return {
        "log_ret": log_ret,
        "log_ret_series": log_ret_series.loc[common_idx],
        "excess_ret": excess_ret,
        "MKT": MKT,
        "SMB": SMB,
        "HML": HML,
        "factor_vols": factor_vols,
        "prices": price_series.values,
        "S0": S0,
        "T_hist": T_hist,
        "rf_daily": rf_daily,
        "rf_ann": rf_ann,
    }


def run_factor_models(excess_ret, MKT, SMB, HML, rf_ann, file=None):
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
    print("\n[Factor Model Regression]", file=file)
    print(f"  {'Parameter':<18} {'Estimate':>10} {'Std Err':>10} {'t-stat':>8} {'p-value':>8}", file=file)
    print("  " + "-" * 60, file=file)
    for i, lbl in enumerate(labels):
        sig = "***" if p_vals[i] < 0.001 else "**" if p_vals[i] < 0.01 else "*" if p_vals[i] < 0.05 else ""
        print(f"  {lbl:<18} {betas[i]:>10.6f} {se[i]:>10.6f} {t_stats[i]:>8.2f} {p_vals[i]:>8.4f} {sig}", file=file)
    print(f"\n  R²      = {R2:.4f}", file=file)
    print(f"  Adj. R² = {R2_adj:.4f}", file=file)

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
    print(f"\n  Estimated annualised mu  : {mu_annual:.2%}", file=file)

    return {
        "alpha_daily": alpha_daily,
        "b_MKT": b_MKT,
        "b_SMB": b_SMB,
        "b_HML": b_HML,
        "mu_annual": mu_annual,
        "residuals": residuals,
        "r_squared": R2,
        "Y_hat": Y_hat,
        "p_val": p_vals[i]
    }


def run_garch(residuals, factor_vols, b_MKT, b_SMB, b_HML, file=None):
    """
    Fits GARCH(1,1) to OLS residuals using the arch library.

    Args:
        residuals: Regression residual array
        factor_vols: Array of [MKT, SMB, HML] daily std devs
        b_MKT, b_SMB, b_HML: Factor loadings from run_factor_models

    Returns:
        dict with h_garch, sigma_t, sigma_total_annual, garch_persist,
              garch_long_run_vol, pvalues
    """
    model = arch_model(residuals * 100, vol='Garch', p=1, q=1, dist='normal', rescale=False)
    res = model.fit(disp='off')

    omega_g = res.params['omega'] / 1e4
    alpha_g_est = res.params['alpha[1]']
    beta_g_est = res.params['beta[1]']
    garch_persist = alpha_g_est + beta_g_est
    garch_long_run_vol = np.sqrt(omega_g / (1 - garch_persist)) * np.sqrt(252)

    h_garch = (res.conditional_volatility / 100) ** 2
    sigma_t = np.sqrt(h_garch)
    sigma_current_daily = sigma_t[-1]
    sigma_total_annual = np.sqrt(
        (b_MKT ** 2 * factor_vols[0] ** 2 + b_SMB ** 2 * factor_vols[1] ** 2 + b_HML ** 2 * factor_vols[2] ** 2) * 252
        + h_garch[-1] * 252
    )

    pvalues = res.pvalues

    print("\n[GARCH(1,1) on Idiosyncratic Residuals]", file=file)
    print(f"  omega               : {omega_g:.2e}  (p={pvalues['omega']:.3f})", file=file)
    print(f"  alpha (ARCH)        : {alpha_g_est:.4f}  (p={pvalues['alpha[1]']:.3f})", file=file)
    print(f"  beta  (GARCH)       : {beta_g_est:.4f}  (p={pvalues['beta[1]']:.3f})", file=file)
    print(f"  Persistence (α+β)   : {garch_persist:.4f}", file=file)
    print(f"  Long-run vol (ann.) : {garch_long_run_vol:.2%}", file=file)
    print(f"  Current cond. vol   : {sigma_current_daily * np.sqrt(252):.2%} (annualised idiosyncratic)", file=file)
    print(f"  Total asset vol     : {sigma_total_annual:.2%} (annualised)", file=file)

    return {
        "h_garch": h_garch,
        "sigma_t": sigma_t,
        "sigma_total_annual": sigma_total_annual,
        "garch_persist": garch_persist,
        "garch_long_run_vol": garch_long_run_vol,
        "pvalues": pvalues,
        "omega_garch": omega_g,
        "alpha_garch": alpha_g_est,
        "beta_garch": beta_g_est,
        "sigma_current_vol": sigma_current_daily      
    }


def run_monte_carlo(mu, sigma, S0, file=None):
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


def fetch_valuation_metrics(ticker, force_refresh=False):
    """
    Fetches valuation metrics for ticker from yfinance, with a daily JSON cache.

    Args:
        ticker: Ticker whose information is required

    Returns:
        dict: Valuation metrics for ticker
    """
    today = str(date.today())

    cache = {}
    if not force_refresh and os.path.exists(VAL_CACHE_FILE):
        with open(VAL_CACHE_FILE, encoding='utf-8') as f:
            cache = json.load(f)

    entry = cache.get(ticker, {})
    if entry.get('date') == today:
        print(f"  [Val Cache] {ticker}: loaded from cache.")
        return entry['metrics']

    for attempt in range(4):
        try:
            session = requests.Session(impersonate="chrome")
            info = yf.Ticker(ticker, session=session).info
            time.sleep(2)
            metrics = {
                'peg': info.get('trailingPegRatio'),
                'fwd_pe': info.get('forwardPE'),
                'ttm_pe': info.get('trailingPE'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'target_price': info.get('targetMeanPrice'),
                'analyst_rec': info.get('recommendationMean'),
                '200MA': info.get('twoHundredDayAverage'),
                '50MA': info.get('fiftyDayAverage'),
                'earnings_growth': info.get('earningsGrowth'),
                'revenue_growth': info.get('revenueGrowth'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
            cache[ticker] = {'date': today, 'metrics': metrics}
            with open(VAL_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
            return metrics
        except Exception as e:
            wait = 30 * (attempt + 1)
            if attempt < 3:
                print(f'  [WARN] {ticker} metrics failed ({e}). Retrying in {wait}s...')
                time.sleep(wait)
            else:
                print(f'  [WARN] {ticker} metrics unavailable after 4 attempts, skipping.')
                return {'peg': None, 'fwd_pe': None, 'ttm_pe': None, 'ev_ebitda': None}


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


def print_recommendation(portfolio_df, opt_weights, scenario_name, file=None):
    """
    Compares optimised weights to current holdings and prints BUY/SELL/HOLD signals.

    Args:
        portfolio_df: DataFrame with 'ticker' and 'weight' columns
        opt_weights: {ticker: target_weight} from run_optimisation
        scenario_name: Label string e.g. 'Max Sharpe'
        file: File object to write to (defaults to stdout)
    """
    current = dict(zip(portfolio_df['Symbol'], portfolio_df['weight']))
    threshold = 0.02

    print(f"\n--- {scenario_name} Rebalancing ---", file=file)
    print(f"{'Ticker':<8} {'Current':>10} {'Target':>10} {'Delta':>10} {'Signal':>8}", file=file)
    for ticker, target in opt_weights.items():
        current_w = current.get(ticker, 0.0)
        delta = target - current_w
        if delta > threshold:
            signal = "BUY"
        elif delta < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        print(f"{ticker:<8} {current_w:>10.1%} {target:>10.1%} {delta:>+10.1%} {signal:>8}", file=file)


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


def plot_outputs(ticker, ret, fm, g, mc, user_path):
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
    path = user_path / "asset_analysis" / filename
    path.mkdir(exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"\n[Done]  Plot saved to {path}")     


def run_user_pipeline(user_path):
    """
    Core Pipeline for a single User.
    """
    connection = init_database(user_path)
    
    try:
        config = load_config(user_path)
        name = config["name"]    
        portfolio_df = load_portfolio(user_path)
        usd_cad = yf.Ticker("USDCAD=X").fast_info['lastPrice']
        
        portfolio_df.loc[portfolio_df['Exchange'] == 'TSX', 'Symbol'] += '.TO'
        print(repr(portfolio_df['Symbol'].iloc[0]))

        tickers = portfolio_df['Symbol'].tolist()
        extra = config.get("extra_tickers", [])
        all_tickers = list(set(tickers + extra)) 
        
        today = date.today().isoformat()
        report_dir = user_path / "reports" / f"{today}"
        report_dir.mkdir(exist_ok=True)
        output_path = report_dir / f"recommendation_{today}.txt"   

        raw_prices = fetch_prices(all_tickers)
        ff_factors = fetch_ff_factors()
        rf_ann = float(ff_factors['RF'].mean() * 252)

        results = []
        db_results = []
        returns_dict = {}   # {ticker: log_ret} for covariance matrix
        mu_dict = {}        # {ticker: mu_annual} for optimisation
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"\n{'='*60}")
            f.write(f"  {name} - {today}")
            f.write(f"{'='*60}")
            for ticker in all_tickers:
                f.write(f"\n{'='*60}")
                f.write(f"  {ticker}")
                f.write(f"{'='*60}")
                try:
                    ret = compute_returns(raw_prices, ticker, ff_factors, file=f)
                    fm = run_factor_models(ret["excess_ret"], ret["MKT"], ret["SMB"], ret["HML"], ret["rf_ann"], file=f)
                    g = run_garch(fm["residuals"], ret["factor_vols"], fm["b_MKT"], fm["b_SMB"], fm["b_HML"], file=f)
                    mc = run_monte_carlo(fm["mu_annual"], g["sigma_total_annual"], float(raw_prices[ticker].iloc[-1]), file=f)
                    val = fetch_valuation_metrics(ticker)
                    score, conf = compute_confidence_score(ret["T_hist"], fm["r_squared"], g["garch_persist"])

                    plot_outputs(ticker, ret, fm, g, mc, report_dir)

                    returns_dict[ticker] = ret["log_ret_series"]
                    mu_dict[ticker] = fm["mu_annual"]
                    portfolio_df.loc[portfolio_df['Symbol'] == ticker, 'Industry'] = val.get('industry')

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
                    
                    db_results.append({
                        # Factor Models Info
                        "alpha_daily": fm["alpha_daily"],
                        "beta_mkt": fm["b_MKT"],
                        "beta_smb": fm["b_SMB"],
                        "beta_hml": fm["b_HML"],
                        "r_squared": fm["r_squared"],
                        "alpha_pval": fm["p_val"],
                        # GARCH Info
                        "garch_omega": g["omega_garch"],
                        "garch_alpha": g["alpha_garch"],
                        "garch_beta": g["beta_garch"],
                        "garch_persistence": g["garch_persist"],
                        "garch_longrun_vol": g["garch_long_run_vol"],
                        "garch_current_vol": g["sigma_current_vol"],
                        # Annulaised Exp. Return and Exp. Vol
                        "mu_annual": fm["mu_annual"],
                        "sigma_annual": g["sigma_total_annual"],
                        # Monte Carlo Info
                        "mc_p05": np.percentile(mc["paths"], 5, axis=1),
                        "mc_p25": np.percentile(mc["paths"], 25, axis=1),
                        "mc_p50": np.percentile(mc["paths"], 50, axis=1),
                        "mc_p75": np.percentile(mc["paths"], 75, axis=1),
                        "mc_p95": np.percentile(mc["paths"], 95, axis=1),
                        "mc_var95": mc["VaR_95"],
                        "mc_cvar95": mc["CVaR_95"],
                        #  Valutation Info
                        "forward_pe": val["fwd_pe"],
                        "ttm_pe": val["ttm_pe"],
                        "peg_ratio": val["peg"],
                        "ev_ebitda": val["ev_ebitda"],
                        "target_price": val["target_price"],
                        "analyst_rec": val["analyst_rec"],
                        "ma_200": val["200MA"],
                        "ma_50": val["50MA"],
                        "earnings_growth": val["earnings_growth"],
                        "revenue_growth": val["revenue_growth"],
                        "sector": val["sector"],
                        "industry": val["industry"],
                        "cape": cape
                    })
                    
                    insert_model_output()

                except Exception as e:
                    f.write(f"  [WARN] {ticker} failed: {e}")

            # ── Compute current market weights from shares × latest price ────────────
            total_value = portfolio_df['Book Value (CAD)'].sum()
            portfolio_df['Weight'] = portfolio_df['Book Value (CAD)'] / total_value
            portfolio_df['Average Cost'] = portfolio_df['Book Value (CAD)'] / portfolio_df['Quantity']
            portfolio_df['Market Price (CAD)'] = portfolio_df.apply(
                lambda r: r['Market Price'] / usd_cad if r['Market Price Currency'] == 'USD' else r['Market Price'], axis=1
            )

            # ── Write Ticker Items to Database ───────────────────────────────────────
            for _, row in portfolio_df.iterrows():
                insert_portfolio_row(connection, END_DATE, row["Symbol"], row["Quantity"], row["Average Cost"],
                                        row["Industry"], row["Market Price (CAD)"], row["Weight"])
                

            # ── Portfolio-level optimisation ─────────────────────────────────────────
            if len(returns_dict) >= 2:
                f.write(f"\n{'='*60}")
                f.write("  Portfolio Optimisation")
                f.write(f"{'='*60}")
                cov_matrix = build_covariance(returns_dict)
                opt_weights = run_optimisation(mu_dict, cov_matrix, rf_ann)
                print_recommendation(portfolio_df, opt_weights["max_sharpe"], "Max Sharpe", file=f)
                insert_recommendation(connection, END_DATE, ticker, "Max Sharpe", current_w, target, signal, )
                print_recommendation(portfolio_df, opt_weights["min_vol"], "Min Volatility", file=f)
            else:
                pprint.pprint(returns_dict)
                print("\n[WARN] Need at least 2 tickers for portfolio optimisation.")

            # ── Per-ticker summary table ──────────────────────────────────────────────
            if results:
                df_results = pd.DataFrame(results)
                float_cols = df_results.select_dtypes(include='float').columns
                df_results[float_cols] = df_results[float_cols].round(2)
                f.write('\n')
                f.write(df_results.to_string(index=False))
    finally:
        connection.close()


def main():
    users = discover_users()
    results = {"success": [], "failed": []}
    
    for user_path in users:
        try:
            run_user_pipeline(user_path)
            results["success"].append(user_path.name)
        except Exception as e:
            print(f"\n  ✗ FAILED for {user_path.name}: {e}")
            traceback.print_exc()
            results["failed"].append(user_path.name)
    
    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"  Success: {results['success']}")
    print(f"  Failed:  {results['failed']}") 


if __name__ == "__main__":
    main()
