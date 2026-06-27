"""
Agent 2 — Quantitative Analysis
=========================
Runs all quantitative analysis per ticker:
  - Fama-French 3-factor OLS regression  → alpha, betas, R²
  - GARCH(1,1) volatility estimation     → time-varying sigma
  - mu and sigma per ticker              → Agent 4 inputs
  - Valuation metrics                    → P/E, PEG, EV/EBITDA
  - Financial health                     → FCF, ROIC, margins
  - Earnings data                        → beat/miss history
  - Earnings dates                       → next report date
  - Ticker classification                → ETF / US stock / CA stock

All yfinance calls live here. Agent 3 reads everything for BL view formation and optimisation..
Agent 4 reads mu_sigma for simulation inputs.

"""

import os
import json
import time
import re
from datetime import date, datetime, timedelta
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import numpy as np
import requests, zipfile, io
from bs4 import BeautifulSoup
import yfinance as yf
from arch import arch_model
from curl_cffi import requests
from dotenv import load_dotenv

from agents.state import PipelineState
from agents.llm import get_llm
from valuation_eval.valuation import compute_valuation_signal

# Constants
R2_LOW_THRESHOLD       = 0.15
GARCH_HIGH_PERSISTENCE = 0.97
PEG_HIGH_THRESHOLD     = 3.0
PEG_LOW_THRESHOLD      = 0.5
CACHE_FILE = "cache/prices_cache.parquet"
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
FF_CACHE_FILE = "cache/ff_factors_cache.parquet"
os.makedirs(os.path.dirname(FF_CACHE_FILE), exist_ok=True)
VAL_CACHE_FILE = "cache/valuation_cache.json"
os.makedirs(os.path.dirname(VAL_CACHE_FILE), exist_ok=True)
HISTORY_YEARS = 5
HEADERS = {
	"User-Agent": (
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
		"AppleWebKit/537.36 (KHTML, like Gecko) "
		"Chrome/120.0.0.0 Safari/537.36"
	),
	"Accept-Language": "en-US,en;q=0.9",
}


def _to_yfinance_ticker(ticker: str) -> str:
	"""Yahoo Finance uses hyphens where TSX uses dots in base symbols (e.g. BEP.UN.TO → BEP-UN.TO)."""
	if ticker.endswith(".TO"):
		base = ticker[:-3].replace(".", "-")
		return base + ".TO"
	return ticker


def _download_with_retry(tickers: list[str], start: date, end: date) -> pd.DataFrame:
	"""
	Downloads daily close prices for the given tickers from yfinance.

	Retries up to 5 times with linear backoff (30s, 60s, 90s, 120s, 150s) to
	handle rate limiting. Uses a curl_cffi Chrome session to reduce blocking.

	Args:
		tickers: List of yfinance-compatible ticker symbols.
		start: Start date for the price history download.
		end: End date for the price history download.

	Returns:
		DataFrame of adjusted close prices indexed by date, one column per ticker.

	Raises:
		RuntimeError: If all 5 download attempts fail.
	"""
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


def _download_translated(portfolio_tickers: list[str], start: date, end: date) -> pd.DataFrame:
	"""Download prices using Yahoo Finance symbol format, then rename columns back to portfolio format."""
	yf_tickers  = [_to_yfinance_ticker(t) for t in portfolio_tickers]
	back_map    = {yf: orig for orig, yf in zip(portfolio_tickers, yf_tickers)}
	df = _download_with_retry(yf_tickers, start, end)
	if df.index.tz is not None:
		df.index = df.index.tz_localize(None)
	return df.rename(columns=back_map)


def fetch_prices(tickers: list[str], start_date: date, today: date, force_refresh: bool = False) -> pd.DataFrame:
	"""
	Fetches adjusted close prices for the given tickers with disk caching.

	On first run, downloads the full history and saves to CACHE_FILE. On
	subsequent runs, only the missing date window is fetched and appended.
	New tickers added to an existing portfolio trigger a full-history download
	for those tickers only.

	Args:
		tickers: List of yfinance-compatible ticker symbols.
		start_date: Earliest date to include in the price history.
		today: Most recent date to fetch (typically the run date).
		force_refresh: If True, ignores the cache and re-downloads all data.

	Returns:
		DataFrame of adjusted close prices indexed by date, one column per ticker.
	"""
	if isinstance(today, str):
		today = date.fromisoformat(today)
	if isinstance(start_date, str):
		start_date = date.fromisoformat(start_date)
	all_tickers = list(tickers)

	if not force_refresh and os.path.exists(CACHE_FILE):
		cached = pd.read_parquet(CACHE_FILE)

		# If all price entries are NaN then remove column
		stale = [t for t in cached.columns if cached[t].isna().all()]
		if stale:
			print(f"  [Cache] Dropping all-NaN columns for re-download: {stale}")
			cached = cached.drop(columns=stale)

		last_cached = cached.index[-1].date()

		# If new tickers added then only download the new ones
		missing_tickers = [t for t in all_tickers if t not in cached.columns]
		if missing_tickers:
			print(f"  [Cache] New tickers {missing_tickers} — downloading full history...")
			new_cols = _download_translated(missing_tickers, start_date, last_cached  + timedelta(days=1))
			cached = cached.join(new_cols, how='left')
			still_null = [t for t in missing_tickers if t in cached.columns and cached[t].isna().all()]
			if still_null:
				print(f"  [Cache] WARNING: {still_null} are still all-NaN after download — symbol may be invalid.")
			cached.to_parquet(CACHE_FILE)
			print(f"  [Cache] New tickers added and saved.")

		if last_cached >= today - timedelta(days=1):
			print(f"  [Cache] Up to date ({last_cached}). Loading from {CACHE_FILE}")
			data = cached
		else:
			gap_start = last_cached + timedelta(days=1)
			print(f"  [Cache] Updating from {gap_start} to {today}...")
			new_data = _download_translated(all_tickers, gap_start, today + timedelta(days=1))
			if not new_data.empty:
				data = pd.concat([cached, new_data]).drop_duplicates().sort_index().dropna(how='all')
				data.to_parquet(CACHE_FILE)
				print(f"  [Cache] Updated and saved to {CACHE_FILE}")
			else:
				print(f"  [Cache] No new rows available yet, using cached data.")
				data = cached
	else:
		print("  [Cache] No cache found. Downloading full history...")
		data = _download_translated(all_tickers, start_date, today + timedelta(days=1))
		data.to_parquet(CACHE_FILE)
		print(f"  [Cache] Prices saved to {CACHE_FILE}")

	return data


def fetch_ff_factors(today: date, start: date, force_refresh: bool = False) -> pd.DataFrame:
	"""
	Downloads the Fama-French 3-factor daily data from Ken French's data library.

	Fetches the F-F_Research_Data_Factors_daily_CSV.zip, parses it, and converts
	from percent to decimal. Results are cached to FF_CACHE_FILE and refreshed
	when stale (FF data has a few-day publishing lag).

	Args:
		today: Current run date, used to assess cache staleness.
		start: Earliest date to include after trimming the downloaded dataset.
		force_refresh: If True, ignores the cache and re-downloads.

	Returns:
		DataFrame indexed by date with columns: Mkt-RF, SMB, HML, RF (all decimal).
	"""
	if not force_refresh and os.path.exists(FF_CACHE_FILE):
		cached = pd.read_parquet(FF_CACHE_FILE)
		last_cached = cached.index[-1].date()

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
	factors = factors.loc[start:today]
	factors.to_parquet(FF_CACHE_FILE)
	print(f"  [FF Cache] Saved to {FF_CACHE_FILE}")
	return factors


def compute_returns(raw: pd.DataFrame, ticker: str, today: date, start: date, ff_factors: pd.DataFrame, file: io.IOBase | None = None) -> dict:
	"""
	Computes log returns, excess returns, and aligns Fama-French factor series.

	Calculates daily log returns from price data, subtracts the day-specific
	risk-free rate to get excess returns, and aligns with FF factor dates.

	Args:
		raw: Price DataFrame from fetch_prices, indexed by date.
		ticker: Ticker symbol to extract from the price DataFrame.
		today: End date of the history window (used for log output only).
		start: Start date of the history window (used for log output only).
		ff_factors: DataFrame from fetch_ff_factors with Mkt-RF, SMB, HML, RF columns.
		file: Optional file object to redirect printed diagnostics. Defaults to stdout.

	Returns:
		Dict with keys: log_ret, log_ret_series, excess_ret, MKT, SMB, HML,
		factor_vols, prices, S0, T_hist, rf_daily, rf_ann.
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
	print(f"  {T_hist} trading days  ({start:%Y-%m-%d} → {today:%Y-%m-%d})", file=file)
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


def run_factor_models(excess_ret: np.ndarray, MKT: np.ndarray, SMB: np.ndarray, HML: np.ndarray, rf_ann: float, cape: float, real_rf: float, file: io.IOBase | None = None) -> dict:
	"""
	Fits a Fama-French 3-factor OLS regression on daily excess returns.

	Estimates alpha and factor loadings via closed-form OLS, computes p-values
	via t-distribution, and blends a CAPE-adjusted ERP with the historical
	factor mean to project an annualised expected return.

	Args:
		excess_ret: Daily excess return array (log return minus daily RF).
		MKT: Daily Mkt-RF factor array.
		SMB: Daily SMB factor array.
		HML: Daily HML factor array.
		rf_ann: Annualised risk-free rate (mean of daily RF * 252).
		cape: Current Shiller CAPE ratio, used to compute the CAPE-based ERP.
		real_rf: 10-year TIPS real yield used as the real risk-free rate in the
			CAPE ERP formula. Replaces the nominal rf minus inflation approximation.
		file: Optional file object to redirect printed diagnostics. Defaults to stdout.

	Returns:
		Dict with keys: alpha_daily, b_MKT, b_SMB, b_HML, mu_annual,
		residuals, r2, r2_adj, Y_hat, p_val_alpha, rf_ann.
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
	
	historical_erp = factor_annual_means[0]
	cape_weight = 0.6           # TUNABLE PARAM
	cape_erp = max(0.0, (1 / cape) - real_rf)
	blended_erp = cape_weight * cape_erp + (1 - cape_weight) * historical_erp

	mu_annual = (
		rf_ann
		+ alpha_daily * 252
		+ b_MKT * blended_erp
		+ b_SMB * factor_annual_means[1]
		+ b_HML * factor_annual_means[2]
	)

	print(f"\n  Real risk-free (TIPS)        : {real_rf:.2%}", file=file)
	print(f"  CAPE earnings yield (1/CAPE) : {1/cape:.2%}", file=file)
	print(f"  CAPE ERP                     : {cape_erp:.2%}", file=file)
	print(f"  Historical ERP (5yr MKT avg) : {historical_erp:.2%}", file=file)
	print(f"  Blended ERP (w={cape_weight:.0%} CAPE)   : {blended_erp:.2%}", file=file)
	print(f"  Estimated annualised mu      : {mu_annual:.2%}", file=file)

	return {
		"alpha_daily": alpha_daily,
		"b_MKT": b_MKT,
		"b_SMB": b_SMB,
		"b_HML": b_HML,
		"mu_annual": mu_annual,
		"residuals": residuals,
		"r2": R2,
		"r2_adj": R2_adj,
		"Y_hat": Y_hat,
		"p_val_alpha": p_vals[0],
		"rf_ann": rf_ann,
	}
	

def run_garch(residuals: np.ndarray, factor_vols: np.ndarray, b_MKT: float, b_SMB: float, b_HML: float, file: io.IOBase | None = None) -> dict:
	"""
	Fits a GARCH(1,1) model to OLS residuals to estimate idiosyncratic volatility.

	Combines the one-step-ahead GARCH variance forecast with factor variance
	to produce a total annualised sigma for use in simulation and optimisation.

	Args:
		residuals: Daily idiosyncratic return residuals from run_factor_models.
		factor_vols: Array of [MKT, SMB, HML] daily standard deviations.
		b_MKT: Market factor loading from the OLS regression.
		b_SMB: SMB factor loading from the OLS regression.
		b_HML: HML factor loading from the OLS regression.
		file: Optional file object to redirect printed diagnostics. Defaults to stdout.

	Returns:
		Dict with keys: h_garch, sigma_total_annual, garch_persist,
		garch_long_run_vol, pvalues, omega_garch, alpha_garch, beta_garch,
		sigma_current_vol, vol_regime.
	"""
	model = arch_model(residuals * 100, vol='Garch', p=1, q=1, dist='normal', rescale=False)
	res = model.fit(disp='off')
	garch_converged = res.convergence_flag == 0

	omega_g = res.params['omega'] / 1e4
	alpha_g_est = res.params['alpha[1]']
	beta_g_est = res.params['beta[1]']
	garch_persist = alpha_g_est + beta_g_est
	garch_long_run_vol = np.sqrt(omega_g / max(1e-6, 1 - garch_persist)) * np.sqrt(252)

	h_garch = (res.conditional_volatility / 100) ** 2

	forecast = res.forecast(horizon=1, reindex=False)
	h_next = forecast.variance.values[-1, 0] / 1e4
	sigma_current_daily = np.sqrt(h_next)
	sigma_total_annual = np.sqrt(
		(b_MKT ** 2 * factor_vols[0] ** 2 + b_SMB ** 2 * factor_vols[1] ** 2 + b_HML ** 2 * factor_vols[2] ** 2) * 252
		+ h_next * 252
	)

	pvalues = res.pvalues

	print("\n[GARCH(1,1) on Idiosyncratic Residuals]", file=file)
	if not garch_converged:
		print(f"  [warn] GARCH did not converge (code {res.convergence_flag}) — estimates may be unreliable", file=file)
	print(f"  omega               : {omega_g:.2e}  (p={pvalues['omega']:.3f})", file=file)
	print(f"  alpha (ARCH)        : {alpha_g_est:.4f}  (p={pvalues['alpha[1]']:.3f})", file=file)
	print(f"  beta  (GARCH)       : {beta_g_est:.4f}  (p={pvalues['beta[1]']:.3f})", file=file)
	print(f"  Persistence (α+β)   : {garch_persist:.4f}", file=file)
	print(f"  Long-run vol (ann.) : {garch_long_run_vol:.2%}", file=file)
	print(f"  Current cond. vol   : {sigma_current_daily * np.sqrt(252):.2%} (annualised idiosyncratic)", file=file)
	print(f"  Total asset vol     : {sigma_total_annual:.2%} (annualised)", file=file)

	return {
		"h_garch": h_garch,
		"sigma_total_annual": sigma_total_annual,
		"garch_persist": garch_persist,
		"garch_long_run_vol": garch_long_run_vol,
		"pvalues": pvalues,
		"omega_garch": omega_g,
		"alpha_garch": alpha_g_est,
		"beta_garch": beta_g_est,
		"sigma_current_vol": sigma_current_daily,
		"vol_regime": "elevated" if garch_persist > 0.97 else "normal",
		"garch_converged": garch_converged,
	}
	

def fetch_valuation_metrics(ticker: str, force_refresh: bool = False) -> dict:
	"""
	Fetches valuation metrics for a ticker from yfinance with a daily JSON cache.

	Retries up to 4 times with linear backoff on failure. Caches results in
	VAL_CACHE_FILE keyed by ticker and today's date so each ticker is only
	fetched once per run.

	Args:
		ticker: yfinance-compatible ticker symbol.
		force_refresh: If True, bypasses the cache and re-fetches from yfinance.

	Returns:
		Dict with keys: peg, fwd_pe, ttm_pe, ev_ebitda, target_price,
		analyst_rec, 200MA, 50MA, sector, industry. Values are None if
		the field is unavailable.
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
			info = yf.Ticker(_to_yfinance_ticker(ticker), session=session).info
			time.sleep(2)
			metrics = {
				'market_cap': info.get('marketCap'),
				'peg': info.get('trailingPegRatio'),
				'fwd_pe': info.get('forwardPE'),
				'ttm_pe': info.get('trailingPE'),
				'ev_ebitda': info.get('enterpriseToEbitda'),
				'target_price': info.get('targetMeanPrice'),
				'analyst_rec': {1: 'Strong Buy', 2: 'Buy', 3: 'Hold', 4: 'Underperform', 5: 'Sell'}.get(
					round(info.get('recommendationMean') or 0) or None
				),
				'earnings_growth': info.get('earningsGrowth'),
				'revenue_growth': info.get('revenueGrowth'),
				'200MA': info.get('twoHundredDayAverage'),
				'50MA': info.get('fiftyDayAverage'),
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
				return {'peg': None, 'fwd_pe': None, 'ttm_pe': None, 'ev_ebitda': None,
						'target_price': None, 'analyst_rec': None, '200MA': None, '50MA': None,
						'sector': None, 'industry': None}
			

def _pct_change(current: float, previous: float) -> float | None:
	"""
	Computes percentage change from previous to current.

	Args:
		current: The more recent value.
		previous: The earlier value used as the base.

	Returns:
		Percentage change as a decimal (e.g. 0.05 for 5%), rounded to 4 decimal
		places. Returns None if previous is zero, None, or conversion fails.
	"""
	try:
		if previous and previous != 0:
			return round((float(current) - float(previous)) / abs(float(previous)), 4)
	except Exception:
		pass
	return None


def _cagr(current: float, past: float, years: int) -> float | None:
	"""
	Computes compound annual growth rate from past to current over the given years.

	Args:
		current: The ending value.
		past: The starting value (must be positive).
		years: Number of years over which to compute the CAGR.

	Returns:
		CAGR as a decimal (e.g. 0.10 for 10%), rounded to 4 decimal places.
		Returns None if past is non-positive or conversion fails.
	"""
	try:
		if past and past > 0 and years > 0:
			return round((float(current) / float(past)) ** (1 / years) - 1, 4)
	except Exception:
		pass
	return None


def fetch_financial_health(ticker: str) -> dict:
	"""
	Fetches multi-year financial statement data and computes derived quality metrics.

	Uses yfinance's .financials, .cashflow, and .balance_sheet properties to
	compute revenue growth, gross margin trend, FCF metrics, earnings quality,
	debt coverage, and ROIC. Each metric is only included if the required line
	items are present in the statements.

	Args:
		ticker: yfinance-compatible ticker symbol.

	Returns:
		Dict of computed metrics. May include: revenue_growth_1yr,
		revenue_growth_3yr, revenue_growth_5yr, gross_margin_current,
		gross_margin_expanding, gross_margin_trend, fcf_current,
		fcf_growth_1yr, fcf_growth_3yr, fcf_growth_5yr, fcf_margin,
		earnings_quality, debt_to_fcf, roic. Returns empty dict on failure.
	"""
	try:
		stock    = yf.Ticker(_to_yfinance_ticker(ticker))
		income   = stock.financials        # annual, up to 4 years
		cashflow = stock.cashflow
		balance  = stock.balance_sheet
		
		financial_health = {}
		
		# Revenue growth
		if "Total Revenue" in income.index and income.shape[1] >= 2:
			rev = income.loc["Total Revenue"]
			financial_health["revenue_growth_1yr"] = _pct_change(rev.iloc[0], rev.iloc[1])
			if income.shape[1] >= 4:
				financial_health["revenue_growth_3yr"] = _cagr(rev.iloc[0], rev.iloc[3], years=3)
			if income.shape[1] >= 6:
				financial_health["revenue_growth_5yr"] = _cagr(rev.iloc[0], rev.iloc[5], years=5)
				
		# Gross margin trend
		if all(r in income.index for r in ["Gross Profit", "Total Revenue"]):
			gp  = income.loc["Gross Profit"]
			rev = income.loc["Total Revenue"]
			margins = gp / rev
			financial_health["gross_margin_current"]   = round(float(margins.iloc[0]), 4)
			financial_health["gross_margin_expanding"] = bool(len(margins) > 1 and margins.iloc[0] > margins.iloc[1])
			if len(margins) >= 3:
				# Check if each year is higher than the next
				financial_health["gross_margin_trend"] = "expanding" if (
					margins.iloc[0] > margins.iloc[1] > margins.iloc[2]
				) else "contracting" if (
					margins.iloc[0] < margins.iloc[1] < margins.iloc[2]
				) else "mixed"
				
		# Free cash flow
		if all(r in cashflow.index for r in ["Operating Cash Flow", "Capital Expenditure"]):
			ocf   = cashflow.loc["Operating Cash Flow"]
			capex = cashflow.loc["Capital Expenditure"]
			fcf   = ocf + capex   # capex is negative in yfinance convention

			financial_health["fcf_current"]    = float(fcf.iloc[0])
			financial_health["fcf_growth_1yr"] = _pct_change(fcf.iloc[0], fcf.iloc[1]) if len(fcf) > 1 else None
			financial_health["fcf_growth_3yr"] = _cagr(fcf.iloc[0], fcf.iloc[3], years=3) if len(fcf) > 3 else None
			financial_health["fcf_growth_5yr"] = _cagr(fcf.iloc[0], fcf.iloc[5], years=5) if len(fcf) > 5 else None

			if "Total Revenue" in income.index:
				rev = income.loc["Total Revenue"]
				financial_health["fcf_margin"] = round(float(fcf.iloc[0] / rev.iloc[0]), 4) if rev.iloc[0] != 0 else None
				
		# Earnings quality — OCF vs net income
		# Consistently above 1.0 means profits are backed by real cash
		if "Operating Cash Flow" in cashflow.index and "Net Income" in income.index:
			ocf_val = float(cashflow.loc["Operating Cash Flow"].iloc[0])
			ni_val  = float(income.loc["Net Income"].iloc[0])
			if ni_val != 0:
				financial_health["earnings_quality"] = round(ocf_val / ni_val, 2)
				
		# Debt to FCF — years of FCF needed to pay off debt
		if "Total Debt" in balance.index and financial_health.get("fcf_current", 0) > 0:
			debt = float(balance.loc["Total Debt"].iloc[0])
			financial_health["debt_to_fcf"] = round(debt / financial_health["fcf_current"], 2)
			
		# ROIC — return on invested capital
		# Measures how efficiently management deploys capital
		if "Operating Income" in income.index and \
		   all(r in balance.index for r in ["Total Assets", "Total Current Liabilities"]):
			op_inc    = float(income.loc["Operating Income"].iloc[0])
			assets    = float(balance.loc["Total Assets"].iloc[0])
			curr_liab = float(balance.loc["Total Current Liabilities"].iloc[0])
			inv_cap   = assets - curr_liab
			if inv_cap > 0:
				nopat = op_inc * (1 - 0.21)   # approximate 21% tax rate
				financial_health["roic"] = round(nopat / inv_cap, 4)

		return financial_health

	except Exception as e:
		print(f"    [warn] Financial health fetch failed for {ticker}: {e}")
		return {}


def fetch_cape() -> float:
	"""
	Fetches the current Shiller CAPE ratio from multpl.com.

	Validates the scraped value is within the plausible range of 10–60.
	Falls back to 25.0 (a neutral market estimate) on any request or parse failure.

	Returns:
		Current CAPE ratio as a float. Returns 25.0 if the request fails or
		the parsed value is outside the plausible range.
	"""
	url = "https://www.multpl.com/shiller-pe"
	
	try:
		response = requests.get(
			url,
			headers={"User-Agent": "Mozilla/5.0"},
			timeout=10
		)
		response.raise_for_status()
		
		soup = BeautifulSoup(response.text, "html.parser")
		element = soup.find(id="current")
		if element is None:
			raise ValueError("CAPE element #current not found in page")
		raw_text = element.text.strip()
		match = re.search(r"\d+\.\d+", raw_text)
		if not match:
			raise ValueError(f"No float found in CAPE element: {raw_text!r}")
		cape = float(match.group())
		if not (10 < cape < 60):
			raise ValueError(f"CAPE value {cape} outside plausible range")
		
		return cape
	
	except Exception as e:
		print(f"  [CAPE] Failed to fetch: {e}. Using fallback value of 25.0")
		return 25.0


def fetch_real_rf() -> float:
	"""
	Fetches the 10-year TIPS real yield from FRED (series DFII10).

	The TIPS real yield is the market-implied real risk-free rate, derived from
	the spread between 10-year nominal Treasuries and inflation-protected bonds.
	Using it directly in the CAPE ERP formula eliminates the need to assume an
	inflation rate. Requires the FRED_API_KEY environment variable.

	Returns:
		10-year TIPS real yield as a decimal (e.g. 0.02 for 2.0%).
		Falls back to 0.02 (2%) if the key is missing or request fails.
	"""
	api_key = os.getenv("FRED_API_KEY")
	if not api_key:
		print("  [Real RF] FRED_API_KEY not set. Using fallback of 2%")
		return 0.02

	url = "https://api.stlouisfed.org/fred/series/observations"
	params = {
		"series_id":  "DFII10",
		"api_key":    api_key,
		"file_type":  "json",
		"sort_order": "desc",
		"limit":      5,        # grab a few to skip weekends/holidays with missing data
	}
	try:
		response = requests.get(url, params=params, timeout=10)
		response.raise_for_status()
		for obs in response.json()["observations"]:
			if obs["value"] != ".":
				real_rf = float(obs["value"]) / 100
				print(f"  [Real RF] TIPS 10yr real yield ({obs['date']}): {real_rf:.2%}")
				return real_rf
		raise ValueError("No valid DFII10 observations in response")
	except Exception as e:
		print(f"  [Real RF] FRED fetch failed ({e}). Using fallback of 2%")
		return 0.02


def fetch_earnings_data(ticker: str) -> dict:
	"""
	Fetches earnings history and upcoming estimate data via yfinance.

	Retrieves actual vs estimated EPS per quarter, computes surprise
	percentages where not already provided by yfinance, and identifies
	the next upcoming earnings date.

	Args:
		ticker: yfinance-compatible ticker symbol.

	Returns:
		Dict with keys: next_earnings_date (str or None), recent_quarters
		(list of dicts with date, eps_estimate, eps_actual, surprise_pct),
		avg_eps_surprise_pct (float or None), consecutive_beats (int or None).
		Returns a stub dict with None/empty values on failure.
	"""
	try:        
		# Earnings history — actual vs estimated EPS per quarter
		earnings_hist = yf.Ticker(_to_yfinance_ticker(ticker)).earnings_dates
		
		if earnings_hist is not None and not earnings_hist.empty:
			# Most recent 12 quarters
			recent = earnings_hist.head(13).reset_index()
			
			# Next upcoming earnings — first row where date is in the future
			today = pd.Timestamp.today(tz="UTC")
			future = recent[recent["Earnings Date"] > today]
			past   = recent[recent["Earnings Date"] <= today]
			next_date = str(future.iloc[0]["Earnings Date"].date()) if not future.empty else None
			
			quarters = []
			for _, row in past.iterrows():
				eps_estimate = row.get("EPS Estimate")
				eps_actual   = row.get("Reported EPS")
				
				# Calculate surprise if both values exist
				surprise_pct = row.get("Surprise(%)")
				if (surprise_pct is None or pd.isna(surprise_pct)) and eps_estimate and eps_actual and eps_estimate != 0:
					surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
				
				quarters.append({
					"date":          str(row.get("Earnings Date", "")),
					"eps_estimate":  float(eps_estimate) if eps_estimate else None,
					"eps_actual":    float(eps_actual) if eps_actual else None,
					"surprise_pct":  round(surprise_pct, 2) if surprise_pct else None,
				})
			
			# Summarise the beat/miss pattern
			surprises = [q["surprise_pct"] for q in quarters[1:] if q["surprise_pct"] is not None and pd.notna(q["surprise_pct"])]
			avg_surprise = round(sum(surprises) / len(surprises), 2) if surprises else None
			
			return {
				"next_earnings_date": next_date,
				"recent_quarters": quarters,
				"avg_eps_surprise_pct": avg_surprise,
				"consecutive_beats": _count_consecutive_beats(quarters),
			}
	
	except Exception as e:
		print(f"    [warn] Earnings data fetch failed for {ticker}: {e}")
	
	return {"next_earnings_date": None, "recent_quarters": [], "avg_eps_surprise_pct": None, "consecutive_beats": None}


def _count_consecutive_beats(quarters: list[dict]) -> int:
	"""
	Counts consecutive quarters with positive EPS surprises, starting from most recent.

	Args:
		quarters: List of quarter dicts each with a 'surprise_pct' key, expected
			in chronological descending order (most recent first).

	Returns:
		Count of consecutive leading quarters where surprise_pct > 0.
		Returns 0 if the most recent quarter missed or has no surprise data.
	"""
	count = 0
	for q in quarters:
		if q["surprise_pct"] is not None and q["surprise_pct"] > 0:
			count += 1
		else:
			break
	return count


# LLM ANOMALY FLAGGING
ANOMALY_PROMPT = """You are a quantitative analyst reviewing model outputs for a portfolio.

For each ticker below, identify any anomalies or noteworthy signals in the 
quantitative data and provide a brief interpretation.

Flag these specific conditions if present:
- R² below {r2_threshold}: factor model explains little of the stock's returns
- GARCH persistence above {garch_threshold}: volatility is near-nonstationary, 
  mean reversion will be slow
- PEG above {peg_high}: expensive relative to growth rate
- PEG below {peg_low}: either very cheap or growth estimates are unreliable
- Negative FCF: company is cash flow negative
- Earnings quality below 0.8: reported profits not backed by cash
- High debt-to-FCF (above 5): meaningful debt burden

Data:
{data}

For each ticker write 2-3 sentences maximum. Be specific — name the metric 
and the value. Note where sentiment and fundamentals point in opposite 
directions if that data is available. Return plain prose, one paragraph per ticker,
labelled with the ticker symbol."""


def _fmt(val: float | None, spec: str) -> str:
	"""
	Formats a numeric value with a Python format spec, returning 'N/A' for None.

	Args:
		val: Numeric value to format, or None.
		spec: Python format spec string (e.g. '.2%', '.3f', '.6f').

	Returns:
		Formatted string, or 'N/A' if val is None.
	"""
	if val is None:
		return "N/A"
	return format(val, spec)


def generate_quant_commentary(
		tickers: list[str],
		factor_results: dict,
		garch_results: dict,
		valuation: dict,
		financial_health: dict,
		earnings_data: dict,
		file = None,
	) -> str:
	"""
	Calls the LLM to flag anomalies and interpret quantitative outputs per ticker.

	Builds a structured metric summary for each ticker and passes it to
	ANOMALY_PROMPT, which asks the LLM to flag R², GARCH persistence, PEG,
	FCF, earnings quality, and debt-to-FCF signals.

	Args:
		tickers: List of ticker symbols to include in the commentary.
		factor_results: Dict of factor model outputs keyed by ticker.
		garch_results: Dict of GARCH outputs keyed by ticker.
		valuation: Dict of valuation metrics keyed by ticker.
		financial_health: Dict of financial health metrics keyed by ticker.
		earnings_data: Dict of earnings history data keyed by ticker.

	Returns:
		Plain prose commentary, one paragraph per ticker labelled with the
		ticker symbol. Returns a fallback string if the LLM call fails.
	"""
	# Build a structured summary of all metrics per ticker
	data_lines = []
	for ticker in tickers:
		fr  = factor_results.get(ticker, {})
		gr  = garch_results.get(ticker, {})
		val = valuation.get(ticker, {})
		fh  = financial_health.get(ticker, {})
		ed  = earnings_data.get(ticker, {})

		data_lines.append(f"""{ticker}:
		Factor model: alpha={_fmt(fr.get('alpha_daily'), '.6f')}, \
			b_mkt={_fmt(fr.get('b_MKT'), '.3f')}, R²={_fmt(fr.get('r2'), '.3f')}, \
			p_val_alpha={_fmt(fr.get('p_val_alpha'), '.3f')}
		GARCH: persistence={_fmt(gr.get('garch_persist'), '.4f')}, \
			sigma_annual={_fmt(gr.get('sigma_total_annual'), '.2%')}, \
			vol_regime={gr.get('vol_regime', 'N/A')}
		Valuation: fwd_pe={val.get('fwd_pe', 'N/A')}, \
			peg={val.get('peg', 'N/A')}, ev_ebitda={val.get('ev_ebitda', 'N/A')}
		Financial health: roic={_fmt(fh.get('roic'), '.4f')}, \
			fcf_margin={_fmt(fh.get('fcf_margin'), '.2%')}, \
			revenue_growth_1yr={_fmt(fh.get('revenue_growth_1yr'), '.2%')}, \
			earnings_quality={fh.get('earnings_quality', 'N/A')}
		Earnings: avg_surprise={ed.get('avg_eps_surprise_pct', 'N/A')}%, \
			consecutive_beats={ed.get('consecutive_beats', 'N/A')}
		""")

	prompt = ANOMALY_PROMPT.format(
		r2_threshold    = R2_LOW_THRESHOLD,
		garch_threshold = GARCH_HIGH_PERSISTENCE,
		peg_high        = PEG_HIGH_THRESHOLD,
		peg_low         = PEG_LOW_THRESHOLD,
		data            = "\n".join(data_lines),
	)

	try:
		llm      = get_llm()
		response = llm.invoke(prompt)
		print(f"\n LLM Commentary:{response.content.strip()}", file=file)
		return response.content.strip()
	except Exception as e:
		print(f"    [warn] LLM commentary failed: {e}")
		return "Quantitative commentary unavailable this run."
	

def agent2_quant(state: PipelineState) -> dict:
	"""
	Agent 2 — Quantitative Analysis.

	Downloads price history and Fama-French factors, fits a 3-factor OLS
	regression and GARCH(1,1) per ticker to produce mu and sigma estimates.
	Also fetches valuation metrics, financial health, and earnings data via
	yfinance, then generates an LLM commentary on anomalous quantitative signals.

	Args:
		state: Pipeline state dict containing tickers, run_date, and errors.

	Returns:
		Partial state update dict with keys: factor_results, garch_results,
		mu_sigma, valuation, financial_health, earnings_data, earnings_dates,
		quant_commentary, errors.
	"""
	print(f"\n  [Agent 2] Quant analyst running")
	tickers  = state["tickers"]
	print(f"            Tickers : {tickers}")
	run_date = date.fromisoformat(state["run_date"])
	start_date = (date.today() - timedelta(days=int(HISTORY_YEARS * 365)))
	errors   = list(state.get("errors") or [])
	existing_errors = len(errors)

	# ── Results containers ────────────────────────────────────────────────────
	factor_results   = {}
	garch_results    = {}
	mu_sigma         = {}
	valuation        = {}
	financial_health = {}
	earnings_data    = {}
	earnings_dates   = {}

	# Fetch Prices
	print(f"\n    Downloading {HISTORY_YEARS}yr price history...")
	try:
		raw_prices = fetch_prices(tickers, start_date, run_date)
	except Exception as e:
		errors.append(f"Price history download failed: {e}")
		print(f"    [error] Price download failed: {e}")
		# Cannot proceed without price data — return early with stubs
		return {
			"factor_results":   {t: {} for t in tickers},
			"garch_results":    {t: {} for t in tickers},
			"mu_sigma":         {t: {"mu_annual": 0.07, "sigma_annual": 0.20} for t in tickers},
			"valuation":        {t: {} for t in tickers},
			"financial_health": {t: {} for t in tickers},
			"earnings_data":    {t: {} for t in tickers},
			"earnings_dates":   {t: None for t in tickers},
			"ticker_info":      {t: {"type": "us_stock"} for t in tickers},
			"quant_commentary": "Quantitative analysis unavailable — price data download failed.",
			"errors":           errors,
		}
	
	# ffill aligns cross-exchange calendars (TSX vs NYSE have different holidays).
	# will add additional "0% return" days which are technically false but have minor effect
	# on historical return and volatility
	raw_prices = raw_prices[tickers].ffill()

	# Build covariance matrix for agent 4
	returns_df = np.log(raw_prices / raw_prices.shift(1)).iloc[1:]
	cov_matrix = returns_df.cov().to_dict()  # serialisable as nested dict
	
	# Fetch FF Factors
	factors_available = False
	ff_factors = None
	try:
		ff_factors = fetch_ff_factors(run_date, start_date)
		factors_available = True
	except Exception as e:
		errors.append(f"Factor history download failed: {e}")
		print(f"    [error] Factor download failed: {e}")

	# Fetch market-level inputs (once, shared across all tickers)
	cape    = fetch_cape()
	real_rf = fetch_real_rf()

	sum_dir = Path(state["user_path"]) / "data" / state["run_date"] / "summaries"
	sum_dir.mkdir(parents=True, exist_ok=True)

	for ticker in tickers:
		fm = None
		g = None

		dir_name = ticker.removesuffix(".TO")
		(sum_dir / dir_name).mkdir(parents=True, exist_ok=True)
		with open(sum_dir / dir_name / "quant.txt", "w", encoding="utf-8") as quant_file:
			if factors_available:
				try:
					ret = compute_returns(raw_prices, ticker, run_date, start_date, ff_factors, file=quant_file)
					fm = run_factor_models(ret["excess_ret"], ret["MKT"], ret["SMB"], ret["HML"], ret["rf_ann"], cape, real_rf, file=quant_file)
					factor_results[ticker] = {
						k: v for k, v in fm.items()
						if k not in ("residuals", "Y_hat")
					}
				except Exception as e:
					errors.append(f"{ticker}: Failed to run factor analysis: {e}")
					factor_results[ticker] = {}

				if fm is not None:
					try:
						g = run_garch(fm["residuals"], ret["factor_vols"], fm["b_MKT"], fm["b_SMB"], fm["b_HML"], file=quant_file)
						garch_results[ticker] = {
							k: v for k, v in g.items()
							if k not in ("h_garch", "pvalues")
						}
					except Exception as e:
						errors.append(f"{ticker}: Failed to run garch analysis: {e}")
						garch_results[ticker] = {}
			
		last_price = float(raw_prices[ticker].dropna().iloc[-1]) if (ticker in raw_prices.columns and raw_prices[ticker].notna().any()) else None
		if fm and g:
			mu_sigma[ticker] = {
				"mu_annual":    fm["mu_annual"],
				"sigma_annual": g["sigma_total_annual"],
				"s_current":    last_price,
				"is_fallback":  False
			}
		else:
			errors.append(f"{ticker}: Failed to get mu and sigma")
			mu_sigma[ticker] = {
				"mu_annual":    0.07,
				"sigma_annual": 0.20,
				"s_current":    last_price,
				"is_fallback":  True,
			}
		
		try:
			financial_health[ticker] = fetch_financial_health(ticker)
		except Exception as e:
			errors.append(f"{ticker}: Financial health fetch failed: {e}")
			financial_health[ticker] = {}
			
		try:
			valuation_metrics = fetch_valuation_metrics(ticker)
			val_signal = compute_valuation_signal(ticker, state["user_path"], valuation_metrics)
			valuation[ticker] = {**valuation_metrics, **val_signal}
		except Exception as e:
			errors.append(f"{ticker}: Valuation fetch failed: {e}")
			valuation[ticker] = {}
			
		try:
			earnings = fetch_earnings_data(ticker)
			earnings_data[ticker] = earnings
			earnings_dates[ticker] = earnings.get("next_earnings_date")
		except Exception as e:
			errors.append(f"{ticker}: earnings data fetch failed: {e}")
			earnings_data[ticker]  = {}
			earnings_dates[ticker] = None
		
		

	# ── LLM anomaly commentary ────────────────────────────────────────────────
	print(f"\n    Generating quantitative commentary...")
	try:
		with open(sum_dir / "llm_quant.txt", "w", encoding="utf-8") as quant_file:
			quant_commentary = generate_quant_commentary(
				tickers,
				factor_results,
				garch_results,
				valuation,
				financial_health,
				earnings_data,
				quant_file
			)
	except Exception as e:
			errors.append(f"Quant Commentary Failed: {e}")
			quant_commentary = None

	print(f"\n  [Agent 2] Complete. New Errors: {len(errors) - existing_errors}")

	return {
		"factor_results":   factor_results,
		"garch_results":    garch_results,
		"mu_sigma":         mu_sigma,
		"valuation":        valuation,
		"financial_health": financial_health,
		"earnings_data":    earnings_data,
		"earnings_dates":   earnings_dates,
		"quant_commentary": quant_commentary,
		"covariance_matrix":cov_matrix,
		"cape":             cape,
		"errors":           errors,
	}