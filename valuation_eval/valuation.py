"""
sector_pe.py
============
Fetches industry and sector average PE ratios from three layered sources,
then integrates them into the valuation signal alongside a stock's own
historical PE. Gives the advisor three comparison points instead of one:

  1. Stock PE vs its own 5-year history     (is this cheap for THIS stock?)
  2. Stock PE vs current sector median      (is this cheap vs peers NOW?)
  3. Stock PE vs long-run sector average    (is this sector historically cheap?)

Data sources (used in priority order, with fallback):
  A. Damodaran (NYU) — authoritative long-run sector PE tables, updated annually
  B. Sector ETF trailing PE via yfinance   — fast, real-time sector proxy
  C. Peer sampling via yfinance            — slower but most accurate for current median

SQLite caching: sector PE data is stored in a sector_pe_cache table so
the pipeline only re-fetches when data is more than 7 days old.
"""

import time
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from valuation_eval.utilities import SECTOR_ETF_MAP, EV_EBITDA_UNRELIABLE_SECTORS, SECTOR_PEERS, SECTOR_PEG_LONGRUN

_YF_SECTOR_MAP = {
	"Consumer Cyclical":  "Consumer Discretionary",
	"Consumer Defensive": "Consumer Staples",
	"Financial Services": "Financials",
}

logger = logging.getLogger("investment_agent")

# ═════════════════════════════════════════════════════════════════════════════
# UPDATED compute_valuation_signal  (replaces Fix 1 from previous session)
# Now incorporates three-way comparison:
#   1. Stock PE vs own 5-year history
#   2. Stock PE vs current sector median
#   3. Sector itself vs its long-run average (Damodaran)
# ═════════════════════════════════════════════════════════════════════════════

def compute_valuation_signal(
	ticker:   str,
	user_path: str,
	metrics:  dict,
) -> dict:
	"""
	Comprehensive relative valuation signal.

	Parameters
	----------
	ticker      : stock ticker
	metrics     : dict with forwardPE, trailingPE, trailingPegRatio,
				  enterpriseToEbitda from yfinance info
	info        : full yfinance .info dict
	sector_pe   : SectorPEResult from SectorPEFetcher
	hist_pe     : dict with 'median', 'p10', 'p90' from _fetch_historical_pe_range()

	Returns
	-------
	dict with valuation_signal, signal_confidence, pe_vs_history_pctile,
		 pe_vs_sector_pct, sector_vs_longrun, signal_breakdown, narrative
	"""
	db_path = Path(user_path) / "history.db"

	fwd_pe    = metrics.get("fwd_pe")
	ttm_pe	  = metrics.get("ttm_pe")
	peg       = metrics.get("peg")
	ev_ebitda = metrics.get("ev_ebitda")

	hist_pe, hist_peg, sector = _fetch_historical_values(ticker)
	sector = _YF_SECTOR_MAP.get(sector, sector)
	sector_peg_longrun = SECTOR_PEG_LONGRUN.get(sector, {})

	sector_data = fetch_sector_metrics_from_peers(sector, db_path)
	sector_fwd_pe    = sector_data["fwd_pe"]    if sector_data["fwd_pe"]    else None
	sector_ev_ebitda = sector_data["ev_ebitda"] if sector_data["ev_ebitda"] else None
	sector_ttm_pe = sector_data["ttm_pe"] if sector_data["ttm_pe"] else None
	sector_peg = sector_data["peg"] if sector_data["peg"] else None

	signals = []   # list of (label, verdict, weight)

	# ── 1. PEG ratio ───────────
	peg_own_ratio, peg_sector_ratio = _peg_signal(peg, hist_peg, sector_peg, sector_peg_longrun, signals)

	# ── 2. PE ──────────────────
	pe_pctile, sector_premium_ratio = _pe_signal(ttm_pe, fwd_pe, hist_pe, sector_ttm_pe, sector_fwd_pe, signals)

	# ── 3. EV/EBITDA ───────────
	ev_sector_ratio = _ev_ebitda_signal(ev_ebitda, sector_ev_ebitda, signals)

	# ── Weighted vote ─────────────────────────────────────────────────────────
	score = {"cheap": 0, "fair": 0, "expensive": 0}
	for _, verdict, weight in signals:
		score[verdict] += weight

	total   = sum(score.values())
	dominant = max(score, key=score.get) if total > 0 else "fair"
	confidence = score[dominant] / total if total > 0 else 0.0

	return {
		"valuation_signal":        dominant,
		"signal_confidence":       round(confidence, 2),
		"pe_vs_history_pctile":    round(pe_pctile, 1) if pe_pctile is not None else None,
		"pe_vs_sector_ratio":	   round(sector_premium_ratio, 1) if sector_premium_ratio is not None else None,
		"sector_pe_current":       sector_ttm_pe["median"] if sector_ttm_pe else None,
		"peg_hist_ratio":		   peg_own_ratio,
		"peg_sector_ratio":		   peg_sector_ratio,
		"ev_sector_ratio":		   ev_sector_ratio,
		"signal_breakdown":        signals,
	}


def _fetch_historical_values(ticker: str, years: int = 3) -> dict:
	"""
	Fetches historical PEG and PE values for the given ticker
	
	Returns:
		PEG: median, p25, p75, p10, p90, filtered_years
			(or Nones if insufficient data)
		PE: 
	"""
	try:
		tk       = yf.Ticker(ticker)
		income = tk.income_stmt
		eps_row = None
		for label in ["Basic EPS", "Diluted EPS"]:
			if label in income.index:
				eps_row = income.loc[label]
				break

		if eps_row is not None:
			eps_by_year = eps_row.dropna()
			eps_by_year.index = eps_by_year.index.year
			earnings = pd.DataFrame({"EPS": eps_by_year}).sort_index(ascending=True)
		else:
			earnings = pd.DataFrame()

		hist_prices = tk.history(period=f"{years + 1}y", interval="1mo")["Close"]

		info    = tk.info
		sector  = info.get("sector")

		hist_peg = _fetch_historical_peg_range(earnings, hist_prices)
		hist_pe  = _fetch_historical_pe_range(earnings, hist_prices)

		return (hist_pe, hist_peg, sector)

	except Exception as e:
		logger.warning(f"Historical fetch failed for {ticker}: {e}")
		return (_empty_pe_range(), _empty_peg_range(), None)


def _fetch_historical_pe_range(earnings: Optional[pd.DataFrame], hist_prices: pd.Series) -> dict:
	"""
	Computes the stock's own trailing PE percentiles over the past N years.
	Uses yfinance: earnings history gives annual EPS, price gives the PE at each point.
	Falls back gracefully if data is unavailable.
	"""
	try:
		if earnings is None or earnings.empty:
			return _empty_pe_range()

		# Match each year's EPS to the average price in that year
		pe_series = []
		for year, row in earnings.iterrows():
			eps = row.get("Earnings") or row.get("EPS")
			if eps and eps > 0:
				year_prices = hist_prices[hist_prices.index.year == int(year)]
				if not year_prices.empty:
					avg_price = year_prices.mean()
					pe_series.append(avg_price / eps)

		if len(pe_series) < 2:
			return _empty_pe_range()

		import numpy as np
		return {
			"median": float(np.median(pe_series)),
			"p10":    float(np.percentile(pe_series, 10)),
			"p90":    float(np.percentile(pe_series, 90)),
			"series": pe_series
		}
	except Exception:
		return _empty_pe_range()

def _empty_pe_range() -> dict:
	return {"median": None, "p10": None, "p90": None}


def _build_valuation_narrative(
	ticker, fwd_pe, peg, pe_pctile, sector_premium_pct,
	sector_pe, dominant,
) -> str:
	"""
	Builds a 2-3 sentence human-readable narrative for the report and advisor prompt.
	This replaces the vague 'trading at a high forward P/E' reasoning.
	"""
	parts = []

	# Stock vs own history
	if pe_pctile is not None:
		if pe_pctile < 25:
			parts.append(
				f"{ticker}'s forward PE of {fwd_pe:.1f}x sits at the "
				f"{pe_pctile:.0f}th percentile of its own 5-year range — "
				f"historically cheap for this specific stock."
			)
		elif pe_pctile > 75:
			parts.append(
				f"{ticker}'s forward PE of {fwd_pe:.1f}x sits at the "
				f"{pe_pctile:.0f}th percentile of its own 5-year range — "
				f"historically stretched."
			)
		else:
			parts.append(
				f"{ticker}'s forward PE of {fwd_pe:.1f}x is near the midpoint "
				f"of its own 5-year history ({pe_pctile:.0f}th percentile)."
			)

	# Stock vs sector
	if sector_premium_pct is not None and sector_pe.current_pe:
		direction = "discount" if sector_premium_pct < 0 else "premium"
		parts.append(
			f"vs the current {sector_pe.sector} sector median of "
			f"{sector_pe.current_pe:.1f}x (peer sample), {ticker} trades at a "
			f"{abs(sector_premium_pct):.0f}% {direction}."
		)

	# Sector vs long-run
	if sector_pe.damodaran_pe and sector_pe.current_pe:
		sector_vs_lr = (sector_pe.current_pe / sector_pe.damodaran_pe - 1) * 100
		if abs(sector_vs_lr) > 10:
			direction = "above" if sector_vs_lr > 0 else "below"
			parts.append(
				f"The {sector_pe.sector} sector itself currently trades "
				f"{abs(sector_vs_lr):.0f}% {direction} its long-run Damodaran average "
				f"({sector_pe.current_pe:.1f}x vs {sector_pe.damodaran_pe:.1f}x), "
				f"providing additional context for this {'discount' if sector_vs_lr > 0 else 'premium'}."
			)

	# PEG
	if peg is not None:
		if peg < 1.5:
			parts.append(f"The PEG of {peg:.2f} suggests growth is not fully priced in.")
		elif peg > 2.5:
			parts.append(f"The PEG of {peg:.2f} suggests the growth premium is elevated.")

	return " ".join(parts) if parts else f"Valuation signal: {dominant}."


# ── PEG: three-layer signal (replaces the single absolute check) ──────────
def _peg_signal(peg, hist_peg, sector_peg_current, sector_peg_longrun, signals):
	"""
	Returns a list of (label, verdict, weight) tuples for PEG.
	Three layers:
	  1. Absolute Lynch threshold (always included)
	  2. vs stock's own history (if available, highest weight)
	  3. vs sector current and long-run median
	"""
	if peg is None:
		return None, None
	
	ratio_vs_own = None
	ratio_vs_sector = None

	# 1. Absolute threshold — always computed, lower weight than relative
	if peg < 1.0:
		signals.append(("peg_absolute", "cheap",     1))
	elif peg < 1.5:
		signals.append(("peg_absolute", "cheap",     1))   # Lynch: still attractive
	elif peg < 2.5:
		signals.append(("peg_absolute", "fair",      1))
	else:
		signals.append(("peg_absolute", "expensive", 1))

	# 2. vs stock's own 5-year history — strongest signal, weight=3
	if hist_peg.get("median") is not None:
		ratio_vs_own = peg / hist_peg["median"]
		if ratio_vs_own < 0.70:
			signals.append(("peg_vs_own_history", "cheap",     3))
		elif ratio_vs_own < 0.90:
			signals.append(("peg_vs_own_history", "cheap",     2))
		elif ratio_vs_own < 1.10:
			signals.append(("peg_vs_own_history", "fair",      1))
		elif ratio_vs_own < 1.30:
			signals.append(("peg_vs_own_history", "expensive", 2))
		else:
			signals.append(("peg_vs_own_history", "expensive", 3))

	# 3. vs current sector peer PEG — weight=2
	if sector_peg_current and sector_peg_current.get("median"):
		ratio_vs_sector = peg / sector_peg_current["median"]
		if ratio_vs_sector < 0.80:
			signals.append(("peg_vs_sector_current", "cheap",     2))
		elif ratio_vs_sector < 1.15:
			signals.append(("peg_vs_sector_current", "fair",      1))
		else:
			signals.append(("peg_vs_sector_current", "expensive", 2))
	elif sector_peg_longrun and sector_peg_longrun.get("median"):
		# 4. vs sector long-run PEG norm — weight=1 (structural context only)
		ratio_vs_sector = peg / sector_peg_longrun["median"]
		if ratio_vs_sector < 0.75:
			signals.append(("peg_vs_sector_longrun", "cheap",     1))
		elif ratio_vs_sector < 1.20:
			signals.append(("peg_vs_sector_longrun", "fair",      1))
		else:
			signals.append(("peg_vs_sector_longrun", "expensive", 1))

	return ratio_vs_own, ratio_vs_sector

def _pe_signal(ttm_pe, fwd_pe, hist_pe, sector_ttm_pe, sector_fwd_pe, signals):
	"""
	Returns a list of (label, verdict, weight) tuples for PEG.
	Three layers:
	  1. Absolute Lynch threshold (always included)
	  2. vs stock's own history (if available, highest weight)
	  3. vs sector current and long-run median
	"""
	if ttm_pe is None and fwd_pe is None:
		return None, None
	
	pe_pctile = None
	ttm_ratio_vs_sector = None
	if ttm_pe:
		if hist_pe.get("p10") is not None:
			lo, hi   = hist_pe["p10"], hist_pe["p90"]
			pe_pctile = max(0.0, min(100.0, (ttm_pe - lo) / (hi - lo) * 100)) if hi > lo else 50.0
			if pe_pctile < 25:
				signals.append(("ttm_pe_vs_own_history", "cheap",     3))
			elif pe_pctile < 50:
				signals.append(("ttm_pe_vs_own_history", "cheap",     1))
			elif pe_pctile < 75:
				signals.append(("ttm_pe_vs_own_history", "fair",      1))
			else:
				signals.append(("ttm_pe_vs_own_history", "expensive", 3))

		if sector_ttm_pe and sector_ttm_pe.get("median"):
			ttm_ratio_vs_sector = ttm_pe / sector_ttm_pe["median"]
			if ttm_ratio_vs_sector < 0.80:
				signals.append(("ttm_pe_vs_sector_current", "cheap",     2))
			elif ttm_ratio_vs_sector < 1.15:
				signals.append(("ttm_pe_vs_sector_current", "fair",      1))
			else:
				signals.append(("ttm_pe_vs_sector_current", "expensive", 2))

	if fwd_pe:
		if sector_fwd_pe and sector_fwd_pe.get("median"):
			fwd_ratio_vs_sector = fwd_pe / sector_fwd_pe["median"]
			if fwd_ratio_vs_sector < 0.80:
				signals.append(("fwd_pe_vs_sector_current", "cheap",     2))
			elif fwd_ratio_vs_sector < 1.15:
				signals.append(("fwd_pe_vs_sector_current", "fair",      1))
			else:
				signals.append(("fwd_pe_vs_sector_current", "expensive", 2))

	return pe_pctile, ttm_ratio_vs_sector


def _ev_ebitda_signal(ev_ebitda, sector_ev_ebitda, signals):
	"""
	Returns a list of (label, verdict, weight) tuples for EV/EBITDA.
	Three layers:
	  1. Absolute Lynch threshold (always included)
	  2. vs stock's own history (if available, highest weight)
	  3. vs sector current and long-run median
	"""
	if ev_ebitda and sector_ev_ebitda and sector_ev_ebitda.get("median"):
		ratio_vs_sector = ev_ebitda / sector_ev_ebitda["median"]
		if ratio_vs_sector < 0.80:
			signals.append(("ev_ebitda_vs_sector_current", "cheap",     2))
		elif ratio_vs_sector < 1.20:
			signals.append(("ev_ebitda_vs_sector_current", "fair",      1))
		else:
			signals.append(("ev_ebitda_vs_sector_current", "expensive", 2))
		return ratio_vs_sector

	return None


def _fetch_historical_peg_range(earnings: Optional[pd.DataFrame], hist_prices: pd.Series) -> dict:
	"""
	Reconstructs the stock's historical PEG range from yfinance data.

	PEG = trailing PE / EPS growth rate.
	Uses annual EPS to compute year-over-year growth, then divides
	the year's average trailing PE by that growth rate.

	Critical: filters out years where EPS growth was an outlier
	(defined as more than 2 standard deviations from the mean growth
	rate across the period). Outlier growth years produce artificially
	low PEGs that would corrupt the historical range.

	Returns:
		median, p25, p75, p10, p90, filtered_years
		(or Nones if insufficient data)
	"""
	try:
		if earnings is None or earnings.empty or len(earnings) < 3:
			return _empty_peg_range()

		# Compute year-over-year EPS growth rates
		eps_col = next(
			(c for c in earnings.columns if "earn" in c.lower() or "eps" in c.lower()),
			earnings.columns[0]
		)
		eps_series  = earnings[eps_col].dropna()
		growth_rates = eps_series.pct_change().dropna()

		# Filter out negative EPS years (PEG undefined) and outlier growth years
		valid_mask   = (growth_rates > 0) & (eps_series.iloc[1:] > 0)
		growth_clean = growth_rates[valid_mask]

		if len(growth_clean) < 2:
			return _empty_peg_range()

		# Remove statistical outliers (> 2 std from mean growth)
		mean_g, std_g  = growth_clean.mean(), growth_clean.std()
		outlier_mask   = (growth_clean - mean_g).abs() < 2 * std_g
		growth_filtered = growth_clean[outlier_mask]

		if len(growth_filtered) < 2:
			return _empty_peg_range()

		peg_series  = []

		for year in growth_filtered.index:
			g = growth_filtered[year] * 100   # convert to percentage
			if g <= 0:
				continue
			year_prices = hist_prices[hist_prices.index.year == int(year)]
			if year_prices.empty:
				continue
			avg_price = year_prices.mean()
			eps_val   = eps_series.get(year)
			if eps_val and eps_val > 0:
				ttm_pe = avg_price / eps_val
				peg    = ttm_pe / g
				# Sanity bounds: PEG outside 0.2–10 is almost certainly a data error
				if 0.2 < peg < 10.0:
					peg_series.append((int(year), g, ttm_pe, peg))

		if len(peg_series) < 2:
			return _empty_peg_range()

		peg_values = [p[3] for p in peg_series]
		return {
			"median":         float(np.median(peg_values)),
			"p25":            float(np.percentile(peg_values, 25)),
			"p75":            float(np.percentile(peg_values, 75)),
			"p10":            float(np.percentile(peg_values, 10)),
			"p90":            float(np.percentile(peg_values, 90)),
			"filtered_years": peg_series,   # (year, growth_pct, pe, peg) tuples
			"n":              len(peg_values),
		}

	except Exception as e:
		logger.warning(f"Historical PEG fetch failed: {e}")
		return _empty_peg_range()
	
def _empty_peg_range() -> dict:
	return {"median": None, "p25": None, "p75": None,
			"p10": None, "p90": None, "filtered_years": [], "n": 0}


def fetch_etf_holdings(etf_ticker: str, top_n: int = 20) -> list[str]:
    """
    Fetches current top holdings of a sector ETF from yfinance.
	ETF ticker appended so it contributes to PE.
    """
    try:
        tk = yf.Ticker(etf_ticker)
        
        # yfinance exposes holdings via .funds_data for ETFs
        holdings = tk.funds_data.top_holdings
        
        if holdings is not None and not holdings.empty:
            tickers = holdings.index.tolist()[:top_n]
            # Add the ETF itself so it contributes to PE median
            if etf_ticker not in tickers:
                tickers.append(etf_ticker)
            return tickers
            
    except Exception as e:
        logger.warning(f"ETF holdings fetch failed for {etf_ticker}: {e}")
    
    return []


def get_peer_list(sector: str) -> list[str]:
    """
    Returns peer list, preferring live ETF holdings over the
    static fallback.
    """
    etf = SECTOR_ETF_MAP.get(sector)
    
    if etf:
        live = fetch_etf_holdings(etf, top_n=20)
        if len(live) >= 10:    # enough peers to be meaningful
            return live
    
    # Fallback to static list + ETF appended
    static = SECTOR_PEERS.get(sector, []).copy()
    if etf and etf not in static:
        static.append(etf)
    return static


def _load_sector_pe_cache(sector: str, db_path: str, ttl_days: int = 7) -> dict | None:
	try:
		conn = sqlite3.connect(db_path)
		row = conn.execute(
			"SELECT fwd_pe, ttm_pe, ev_ebitda, peg, fetched_at FROM sector_pe_cache WHERE sector = ?",
			(sector,)
		).fetchone()
		conn.close()
		if not row:
			return None
		if datetime.now() - datetime.fromisoformat(row[4]) > timedelta(days=ttl_days):
			return None
		return {
			"fwd_pe":    json.loads(row[0]) if row[0] else None,
			"ttm_pe":    json.loads(row[1]) if row[1] else None,
			"ev_ebitda": json.loads(row[2]) if row[2] else None,
			"peg":       json.loads(row[3]) if row[3] else None,
		}
	except Exception as e:
		logger.warning(f"Sector PE cache read failed: {e}")
		return None


def _save_sector_pe_cache(sector: str, result: dict, db_path: str) -> None:
	try:
		conn = sqlite3.connect(db_path)
		conn.execute("""
			INSERT OR REPLACE INTO sector_pe_cache (sector, fwd_pe, ttm_pe, ev_ebitda, peg, fetched_at)
			VALUES (?, ?, ?, ?, ?, ?)
		""", (
			sector,
			json.dumps(result["fwd_pe"]),
			json.dumps(result["ttm_pe"]),
			json.dumps(result["ev_ebitda"]),
			json.dumps(result["peg"]),
			datetime.now().isoformat(),
		))
		conn.commit()
		conn.close()
	except Exception as e:
		logger.warning(f"Sector PE cache write failed: {e}")


def fetch_sector_metrics_from_peers(
	sector: str,
	db_path: str,
	sample_size: int = 15,
) -> dict:
	"""
	Single pass over sector peers fetching all multiples at once.
	Returns medians for forward PE, trailing PE, and EV/EBITDA.
	Results are cached in sector_pe_cache (TTL 7 days) to avoid re-fetching
	for multiple tickers in the same sector within a run.
	"""
	cached = _load_sector_pe_cache(sector, db_path)
	if cached:
		logger.debug(f"Sector PE Cache: hit for {sector}")
		return cached

	if sector in EV_EBITDA_UNRELIABLE_SECTORS:
		ev_ebitda_valid = False
	else:
		ev_ebitda_valid = True

	peers = get_peer_list(sector)

	fwd_pe_vals    = []
	ttm_pe_vals    = []
	ev_ebitda_vals = []
	peg_vals  = []

	for ticker in peers:
		try:
			info = yf.Ticker(ticker).info      # one call, all fields

			fwd  = info.get("forwardPE")
			ttm  = info.get("trailingPE")
			ev   = info.get("enterpriseToEbitda")
			peg  = info.get("trailingPegRatio") or info.get("pegRatio")

			if fwd:
				fwd_pe_vals.append(float(fwd))

			if ttm:
				ttm_pe_vals.append(float(ttm))

			if ev_ebitda_valid and ev:
				ev_ebitda_vals.append(float(ev))

			if peg:
				peg_vals.append(peg)

			time.sleep(0.15)

		except Exception as e:
			logger.warning(f"Peer fetch failed for {ticker}: {e}")
			continue

	def _stats(vals):
		if len(vals) < 5:
			return None
		return {
			"median": float(np.median(vals)),
			"p25":    float(np.percentile(vals, 25)),
			"p75":    float(np.percentile(vals, 75)),
			"n":      len(vals),
		}

	result = {
		"fwd_pe":    _stats(fwd_pe_vals),
		"ttm_pe":    _stats(ttm_pe_vals),
		"ev_ebitda": _stats(ev_ebitda_vals),
		"peg":		 _stats(peg_vals),
	}
	_save_sector_pe_cache(sector, result, db_path)
	return result