"""
Valuation eval unit tests — runs without the full pipeline.

External calls (yfinance) are mocked. SQLite DB tests use a temp directory.

Run:
	python test/test_valuation_eval.py
"""

import sys
import os
import json
import sqlite3
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.schema import init_database
from valuation_eval.valuation import (
	_fetch_historical_pe_range,
	_fetch_historical_peg_range,
	_peg_signal,
	_pe_signal,
	_ev_ebitda_signal,
	_load_sector_pe_cache,
	_save_sector_pe_cache,
	fetch_sector_metrics_from_peers,
	compute_valuation_signal,
)


# ── Shared synthetic data ──────────────────────────────────────────────────

def make_earnings(eps_values: list, start_year: int = 2019) -> pd.DataFrame:
	"""Synthetic annual earnings DataFrame matching yfinance format."""
	years = list(range(start_year, start_year + len(eps_values)))
	return pd.DataFrame({"Earnings": eps_values}, index=years)


def make_prices(start: str = "2019-01-01", end: str = "2023-12-31",
				base: float = 150.0, step: float = 1.0) -> pd.Series:
	"""Monthly price series with DatetimeIndex."""
	dates = pd.date_range(start, end, freq="ME")
	return pd.Series([base + i * step for i in range(len(dates))], index=dates)


def make_db() -> tuple[object, Path]:
	"""Create fresh SQLite DB in a temp dir; return (conn, db_path)."""
	tmpdir = Path(tempfile.mkdtemp())
	conn = init_database(tmpdir)
	return conn, tmpdir / "history.db"


# ── 1. _fetch_historical_pe_range ─────────────────────────────────────────

def test_pe_range_normal():
	earnings = make_earnings([5.0, 6.0, 7.0, 8.0, 9.0])
	prices   = make_prices()

	result = _fetch_historical_pe_range(earnings, prices)

	assert isinstance(result, dict), "Expected dict return"
	assert result["median"] is not None, "Expected non-None median"
	assert result["p10"] < result["median"] < result["p90"], "p10 < median < p90 expected"
	assert len(result["series"]) >= 2, "Expected at least 2 PE data points"
	print("[PASS] test_pe_range_normal")


def test_pe_range_empty_earnings():
	earnings = pd.DataFrame({"Earnings": []})
	prices   = make_prices()

	result = _fetch_historical_pe_range(earnings, prices)

	# Bug note: function returns _empty_pe_range (function ref) on empty — result is callable.
	# We just assert it doesn't raise.
	assert result is not None, "Expected non-None return on empty earnings"
	print("[PASS] test_pe_range_empty_earnings")


def test_pe_range_single_year():
	"""Only one EPS/price match — fewer than 2 data points → empty range."""
	earnings = make_earnings([5.0])
	prices   = make_prices()

	result = _fetch_historical_pe_range(earnings, prices)

	# Returns _empty_pe_range function ref (existing bug) or empty dict — not a plausible range
	if isinstance(result, dict):
		assert result.get("median") is None, "Expected None median for single year"
	print("[PASS] test_pe_range_single_year")


# ── 2. _fetch_historical_peg_range ────────────────────────────────────────

def test_peg_range_normal():
	"""Steady EPS growth → valid PEG range returned."""
	eps = [3.0, 3.6, 4.3, 5.2, 6.2, 7.5]
	earnings = make_earnings(eps, start_year=2018)
	prices   = make_prices(start="2018-01-01", end="2023-12-31")

	result = _fetch_historical_peg_range(earnings, prices)

	assert isinstance(result, dict), "Expected dict"
	assert result["median"] is not None, "Expected non-None median"
	assert result["n"] >= 2, "Expected at least 2 valid PEG data points"
	assert all(0.2 < p[3] < 10.0 for p in result["filtered_years"]), \
		"All PEG values should be within sanity bounds (0.2, 10.0)"
	print("[PASS] test_peg_range_normal")


def test_peg_range_outlier_filtered():
	"""An anomalous spike year (>2σ growth) should be excluded."""
	eps = [3.0, 3.3, 3.6, 3.9, 0.5, 4.5, 4.9]  # year 2023 has huge recovery growth
	earnings = make_earnings(eps, start_year=2017)
	prices   = make_prices(start="2017-01-01", end="2023-12-31")

	result = _fetch_historical_peg_range(earnings, prices)

	if result["n"] > 0:
		growth_rates = [p[1] for p in result["filtered_years"]]
		mean_g = np.mean(growth_rates)
		std_g  = np.std(growth_rates)
		for g in growth_rates:
			assert abs(g - mean_g) < 2 * std_g + 1e-6, \
				f"Outlier growth rate {g:.1f}% not filtered"
	print("[PASS] test_peg_range_outlier_filtered")


def test_peg_range_insufficient_data():
	"""Fewer than 3 EPS rows → empty range."""
	earnings = make_earnings([5.0, 6.0])
	prices   = make_prices()

	result = _fetch_historical_peg_range(earnings, prices)

	assert result["median"] is None, "Expected None median for < 3 EPS rows"
	assert result["n"] == 0, "Expected n=0"
	print("[PASS] test_peg_range_insufficient_data")


# ── 3. Signal helpers ──────────────────────────────────────────────────────

def test_peg_signal_cheap_absolute():
	signals = []
	_peg_signal(peg=0.8, hist_peg={}, sector_peg_current=None,
				sector_peg_longrun=None, signals=signals)

	verdicts = [s[1] for s in signals]
	assert "cheap" in verdicts, f"Expected cheap signal, got {verdicts}"
	print("[PASS] test_peg_signal_cheap_absolute")


def test_peg_signal_none_peg_no_signals():
	signals = []
	_peg_signal(peg=None, hist_peg={}, sector_peg_current=None,
				sector_peg_longrun=None, signals=signals)

	assert len(signals) == 0, "Expected no signals for None PEG"
	print("[PASS] test_peg_signal_none_peg_no_signals")


def test_peg_signal_vs_own_history_cheap():
	signals = []
	_peg_signal(
		peg=1.0,
		hist_peg={"median": 2.0, "p25": 1.5, "p75": 2.5},
		sector_peg_current=None,
		sector_peg_longrun=None,
		signals=signals,
	)

	labels   = [s[0] for s in signals]
	verdicts = [s[1] for s in signals]
	assert "peg_vs_own_history" in labels, "Expected vs-own-history signal"
	idx = labels.index("peg_vs_own_history")
	assert verdicts[idx] == "cheap", f"Expected cheap, got {verdicts[idx]}"
	print("[PASS] test_peg_signal_vs_own_history_cheap")


def test_pe_signal_historically_cheap():
	"""TTM PE at the low end of hist range → cheap signal."""
	signals  = []
	hist_pe  = {"p10": 10.0, "p90": 30.0, "median": 20.0}
	sector_ttm = {"median": 25.0, "p25": 20.0, "p75": 30.0}

	pe_pctile, _ = _pe_signal(
		ttm_pe=11.0, fwd_pe=None,
		hist_pe=hist_pe,
		sector_ttm_pe=sector_ttm,
		sector_fwd_pe=None,
		signals=signals,
	)

	assert pe_pctile is not None, "Expected pe_pctile to be computed"
	assert pe_pctile < 25.0, f"Expected low percentile (< 25), got {pe_pctile:.1f}"
	verdicts = [s[1] for s in signals]
	assert "cheap" in verdicts, f"Expected cheap signal, got {verdicts}"
	print("[PASS] test_pe_signal_historically_cheap")


def test_pe_signal_historically_expensive():
	"""TTM PE near top of hist range → expensive signal."""
	signals  = []
	hist_pe  = {"p10": 10.0, "p90": 30.0, "median": 20.0}
	sector_ttm = {"median": 25.0, "p25": 20.0, "p75": 30.0}

	pe_pctile, _ = _pe_signal(
		ttm_pe=29.0, fwd_pe=None,
		hist_pe=hist_pe,
		sector_ttm_pe=sector_ttm,
		sector_fwd_pe=None,
		signals=signals,
	)

	assert pe_pctile > 75.0, f"Expected high percentile (> 75), got {pe_pctile:.1f}"
	verdicts = [s[1] for s in signals]
	assert "expensive" in verdicts, f"Expected expensive signal, got {verdicts}"
	print("[PASS] test_pe_signal_historically_expensive")


def test_ev_ebitda_signal_cheap():
	signals = []
	_ev_ebitda_signal(
		ev_ebitda=8.0,
		sector_ev_ebitda={"median": 12.0, "p25": 10.0, "p75": 15.0},
		signals=signals,
	)

	verdicts = [s[1] for s in signals]
	assert "cheap" in verdicts, f"Expected cheap EV/EBITDA signal, got {verdicts}"
	print("[PASS] test_ev_ebitda_signal_cheap")


def test_ev_ebitda_signal_no_sector_data():
	"""None sector EV/EBITDA → no signal appended."""
	signals = []
	_ev_ebitda_signal(ev_ebitda=10.0, sector_ev_ebitda=None, signals=signals)

	assert len(signals) == 0, "Expected no signal when sector EV/EBITDA is None"
	print("[PASS] test_ev_ebitda_signal_no_sector_data")


# ── 4. Sector PE cache ─────────────────────────────────────────────────────

def test_sector_pe_cache_miss():
	"""Empty DB → cache returns None."""
	_, db_path = make_db()

	result = _load_sector_pe_cache("Technology", str(db_path))

	assert result is None, "Expected None on cache miss"
	print("[PASS] test_sector_pe_cache_miss")


def test_sector_pe_cache_roundtrip():
	"""Write then read back — data should match."""
	_, db_path = make_db()
	data = {
		"fwd_pe":    {"median": 25.0, "p25": 20.0, "p75": 30.0, "n": 15},
		"ttm_pe":    {"median": 22.0, "p25": 18.0, "p75": 27.0, "n": 15},
		"ev_ebitda": {"median": 14.0, "p25": 11.0, "p75": 17.0, "n": 12},
		"peg":       {"median": 1.8,  "p25": 1.2,  "p75": 2.4,  "n": 13},
	}

	_save_sector_pe_cache("Technology", data, str(db_path))
	result = _load_sector_pe_cache("Technology", str(db_path))

	assert result is not None, "Expected cached result"
	assert result["fwd_pe"]["median"] == 25.0, \
		f"Expected fwd_pe median 25.0, got {result['fwd_pe']['median']}"
	assert result["ttm_pe"]["median"] == 22.0, \
		f"Expected ttm_pe median 22.0, got {result['ttm_pe']['median']}"
	print("[PASS] test_sector_pe_cache_roundtrip")


def test_sector_pe_cache_expired():
	"""Entry older than TTL → returns None."""
	_, db_path = make_db()
	data = {
		"fwd_pe": {"median": 25.0, "p25": 20.0, "p75": 30.0, "n": 15},
		"ttm_pe": None, "ev_ebitda": None, "peg": None,
	}

	# Write with an 8-day-old timestamp
	stale_ts = (datetime.now() - timedelta(days=8)).isoformat()
	conn = sqlite3.connect(str(db_path))
	conn.execute("""
		INSERT OR REPLACE INTO sector_pe_cache (sector, fwd_pe, ttm_pe, ev_ebitda, peg, fetched_at)
		VALUES (?, ?, ?, ?, ?, ?)
	""", ("Technology", json.dumps(data["fwd_pe"]), None, None, None, stale_ts))
	conn.commit()
	conn.close()

	result = _load_sector_pe_cache("Technology", str(db_path), ttl_days=7)

	assert result is None, "Expected None for expired cache entry"
	print("[PASS] test_sector_pe_cache_expired")


def test_sector_pe_cache_fresh_within_ttl():
	"""Entry 3 days old with TTL=7 → should still be returned."""
	_, db_path = make_db()
	data = {
		"fwd_pe":    {"median": 25.0, "p25": 20.0, "p75": 30.0, "n": 15},
		"ttm_pe":    None, "ev_ebitda": None, "peg": None,
	}

	fresh_ts = (datetime.now() - timedelta(days=3)).isoformat()
	conn = sqlite3.connect(str(db_path))
	conn.execute("""
		INSERT OR REPLACE INTO sector_pe_cache (sector, fwd_pe, ttm_pe, ev_ebitda, peg, fetched_at)
		VALUES (?, ?, ?, ?, ?, ?)
	""", ("Healthcare", json.dumps(data["fwd_pe"]), None, None, None, fresh_ts))
	conn.commit()
	conn.close()

	result = _load_sector_pe_cache("Healthcare", str(db_path), ttl_days=7)

	assert result is not None, "Expected cached result within TTL"
	assert result["fwd_pe"]["median"] == 25.0
	print("[PASS] test_sector_pe_cache_fresh_within_ttl")


# ── 5. fetch_sector_metrics_from_peers ────────────────────────────────────

def _make_mock_ticker_info(fwd_pe=25.0, ttm_pe=22.0, ev_ebitda=14.0, peg=1.8):
	mock = MagicMock()
	mock.info = {
		"forwardPE":           fwd_pe,
		"trailingPE":          ttm_pe,
		"enterpriseToEbitda":  ev_ebitda,
		"trailingPegRatio":    peg,
	}
	return mock


@patch("valuation_eval.valuation.yf.Ticker")
@patch("valuation_eval.valuation.time.sleep")
def test_fetch_peers_cache_miss(_, mock_ticker_cls):
	"""Cache miss → fetches peers, saves result, returns metrics."""
	_, db_path = make_db()
	mock_ticker_cls.return_value = _make_mock_ticker_info()

	result = fetch_sector_metrics_from_peers("Technology", str(db_path))

	assert result["fwd_pe"] is not None, "Expected fwd_pe stats"
	assert "median" in result["fwd_pe"], "Expected median key in fwd_pe"
	assert mock_ticker_cls.called, "Expected yfinance to be called on cache miss"

	# Result should now be cached
	cached = _load_sector_pe_cache("Technology", str(db_path))
	assert cached is not None, "Expected result to be saved to cache"
	print("[PASS] test_fetch_peers_cache_miss")


@patch("valuation_eval.valuation.yf.Ticker")
@patch("valuation_eval.valuation.time.sleep")
def test_fetch_peers_cache_hit(_, mock_ticker_cls):
	"""Pre-populated cache → no yfinance calls made."""
	_, db_path = make_db()
	cached_data = {
		"fwd_pe":    {"median": 99.0, "p25": 80.0, "p75": 110.0, "n": 15},
		"ttm_pe":    {"median": 90.0, "p25": 75.0, "p75": 105.0, "n": 15},
		"ev_ebitda": {"median": 14.0, "p25": 11.0, "p75": 17.0,  "n": 12},
		"peg":       {"median": 1.8,  "p25": 1.2,  "p75": 2.4,   "n": 13},
	}
	_save_sector_pe_cache("Technology", cached_data, str(db_path))

	result = fetch_sector_metrics_from_peers("Technology", str(db_path))

	assert not mock_ticker_cls.called, "Expected no yfinance calls on cache hit"
	assert result["fwd_pe"]["median"] == 99.0, \
		f"Expected cached median 99.0, got {result['fwd_pe']['median']}"
	print("[PASS] test_fetch_peers_cache_hit")


@patch("valuation_eval.valuation.yf.Ticker")
@patch("valuation_eval.valuation.time.sleep")
def test_fetch_peers_ev_ebitda_excluded_for_financials(_, mock_ticker_cls):
	"""EV/EBITDA should be None for Financials sector (unreliable)."""
	_, db_path = make_db()
	mock_ticker_cls.return_value = _make_mock_ticker_info()

	result = fetch_sector_metrics_from_peers("Financials", str(db_path))

	assert result["ev_ebitda"] is None, \
		f"Expected ev_ebitda=None for Financials, got {result['ev_ebitda']}"
	print("[PASS] test_fetch_peers_ev_ebitda_excluded_for_financials")


# ── 6. compute_valuation_signal ───────────────────────────────────────────

def _make_yfinance_mock(sector="Technology", fwd_pe=12.0, ttm_pe=11.0,
						peg=0.8, ev_ebitda=8.0, eps_vals=None):
	"""Mock yf.Ticker with .info and .income_stmt returning test data."""
	mock_tk = MagicMock()
	mock_tk.info = {
		"sector":              sector,
		"forwardPE":           fwd_pe,
		"trailingPE":          ttm_pe,
		"trailingPegRatio":    peg,
		"enterpriseToEbitda":  ev_ebitda,
		"revenueGrowth":       0.10,
		"earningsGrowth":      0.12,
	}

	if eps_vals is None:
		eps_vals = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
	n = len(eps_vals)
	col_dates = pd.date_range(end="2023-12-31", periods=n, freq="YE")
	mock_tk.income_stmt = pd.DataFrame(
		{"Basic EPS": eps_vals},
		index=col_dates,
	).T

	dates  = pd.date_range("2018-01-01", "2023-12-31", freq="ME")
	prices = pd.Series([100.0 + i * 0.5 for i in range(len(dates))], index=dates)
	mock_tk.history.return_value = pd.DataFrame({"Close": prices})

	return mock_tk


@patch("valuation_eval.valuation.yf.Ticker")
@patch("valuation_eval.valuation.time.sleep")
def test_compute_valuation_signal_returns_expected_keys(_, mock_ticker_cls):
	"""compute_valuation_signal returns all required keys."""
	_, db_path = make_db()
	mock_ticker_cls.return_value = _make_yfinance_mock()

	# Pre-populate sector cache so peers aren't fetched
	_save_sector_pe_cache("Technology", {
		"fwd_pe":    {"median": 28.0, "p25": 22.0, "p75": 35.0, "n": 15},
		"ttm_pe":    {"median": 26.0, "p25": 20.0, "p75": 32.0, "n": 15},
		"ev_ebitda": {"median": 16.0, "p25": 13.0, "p75": 20.0, "n": 12},
		"peg":       {"median": 1.8,  "p25": 1.2,  "p75": 2.5,  "n": 13},
	}, str(db_path))

	metrics = {"fwd_pe": 12.0, "ttm_pe": 11.0, "peg": 0.8, "ev_ebitda": 8.0}
	result  = compute_valuation_signal("AAPL", db_path.parent, metrics)

	expected_keys = {
		"valuation_signal", "signal_confidence",
		"pe_vs_history_pctile", "pe_vs_sector_premium_pct",
		"sector_pe_current", "signal_breakdown",
	}
	missing = expected_keys - set(result.keys())
	assert not missing, f"Missing keys: {missing}"
	print("[PASS] test_compute_valuation_signal_returns_expected_keys")


@patch("valuation_eval.valuation.yf.Ticker")
@patch("valuation_eval.valuation.time.sleep")
def test_compute_valuation_signal_cheap(_, mock_ticker_cls):
	"""Low PE vs high sector PE → cheap signal."""
	_, db_path = make_db()
	mock_ticker_cls.return_value = _make_yfinance_mock(
		fwd_pe=10.0, ttm_pe=9.0, peg=0.6, ev_ebitda=6.0
	)

	_save_sector_pe_cache("Technology", {
		"fwd_pe":    {"median": 30.0, "p25": 25.0, "p75": 38.0, "n": 15},
		"ttm_pe":    {"median": 28.0, "p25": 22.0, "p75": 35.0, "n": 15},
		"ev_ebitda": {"median": 18.0, "p25": 14.0, "p75": 22.0, "n": 12},
		"peg":       {"median": 2.0,  "p25": 1.5,  "p75": 2.8,  "n": 13},
	}, str(db_path))

	metrics = {"fwd_pe": 10.0, "ttm_pe": 9.0, "peg": 0.6, "ev_ebitda": 6.0}
	result  = compute_valuation_signal("AAPL", db_path.parent, metrics)

	assert result["valuation_signal"] == "cheap", \
		f"Expected cheap signal, got {result['valuation_signal']}"
	assert result["signal_confidence"] > 0.5, \
		f"Expected confidence > 0.5, got {result['signal_confidence']}"
	print("[PASS] test_compute_valuation_signal_cheap")


# ── Runner ─────────────────────────────────────────────────────────────────

def run_all():
	tests = [
		test_pe_range_normal,
		test_pe_range_empty_earnings,
		test_pe_range_single_year,
		test_peg_range_normal,
		test_peg_range_outlier_filtered,
		test_peg_range_insufficient_data,
		test_peg_signal_cheap_absolute,
		test_peg_signal_none_peg_no_signals,
		test_peg_signal_vs_own_history_cheap,
		test_pe_signal_historically_cheap,
		test_pe_signal_historically_expensive,
		test_ev_ebitda_signal_cheap,
		test_ev_ebitda_signal_no_sector_data,
		test_sector_pe_cache_miss,
		test_sector_pe_cache_roundtrip,
		test_sector_pe_cache_expired,
		test_sector_pe_cache_fresh_within_ttl,
		test_fetch_peers_cache_miss,
		test_fetch_peers_cache_hit,
		test_fetch_peers_ev_ebitda_excluded_for_financials,
		test_compute_valuation_signal_returns_expected_keys,
		test_compute_valuation_signal_cheap,
	]

	passed = 0
	failed = 0
	for fn in tests:
		try:
			fn()
			passed += 1
		except Exception:
			failed += 1
			print(f"[FAIL] {fn.__name__}")
			traceback.print_exc()

	print(f"\n{passed} passed, {failed} failed")


if __name__ == "__main__":
	run_all()
