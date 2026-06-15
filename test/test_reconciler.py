"""
Reconciler unit tests — runs without the full pipeline.

External calls (yfinance opening prices, price cache download) are mocked.
Each test creates its own isolated SQLite DB in a temp directory.

Run:
	python test/test_reconciler.py
"""

import sys
import os
import io
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.schema import (
	init_database,
	insert_portfolio_row,
	insert_recommendation,
	insert_virtual_portfolio,
)
from database.reconciler import (
	reconcile_virtual_portfolio,
	build_divergence_data,
	_build_virtual_portfolio,
)

# ── Shared constants ───────────────────────────────────────────────────────
TICKERS       = ["AAPL", "MSFT", "GOOG"]
STRATEGIES    = ["max_sharpe", "min_variance", "risk_parity", "robust_mv", "target_return"]
REC_DATE      = "2026-06-10"	# last recommendation date
OPENING_DATE  = "2026-06-11"	# REC_DATE + 1 day — where VP is seeded
TODAY         = "2026-06-12"	# "now" passed to reconcile_virtual_portfolio
PORTFOLIO_VAL = 50_000.0		# total real portfolio value

# Synthetic price data returned by mocked external calls
OPEN_PRICES = pd.Series({"AAPL": 150.0, "MSFT": 300.0, "GOOG": 125.0})
RAW_PRICES  = pd.DataFrame(
	{
		"AAPL": [148.0, 150.0, 152.0],
		"MSFT": [295.0, 300.0, 305.0],
		"GOOG": [123.0, 125.0, 127.0],
	},
	index=pd.date_range("2026-06-10", periods=3),
)


# ── Fixture helpers ────────────────────────────────────────────────────────
def make_db():
	"""Create a fresh SQLite DB in a temp directory."""
	tmpdir = tempfile.mkdtemp()
	conn = init_database(Path(tmpdir))
	return conn


def seed_portfolios(conn, dates):
	"""Insert equal-weight portfolio rows for each supplied date."""
	for d in dates:
		for ticker in TICKERS:
			w  = 1.0 / len(TICKERS)
			mv = PORTFOLIO_VAL * w
			insert_portfolio_row(
				conn, d, ticker,
				quantity=100.0,
				avg_cost=mv / 100.0,
				asset_class="Equity",
				market_price=mv / 100.0,
				market_value=mv,
				weight=w,
			)


def seed_recommendations(conn, rec_date=REC_DATE, weight=None):
	"""Insert equal-weight recommendations for all strategies and tickers."""
	w = weight if weight is not None else 1.0 / len(TICKERS)
	for strategy in STRATEGIES:
		for ticker in TICKERS:
			insert_recommendation(
				conn, rec_date, ticker, strategy,
				current_w=w, recommended_w=w,
				action="HOLD", mu=0.08, sigma=0.20,
			)


def seed_vp(conn, vp_date=OPENING_DATE):
	"""Insert a synthetic VP snapshot (one entry per strategy × ticker)."""
	positions = {}
	for strategy in STRATEGIES:
		positions[strategy] = {}
		for ticker in TICKERS:
			price = OPEN_PRICES[ticker]
			w     = 1.0 / len(TICKERS)
			mv    = PORTFOLIO_VAL * w
			positions[strategy][ticker] = {
				"weight":       w,
				"shares":       mv / price,
				"price":        price,
				"market_value": mv,
			}
	insert_virtual_portfolio(conn, vp_date, positions)


def sink():
	"""Return a writable buffer to absorb print output from reconciler."""
	return io.StringIO()


# ── Tests ──────────────────────────────────────────────────────────────────
def test_no_recommendations():
	"""reconcile_virtual_portfolio returns None when recommendations table is empty."""
	conn = make_db()
	seed_portfolios(conn, [TODAY])

	result = reconcile_virtual_portfolio(conn, TODAY, sink())

	assert result is None, f"Expected None with no recs, got {type(result)}"
	print("[PASS] test_no_recommendations")


def test_first_run_seeds_vp():
	"""No prior VP → seeds each strategy at real portfolio value; VP written to DB."""
	conn = make_db()
	seed_portfolios(conn, [REC_DATE, TODAY])
	seed_recommendations(conn)

	with	patch("database.reconciler._fetch_opening_prices", return_value=OPEN_PRICES), \
			patch("database.reconciler.fetch_prices",          return_value=RAW_PRICES):
		result = reconcile_virtual_portfolio(conn, TODAY, sink())

	assert result is not None, "Expected reconcile result, got None"
	assert set(result["curr_vp_values"].keys()) == set(STRATEGIES), \
		f"Expected strategies {STRATEGIES}, got {list(result['curr_vp_values'].keys())}"

	# VP rows should now exist in DB for OPENING_DATE
	stored_strategies = conn.execute(
		"SELECT COUNT(DISTINCT strategy) FROM virtual_portfolio WHERE date = ?",
		(OPENING_DATE,)
	).fetchone()[0]
	assert stored_strategies == len(STRATEGIES), \
		f"Expected {len(STRATEGIES)} strategies in DB, got {stored_strategies}"

	for strategy, val in result["curr_vp_values"].items():
		assert val > 0, f"{strategy} curr_vp_value should be positive"

	print("[PASS] test_first_run_seeds_vp")


def test_second_run_loads_from_db():
	"""VP already exists for opening_date → loads from DB, computes divergence without yfinance."""
	conn = make_db()
	seed_portfolios(conn, [REC_DATE, TODAY])
	seed_recommendations(conn)
	seed_vp(conn, OPENING_DATE)	# pre-seed so _load_or_create_vp takes the DB path

	# _fetch_opening_prices should NOT be called — VP is loaded from DB
	with	patch("database.reconciler.fetch_prices", return_value=RAW_PRICES), \
			patch("database.reconciler._fetch_opening_prices", side_effect=AssertionError("Should not call _fetch_opening_prices")):
		result = reconcile_virtual_portfolio(conn, TODAY, sink())

	assert result is not None, "Expected reconcile result, got None"
	assert len(result["last_vp_values"]) == len(STRATEGIES), \
		f"Expected last_vp_values for all {len(STRATEGIES)} strategies, got {result['last_vp_values']}"
	for strategy, val in result["last_vp_values"].items():
		assert val > 0, f"{strategy} last_vp_value should be positive, got {val}"

	print("[PASS] test_second_run_loads_from_db")


def test_build_divergence_data_structure():
	"""build_divergence_data returns correct top-level keys and per-strategy sub-keys."""
	conn = make_db()
	seed_portfolios(conn, [REC_DATE, TODAY])
	seed_recommendations(conn)
	seed_vp(conn, OPENING_DATE)

	with patch("database.reconciler.fetch_prices", return_value=RAW_PRICES):
		reconcile_result = reconcile_virtual_portfolio(conn, TODAY, sink())

	data = build_divergence_data(conn, TODAY, reconcile_result)

	assert data is not None, "build_divergence_data returned None"

	for key in ("today", "curr_rp_value", "real_return", "strategies", "contributions", "history"):
		assert key in data, f"Missing top-level key '{key}'"

	assert len(data["strategies"]) == len(STRATEGIES), \
		f"Expected {len(STRATEGIES)} strategies, got {len(data['strategies'])}"

	for strategy, s in data["strategies"].items():
		for key in ("curr_vp_value", "virtual_return", "dollar_divergence", "pct_divergence"):
			assert key in s, f"Missing '{key}' in strategy '{strategy}'"

	print("[PASS] test_build_divergence_data_structure")


def test_weight_normalization():
	"""Weights that don't sum to 1.0 are normalized inside _build_virtual_portfolio."""
	# Each ticker has weight 0.5, sum = 1.5 — should normalize to 1.0
	last_rec        = [(t, 0.5, "max_sharpe") for t in TICKERS]
	opening_vp_value = {"max_sharpe": PORTFOLIO_VAL}

	vp = _build_virtual_portfolio(last_rec, opening_vp_value, OPEN_PRICES)

	assert "max_sharpe" in vp, "Expected 'max_sharpe' strategy in VP"
	total_weight = sum(pos["weight"] for pos in vp["max_sharpe"].values())
	assert abs(total_weight - 1.0) < 1e-6, \
		f"Normalised weights should sum to 1.0, got {total_weight:.6f}"

	print("[PASS] test_weight_normalization")


def test_nan_price_skips_ticker():
	"""Ticker with NaN opening price is excluded; valid tickers are still included."""
	prices = pd.Series({"AAPL": 150.0, "MSFT": float("nan"), "GOOG": 125.0})
	last_rec         = [(t, 1.0 / len(TICKERS), "max_sharpe") for t in TICKERS]
	opening_vp_value = {"max_sharpe": PORTFOLIO_VAL}

	vp = _build_virtual_portfolio(last_rec, opening_vp_value, prices)

	assert "max_sharpe" in vp
	assert "MSFT" not in vp["max_sharpe"], "MSFT with NaN price should be excluded from VP"
	assert "AAPL" in vp["max_sharpe"],     "AAPL should be present in VP"
	assert "GOOG" in vp["max_sharpe"],     "GOOG should be present in VP"

	print("[PASS] test_nan_price_skips_ticker")


# ── Runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
	tests = [
		test_no_recommendations,
		test_first_run_seeds_vp,
		test_second_run_loads_from_db,
		test_build_divergence_data_structure,
		test_weight_normalization,
		test_nan_price_skips_ticker,
	]

	failed = 0
	for t in tests:
		try:
			t()
		except Exception as e:
			print(f"[FAIL] {t.__name__}: {e}")
			traceback.print_exc()
			failed += 1

	print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
	sys.exit(1 if failed else 0)
