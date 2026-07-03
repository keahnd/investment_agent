"""
Unit tests for value_rebalancing and compute_composite_score.

Focuses on the key-name bugs that caused MSFT (historically cheap valuation)
to be scored as SELL. Run with:
	python test/test_value_rebalancing.py
"""

import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent3_advisor import compute_composite_score, value_rebalancing


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bl(view_return=0.02, confidence=3, sentiment="bullish"):
	return {
		"view_return":        view_return,
		"view_return_delta":  view_return,
		"confidence":         confidence,
		"sentiment_direction": sentiment,
	}

def _val(signal="cheap", pe_pctile=15, sector_ratio=0.9, confidence=0.8, peg=1.2):
	return {
		"valuation_signal":    signal,
		"signal_confidence":   confidence,
		"pe_vs_history_pctile": pe_pctile,
		"pe_vs_sector_ratio":  sector_ratio,
		"peg":                 peg,
	}

def _fh(gross_margin=0.70):
	return {"gross_margin": gross_margin}

def _ed(consecutive_beats=5):
	return {"consecutive_beats": consecutive_beats}


# ── 1. compute_composite_score tests ─────────────────────────────────────────

def test_composite_score_cheap_msft():
	"""MSFT at historically cheap valuation + positive view → BUY, positive score."""
	result = compute_composite_score(
		bl_view          = _bl(view_return=0.03),
		current_weight   = 0.08,
		valuation_result = _val(signal="cheap", pe_pctile=12),
		earnings_data    = _ed(consecutive_beats=5),
		financial_data   = _fh(gross_margin=0.72),
	)
	assert result["action"] == "BUY", \
		f"Expected BUY for cheap MSFT-like ticker, got {result['action']!r}. Note: {result['note']}"
	assert result["composite_score"] > 0, \
		f"Expected positive composite score, got {result['composite_score']:.3f}"
	print(f"[PASS] test_composite_score_cheap_msft  (score={result['composite_score']:.3f}, action={result['action']})")


def test_composite_score_key_names():
	"""
	pe_vs_history_pctile=10 (cheap) vs pe_percentile_vs_history=90 (expensive).
	compute_composite_score must use the correct key (pe_vs_history_pctile).
	If it reads pe_percentile_vs_history, pe_pctile will be None and the action
	decision will be wrong.
	"""
	val = {
		"valuation_signal":       "cheap",
		"signal_confidence":      0.8,
		"pe_vs_history_pctile":   10,    # correct key — historically CHEAP
		"pe_percentile_vs_history": 90,  # wrong key with opposite value
		"pe_vs_sector_ratio":     0.85,
		"peg":                    1.1,
	}
	result = compute_composite_score(
		bl_view          = _bl(view_return=0.02),
		current_weight   = 0.10,
		valuation_result = val,
		earnings_data    = _ed(),
		financial_data   = _fh(),
	)
	assert result["action"] in ("BUY", "HOLD"), \
		f"Expected BUY/HOLD (cheap by correct key), got {result['action']!r}. " \
		f"Bug: function likely read wrong key pe_percentile_vs_history=90 instead of pe_vs_history_pctile=10."
	print(f"[PASS] test_composite_score_key_names  (action={result['action']}, score={result['composite_score']:.3f})")


def test_composite_score_expensive_with_negative_view():
	"""Genuinely expensive ticker with negative BL view → SELL. Should stay SELL after fixes."""
	result = compute_composite_score(
		bl_view          = _bl(view_return=-0.05, sentiment="bearish"),
		current_weight   = 0.12,
		valuation_result = _val(signal="expensive", pe_pctile=88, sector_ratio=1.4, peg=3.5),
		earnings_data    = _ed(consecutive_beats=1),
		financial_data   = _fh(gross_margin=0.30),
	)
	assert result["action"] in ("SELL", "TRIM"), \
		f"Expected SELL/TRIM for genuinely expensive + bearish ticker, got {result['action']!r}"
	assert result["composite_score"] < 0, \
		f"Expected negative score for expensive ticker, got {result['composite_score']:.3f}"
	print(f"[PASS] test_composite_score_expensive_with_negative_view  (action={result['action']}, score={result['composite_score']:.3f})")


def test_composite_score_cheap_signal_overrides_negative_view():
	"""
	Cheap valuation + at least 1 quality point should produce BUY even if BL view
	is mildly negative. Condition 2: historically_cheap AND quality_score >= 1 → BUY.
	"""
	result = compute_composite_score(
		bl_view          = _bl(view_return=-0.03, sentiment="bearish"),
		current_weight   = 0.08,
		valuation_result = _val(signal="cheap", pe_pctile=18, peg=1.3),
		earnings_data    = _ed(consecutive_beats=5),
		financial_data   = _fh(gross_margin=0.65),
	)
	assert result["action"] == "BUY", \
		f"Expected BUY (cheap + quality >= 1 overrides mild negative view), got {result['action']!r}"
	print(f"[PASS] test_composite_score_cheap_signal_overrides_negative_view  (action={result['action']}, score={result['composite_score']:.3f})")


# ── 2. value_rebalancing tests ────────────────────────────────────────────────

_TICKERS = ["MSFT", "AAPL", "JPM"]

_CURRENT_WEIGHTS = {"MSFT": 0.33, "AAPL": 0.33, "JPM": 0.34}

_BL_VIEWS = {
	"MSFT": _bl(view_return=0.03,  sentiment="bullish"),
	"AAPL": _bl(view_return=0.01,  sentiment="neutral"),
	"JPM":  _bl(view_return=-0.01, sentiment="neutral"),
}

_VALUATION = {
	"MSFT": _val(signal="cheap",     pe_pctile=15, peg=1.2),
	"AAPL": _val(signal="fair",      pe_pctile=50, peg=2.1),
	"JPM":  _val(signal="fair",      pe_pctile=45, peg=1.1),
}

_FH = {
	"MSFT": {"gross_margin": 0.72},
	"AAPL": {"gross_margin": 0.45},
	"JPM":  {"gross_margin": 0.60},
}

_ED = {
	"MSFT": {"consecutive_beats": 6},
	"AAPL": {"consecutive_beats": 3},
	"JPM":  {"consecutive_beats": 2},
}


def test_value_rebalancing_msft_not_sell():
	"""MSFT (cheap valuation, high quality, positive view) should not be trimmed."""
	weights = value_rebalancing(
		tickers          = _TICKERS,
		current_weights  = _CURRENT_WEIGHTS,
		bl_views         = _BL_VIEWS,
		valuation        = _VALUATION,
		financial_health = _FH,
		earnings_data    = _ED,
	)

	msft_new = weights.get("MSFT", 0.0)
	msft_old = _CURRENT_WEIGHTS["MSFT"]

	assert msft_new >= msft_old - 0.01, (
		f"MSFT weight should not decrease for a cheap, high-quality ticker. "
		f"Current: {msft_old:.1%}, New: {msft_new:.1%}"
	)
	print(f"[PASS] test_value_rebalancing_msft_not_sell  "
		  f"(MSFT: {msft_old:.1%} -> {msft_new:.1%})")


def test_value_rebalancing_weights_sum_to_one():
	"""Weights from value_rebalancing must sum to ~1.0 and stay within bounds."""
	weights = value_rebalancing(
		tickers          = _TICKERS,
		current_weights  = _CURRENT_WEIGHTS,
		bl_views         = _BL_VIEWS,
		valuation        = _VALUATION,
		financial_health = _FH,
		earnings_data    = _ED,
	)

	total = sum(weights.values())
	assert abs(total - 1.0) < 1e-3, f"Weights sum to {total:.6f}, expected ~1.0"
	for ticker, w in weights.items():
		assert w >= 0, f"{ticker} has negative weight {w:.4f}"
	print(f"[PASS] test_value_rebalancing_weights_sum_to_one  (sum={total:.6f})")
	print(f"       Weights: { {t: f'{w:.1%}' for t, w in weights.items()} }")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
	tests = [
		test_composite_score_cheap_msft,
		test_composite_score_key_names,
		test_composite_score_expensive_with_negative_view,
		test_composite_score_cheap_signal_overrides_negative_view,
		test_value_rebalancing_msft_not_sell,
		test_value_rebalancing_weights_sum_to_one,
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
