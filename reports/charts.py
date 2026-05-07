"""
reports/charts.py
==================
Generates all Monte Carlo fan charts as PNG files.
Uses 100 paths for visualisation — statistical outputs
already computed from 10k simulation in Agent 4.

Returns a dict of {chart_name: file_path} for report embedding.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ── Dark theme constants ──────────────────────────────────────────
DARK   = '#0f1117'
PANEL  = '#1a1d27'
CYAN   = '#00d4ff'
ORANGE = '#ff6b35'
GREEN  = '#00ff9f'
RED    = '#ff4757'
GREY   = '#8892a4'
WHITE  = '#e8eaf0'

N_CHART_PATHS = 100
N_STEPS       = 252
DF_T          = 6


def _style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors=GREY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2d3a')
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.grid(True, color='#2a2d3a', linewidth=0.5, alpha=0.7)


def _draw_fan_chart(ax, paths, s_current, title, y_label="Value"):
    """
    Draws a Monte Carlo fan chart on the given axes.
    Shows 200 sample paths and percentile bands.
    """
    t_axis = np.arange(N_STEPS + 1)

    # Sample paths
    sample_idx = np.random.choice(paths.shape[1], min(200, paths.shape[1]), replace=False)
    for i in sample_idx:
        ax.plot(t_axis, paths[:, i], alpha=0.03, linewidth=0.4, color=CYAN)

    # Percentile bands
    p5  = np.percentile(paths, 5,  axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    ax.fill_between(t_axis, p5,  p95, alpha=0.12, color=CYAN,   label='5–95%')
    ax.fill_between(t_axis, p25, p75, alpha=0.22, color=CYAN,   label='25–75%')
    ax.plot(t_axis, p50, color=GREEN,  linewidth=1.5, label='Median')
    ax.plot(t_axis, p5,  color=RED,    linewidth=0.8, linestyle='--', label='5th pct')
    ax.plot(t_axis, p95, color=ORANGE, linewidth=0.8, linestyle='--', label='95th pct')
    ax.axhline(s_current, color=GREY, linewidth=0.8, linestyle=':', label='Current')

    _style_ax(ax, title)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, ncol=3)


def generate_ticker_chart(ticker, s_current, paths, mc_stats, output_path):
    """
    Generates a two-panel chart for a single ticker.
    Left panel: Monte Carlo fan chart.
    Right panel: Terminal price distribution with key statistics.
    """
    S_T   = paths[-1, :]

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # Panel A — fan chart
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_fan_chart(ax1, paths, s_current, f"{ticker} — Monte Carlo (1 Year)")

    # Panel B — terminal distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(S_T, bins=60, color=CYAN, alpha=0.7, edgecolor='none', density=True)

    e_st   = mc_stats.get("expected_value") or S_T.mean()
    var_95 = mc_stats.get("var_95")
    p5_val = np.percentile(S_T, 5)
    p95_val= np.percentile(S_T, 95)

    ax2.axvline(s_current, color=WHITE,  linewidth=1.2, linestyle=':',
                label=f'Current ({s_current:.0f})')
    ax2.axvline(e_st,      color=GREEN,  linewidth=1.5,
                label=f'E[S_T] ({e_st:.0f})')
    ax2.axvline(p5_val,    color=RED,    linewidth=1.0, linestyle='--',
                label=f'5th pct ({p5_val:.0f})')
    ax2.axvline(p95_val,   color=ORANGE, linewidth=1.0, linestyle='--',
                label=f'95th pct ({p95_val:.0f})')

    _style_ax(ax2, f"{ticker} — Terminal Price Distribution")
    ax2.set_xlabel("Terminal Price ($)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)

    fig.suptitle(
        f"{ticker} — 1-Year Monte Carlo Simulation",
        color=WHITE, fontsize=12, fontweight='bold'
    )

    plt.savefig(str(output_path), dpi=120, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)


def generate_portfolio_chart(title, paths, total_value, mc_stats, output_path):
    """
    Generates a two-panel chart for a portfolio simulation.
    Left panel: Portfolio value fan chart in CAD.
    Right panel: Terminal value distribution with key risk statistics.
    """
    S_T = paths[-1, :]

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # Panel A — fan chart
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_fan_chart(
        ax1, paths, total_value,
        f"{title} — Portfolio Value (CAD)",
        y_label="Portfolio Value (CAD)"
    )

    # Panel B — terminal distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(S_T, bins=60, color=CYAN, alpha=0.7, edgecolor='none', density=True)

    e_val  = mc_stats.get("expected_value") or S_T.mean()
    var_95 = mc_stats.get("var_95", 0)
    p5_val = np.percentile(S_T, 5)
    p95_val= np.percentile(S_T, 95)

    ax2.axvline(total_value, color=WHITE,  linewidth=1.2, linestyle=':',
                label=f'Current (${total_value:,.0f})')
    ax2.axvline(e_val,       color=GREEN,  linewidth=1.5,
                label=f'E[V_T] (${e_val:,.0f})')
    ax2.axvline(p5_val,      color=RED,    linewidth=1.0, linestyle='--',
                label=f'5th pct (${p5_val:,.0f})')
    ax2.axvline(p95_val,     color=ORANGE, linewidth=1.0, linestyle='--',
                label=f'95th pct (${p95_val:,.0f})')

    # Annotate VaR
    if var_95:
        ax2.axvline(total_value - var_95, color=RED, linewidth=1.2,
                    linestyle='-.', label=f'VaR95 (${var_95:,.0f})')

    _style_ax(ax2, f"{title} — Terminal Value Distribution")
    ax2.set_xlabel("Terminal Portfolio Value (CAD)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE)

    fig.suptitle(title, color=WHITE, fontsize=12, fontweight='bold')

    plt.savefig(str(output_path), dpi=120, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)


def generate_all_charts(state: dict, charts_dir: Path) -> dict:
	"""
	Generates all charts for the report.
	Returns {chart_name: file_path} dict for HTML embedding.
	"""
	charts_dir.mkdir(parents=True, exist_ok=True)
	chart_paths = {}
	tickers     = state["tickers"]
	print(f"      Generating {len(tickers)} ticker charts...")
 
	# Per-ticker charts — read paths directly from state
	for ticker in tickers:
		ms    = state["mu_sigma"].get(ticker, {})
		mc    = state["mc_current"].get(ticker, {})
		paths = mc.get("sample_paths")          # already in state from Agent 4
		
		if paths is None:
			print(f"      [warn] No paths in state for {ticker}, skipping chart")
			continue
		
		paths = np.array(paths)          # convert back from list if JSON roundtrip
		s0    = ms.get("s_current", 100.0)
		
		output_path = charts_dir / f"mc_{ticker}.png"
		generate_ticker_chart(
			ticker      = ticker,
			s_current   = s0,
			paths       = paths,         # pass directly
			mc_stats    = mc,
			output_path = output_path,
		)

	# Current portfolio chart
	curr_sim  = state.get("mc_port_current") or {}
	curr_paths = curr_sim.get("sample_paths")
	if curr_paths is not None:
		generate_portfolio_chart(
			title       = "Current Portfolio",
			paths       = np.array(curr_paths),
			total_value = state["total_portfolio_value"],
			mc_stats    = curr_sim,
			output_path = charts_dir / "mc_portfolio_current.png",
		)
	else:
		print(f"      [warn] No paths in state for current port, skipping chart")

	# Per-strategy charts
	for strategy, sim in (state.get("mc_port_rebalanced") or {}).items():
		if sim is None:
			continue
		strat_paths = sim.get("sample_paths")
		if strat_paths is None:
			print(f"      [warn] No paths in state for {strategy}, skipping chart")
			continue
		generate_portfolio_chart(
			title       = f"{strategy.replace('_', ' ').title()} Portfolio",
			paths       = np.array(strat_paths),
			total_value = state["total_portfolio_value"],
			mc_stats    = sim,
			output_path = charts_dir / f"mc_portfolio_{strategy}.png",
		)