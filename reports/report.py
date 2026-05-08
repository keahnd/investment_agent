"""
reports/report.py
==================
Generates the weekly portfolio PDF report using fpdf2.
Receives final_state and chart_paths, produces a dated PDF file.

Sections:
  1. Cover page
  2. Portfolio snapshot
  3. Virtual portfolio divergence
  4. Model outputs per asset
  5. Monte Carlo charts
  6. Black-Litterman view table
  7. Recommendation table
  8. Commentary
"""

from fpdf import FPDF
from pathlib import Path
from datetime import date


# ── Page layout constants ─────────────────────────────────────────
PAGE_W      = 210    # A4 width mm
PAGE_H      = 297    # A4 height mm
MARGIN      = 15     # left/right margin mm
CONTENT_W   = PAGE_W - 2 * MARGIN   # 180mm usable width

# ── Colour palette (RGB) ──────────────────────────────────────────
C_BLACK     = (15,  17,  23)    # #0f1117
C_PANEL     = (26,  29,  39)    # #1a1d27
C_CYAN      = (0,   212, 255)   # #00d4ff
C_GREEN     = (0,   255, 159)   # #00ff9f
C_RED       = (255, 71,  87)    # #ff4757
C_ORANGE    = (255, 107, 53)    # #ff6b35
C_GREY      = (136, 146, 164)   # #8892a4
C_WHITE     = (232, 234, 240)   # #e8eaf0
C_DARK_GREY = (42,  45,  58)    # #2a2d3a

# ── Font sizes ────────────────────────────────────────────────────
FS_TITLE    = 22
FS_H1       = 16
FS_H2       = 12
FS_BODY     = 9
FS_SMALL    = 8
FS_TINY     = 7


def _safe(text) -> str:
    """
    Replaces common Unicode characters with Latin-1 equivalents.
    Prevents fpdf2 UnicodeEncodeError when using built-in Helvetica font.
    """
    if text is None:
        return "N/A"
    text = str(text)
    replacements = {
        "\u2014": "-",    # em dash —
        "\u2013": "-",    # en dash –
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "*",    # bullet
        "\u25cf": "*",    # filled circle *
        "\u25cb": "o",    # empty circle .
        "\u00b0": "deg",  # degree sign
        "\u00b2": "2",    # superscript 2
        "\u00b3": "3",    # superscript 3
        "\u00e9": "e",    # é
        "\u00e8": "e",    # è
        "\u00e0": "a",    # à
        "\u00fc": "u",    # ü
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Catch anything else outside Latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


class PortfolioReport(FPDF):
	"""
	Extends FPDF with helper methods for consistent styling.
	Dark theme throughout matching the chart palette.
	"""

	def __init__(self):
		super().__init__()
		self.set_auto_page_break(auto=True, margin=20)
		self.set_margins(MARGIN, MARGIN, MARGIN)

	# ── Styling helpers ───────────────────────────────────────────

	def set_bg(self):
		"""Fill current page background with dark colour."""
		self.set_fill_color(*C_BLACK)
		self.rect(0, 0, PAGE_W, PAGE_H, 'F')

	def h1(self, text):
		"""Section header."""
		self.set_font("Helvetica", "B", FS_H1)
		self.set_text_color(*C_CYAN)
		self.ln(4)
		self.cell(0, 10, _safe(text), ln=True)
		# Underline
		self.set_draw_color(*C_CYAN)
		self.set_line_width(0.3)
		x = self.get_x()
		y = self.get_y()
		self.line(MARGIN, y, PAGE_W - MARGIN, y)
		self.ln(3)
		self.set_text_color(*C_WHITE)

	def h2(self, text):
		"""Subsection header."""
		self.set_font("Helvetica", "B", FS_H2)
		self.set_text_color(*C_ORANGE)
		self.ln(2)
		self.cell(0, 8, _safe(text), ln=True)
		self.set_text_color(*C_WHITE)

	def body(self, text, size=FS_BODY):
		"""Regular body text with auto line wrap."""
		self.set_font("Helvetica", "", size)
		self.set_text_color(*C_WHITE)
		self.multi_cell(0, 5, _safe(text))
		self.ln(1)

	def small(self, text):
		self.body(text, size=FS_SMALL)

	def divider(self):
		"""Thin horizontal rule."""
		self.set_draw_color(*C_DARK_GREY)
		self.set_line_width(0.2)
		self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
		self.ln(3)

	def kv(self, label, value, label_w=60):
		"""Key-value pair on one line."""
		self.set_font("Helvetica", "B", FS_BODY)
		self.set_text_color(*C_GREY)
		self.cell(label_w, 5, label)
		self.set_font("Helvetica", "", FS_BODY)
		self.set_text_color(*C_WHITE)
		self.cell(0, 5, _safe(str(value)) if value is not None else "N/A", ln=True)

	def flag_cell(self, value, good_above=None, bad_below=None, fmt=None):
		"""
		Writes a coloured cell based on threshold checks.
		Green if above good_above, red if below bad_below, white otherwise.
		"""
		if value is None:
			self.set_text_color(*C_GREY)
			self.cell(18, 6, "N/A", border=1, align="R")
			self.set_text_color(*C_WHITE)
			return

		color = C_WHITE
		if good_above is not None and value > good_above:
			color = C_GREEN
		elif bad_below is not None and value < bad_below:
			color = C_RED

		text = fmt.format(value) if fmt else str(value)
		self.set_text_color(*color)
		self.cell(18, 6, _safe(text), border=1, align="R")
		self.set_text_color(*C_WHITE)

	def action_cell(self, action, width=22):
		"""Writes BUY/SELL/HOLD with colour coding."""
		colors = {"BUY": C_GREEN, "SELL": C_RED, "HOLD": C_GREY}
		self.set_text_color(*colors.get(action, C_WHITE))
		self.set_font("Helvetica", "B", FS_TINY)
		self.cell(width, 6, action or "N/A", border=1, align="C")
		self.set_text_color(*C_WHITE)
		self.set_font("Helvetica", "", FS_BODY)

	def table_header(self, cols):
		"""
		Writes a table header row.
		cols: list of (label, width) tuples
		"""
		self.set_fill_color(*C_PANEL)
		self.set_font("Helvetica", "B", FS_TINY)
		self.set_text_color(*C_CYAN)
		for label, width in cols:
			self.cell(width, 6, label, border=1, fill=True, align="C")
		self.ln()
		self.set_text_color(*C_WHITE)
		self.set_font("Helvetica", "", FS_BODY)

	def page_header_bar(self, title):
		"""Thin coloured bar at top of continuation pages."""
		self.set_fill_color(*C_PANEL)
		self.rect(0, 0, PAGE_W, 12, 'F')
		self.set_font("Helvetica", "B", FS_SMALL)
		self.set_text_color(*C_CYAN)
		self.set_y(3)
		self.cell(0, 6, title, align="C", ln=True)
		self.set_y(16)
		self.set_text_color(*C_WHITE)

	def header(self):
		pass   # handled manually per section

	def footer(self):
		self.set_y(-12)
		self.set_font("Helvetica", "", FS_TINY)
		self.set_text_color(*C_GREY)
		self.cell(0, 5, f"Page {self.page_no()}", align="C")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def _section_cover(pdf: PortfolioReport, state: dict):
    """Cover page with run metadata and macro snapshot."""
    pdf.add_page()
    pdf.set_bg()

    # Logo area / top accent bar
    pdf.set_fill_color(*C_CYAN)
    pdf.rect(0, 0, PAGE_W, 4, 'F')

    # Title block
    pdf.set_y(40)
    pdf.set_font("Helvetica", "B", FS_TITLE)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(0, 14, "Portfolio Report", align="C", ln=True)

    pdf.set_font("Helvetica", "", FS_H2)
    pdf.set_text_color(*C_GREY)
    pdf.cell(0, 8, f"Week of {state['run_date']}", align="C", ln=True)
    pdf.cell(0, 8, state["user_name"].replace("_", " ").title(), align="C", ln=True)

    pdf.ln(10)
    pdf.divider()
    pdf.ln(6)

    # Portfolio summary box
    total = state.get("total_portfolio_value", 0)
    rate  = state.get("usd_cad_rate", 1.36)
    pdf.set_font("Helvetica", "B", FS_H1)
    pdf.set_text_color(*C_CYAN)
    pdf.cell(0, 10, f"${total:,.2f} CAD", align="C", ln=True)
    pdf.set_font("Helvetica", "", FS_SMALL)
    pdf.set_text_color(*C_GREY)
    pdf.cell(0, 6, f"USD/CAD rate: {rate:.4f}", align="C", ln=True)

    pdf.ln(10)
    pdf.divider()
    pdf.ln(6)

    # Macro snapshot
    pdf.set_font("Helvetica", "B", FS_H2)
    pdf.set_text_color(*C_ORANGE)
    pdf.cell(0, 8, "Macro Snapshot", align="C", ln=True)
    pdf.ln(4)

    cape = state.get("cape", 25.0)
    fg   = state.get("fear_greed") or {}
    aaii = state.get("aaii_sentiment") or {}

    macro_items = [
        ("Shiller CAPE",          f"{cape:.1f}",                              cape > 30),
        ("Fear & Greed Index",    f"{fg.get('score', 'N/A'):.2f} — {fg.get('rating', 'N/A')}", False),
        ("AAII Bullish",          aaii.get("bullish", "N/A"),                 False),
        ("AAII Bearish",          aaii.get("bearish", "N/A"),                 False),
        ("AAII Bull-Bear Spread", aaii.get("bull_bear_spread", "N/A"),        False),
    ]

    for label, value, warn in macro_items:
        pdf.set_x(MARGIN + 30)
        pdf.set_font("Helvetica", "B", FS_BODY)
        pdf.set_text_color(*C_GREY)
        pdf.cell(70, 6, label)
        pdf.set_font("Helvetica", "", FS_BODY)
        pdf.set_text_color(*C_RED if warn else C_WHITE)
        pdf.cell(0, 6, _safe(str(value)), ln=True)

    # Bottom accent bar
    pdf.set_fill_color(*C_CYAN)
    pdf.rect(0, PAGE_H - 4, PAGE_W, 4, 'F')


def _section_portfolio_snapshot(pdf: PortfolioReport, state: dict):
	"""
	Section 1 — Portfolio Snapshot.
	Table: ticker, weight, market value, valuation metrics.
	"""
	pdf.add_page()
	pdf.set_bg()
	pdf.h1("1. Portfolio Snapshot")

	tickers   = state["tickers"]
	weights   = state["current_weights"]
	valuation = state.get("valuation") or {}
	total_val = state.get("total_portfolio_value", 0)
	rows      = state.get("portfolio_rows") or []
	row_map   = {r["ticker"]: r for r in rows}

	# Column widths — sum to CONTENT_W (180mm)
	cols = [
		("Ticker",   22),
		("Shares",   18),
		("Price",    22),
		("Mkt Val",  22),
		("Weight",   18),
		("Fwd P/E",  18),
		("PEG",      18),
		("EV/EBITDA",18),
		("Analyst",  20),
	]

	pdf.table_header(cols)

	for ticker in tickers:
		row = row_map.get(ticker, {})
		val = valuation.get(ticker, {})
		w   = weights.get(ticker, 0)

		pdf.set_font("Helvetica", "B", FS_SMALL)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(22, 6, ticker, border=1)

		pdf.set_font("Helvetica", "", FS_SMALL)
		pdf.set_text_color(*C_WHITE)
		pdf.cell(18, 6, f"{row.get('shares', 0):.2f}",     border=1, align="R")
		pdf.cell(22, 6, f"${row.get('avg_cost', 0):.2f}",  border=1, align="R")
		pdf.cell(22, 6, f"${row.get('market_value', 0):,.0f}", border=1, align="R")
		pdf.cell(18, 6, f"{w:.1%}",                        border=1, align="R")

		pdf.flag_cell(val.get("fwd_pe"),   bad_below=10,  good_above=None, fmt="{:.1f}x")
		pdf.flag_cell(val.get("peg"),      bad_below=None, good_above=None, fmt="{:.2f}")
		pdf.flag_cell(val.get("ev_ebitda"),bad_below=None, good_above=None, fmt="{:.1f}x")

		rec = val.get("analyst_rec") or "N/A"
		pdf.set_text_color(*C_GREEN if "Buy" in rec else C_WHITE)
		pdf.cell(20, 6, _safe(rec[:10]), border=1, align="C")
		pdf.set_text_color(*C_WHITE)
		pdf.ln()
        
		if pdf.get_y() > PAGE_H - 90:
			pdf.add_page()
			pdf.set_bg()
			pdf.page_header_bar("1. Portfolio Snapshot (continued)")

	# Total row
	pdf.set_font("Helvetica", "B", FS_SMALL)
	pdf.set_text_color(*C_CYAN)
	pdf.cell(22, 6, "TOTAL", border=1)
	pdf.cell(18 + 22, 6, "", border=1)
	pdf.cell(22, 6, f"${total_val:,.0f}", border=1, align="R")
	pdf.cell(18, 6, "100.0%", border=1, align="R")
	pdf.ln(10)


def _section_virtual_portfolio(pdf: PortfolioReport, divergence_data: dict):
    """
    Section 2 — Virtual Portfolio Divergence.
    Reads from divergence_data dict built by build_divergence_data().
    Shows placeholder if data not available.
    """
    pdf.add_page()
    pdf.set_bg()
    pdf.h1("2. Virtual Portfolio Divergence")

    if not divergence_data:
        pdf.set_text_color(*C_GREY)
        pdf.set_font("Helvetica", "I", FS_BODY)
        pdf.multi_cell(0, 6,
            "Virtual portfolio divergence data not available. "
            "This section populates after the first full pipeline run "
            "with prior recommendations in the database."
        )
        return

    # ── Top-line summary ──────────────────────────────────────────
    curr_rp = divergence_data["curr_rp_value"]
    real_ret = divergence_data.get("real_return")

    pdf.kv("Real Portfolio Value",
           f"${curr_rp:,.2f} CAD")
    pdf.kv("Real Portfolio Return",
           f"{real_ret:+.2%}" if real_ret is not None else "N/A")
    pdf.ln(4)

    # ── Per-strategy divergence table ─────────────────────────────
    strategies = divergence_data.get("strategies") or {}

    if strategies:
        cols = [
            ("Strategy",     50),
            ("VP Value",     35),
            ("VP Return",    30),
            ("$ Divergence", 35),
            ("% Divergence", 30),
        ]
        pdf.table_header(cols)

        for strategy, data in strategies.items():
            curr_vp  = data.get("curr_vp_value", 0)
            vp_ret   = data.get("virtual_return")
            dollar_d = data.get("dollar_divergence", 0)
            pct_d    = data.get("pct_divergence", 0)

            pdf.set_font("Helvetica", "", FS_SMALL)
            pdf.set_text_color(*C_WHITE)
            pdf.cell(50, 6,
                     _safe(strategy.replace("_", " ").title()), border=1)
            pdf.cell(35, 6, f"${curr_vp:,.2f}", border=1, align="R")

            ret_color = C_GREEN if vp_ret and vp_ret > 0 else C_RED
            pdf.set_text_color(*ret_color)
            pdf.cell(30, 6,
                     f"{vp_ret:+.2%}" if vp_ret is not None else "N/A",
                     border=1, align="R")

            div_color = C_GREEN if dollar_d > 0 else C_RED
            pdf.set_text_color(*div_color)
            pdf.cell(35, 6, f"${dollar_d:+,.2f}", border=1, align="R")
            pdf.cell(30, 6,
                     f"{pct_d:+.2%}" if pct_d is not None else "N/A",
                     border=1, align="R")
            pdf.set_text_color(*C_WHITE)
            pdf.ln()

    # ── Cumulative history ────────────────────────────────────────
    history = divergence_data.get("history") or []
    if history:
        pdf.ln(6)
        pdf.h2("Cumulative Divergence History (last 20 entries)")

        cols = [
            ("Date",        30),
            ("Strategy",    50),
            ("VP Value",    32),
            ("RP Value",    32),
            ("Divergence",  36),
        ]
        pdf.table_header(cols)

        for row in history[:20]:
            d, strat, vp, rp = row[0], row[1], row[2], row[3]
            divergence = vp - rp

            pdf.set_font("Helvetica", "", FS_TINY)
            pdf.set_text_color(*C_WHITE)
            pdf.cell(30, 5, _safe(str(d)),    border=1)
            pdf.cell(50, 5,
                     _safe(str(strat).replace("_", " ").title()),
                     border=1)
            pdf.cell(32, 5, f"${vp:,.2f}",  border=1, align="R")
            pdf.cell(32, 5, f"${rp:,.2f}",  border=1, align="R")

            div_color = C_GREEN if divergence > 0 else C_RED
            pdf.set_text_color(*div_color)
            pdf.cell(36, 5,
                     f"${divergence:+,.2f}", border=1, align="R")
            pdf.set_text_color(*C_WHITE)
            pdf.ln()


def _section_model_outputs(pdf: PortfolioReport, state: dict, charts_dir):
	"""
	Section 5 — Model Outputs Per Asset.
	Factor model, GARCH, financial health, earnings per ticker.
	"""
	pdf.add_page()
	pdf.set_bg()
	pdf.h1("5. Model Outputs Per Asset")

	tickers  = state["tickers"]

	for ticker in tickers:
		fr  = (state.get("factor_results")   or {}).get(ticker, {})
		gr  = (state.get("garch_results")    or {}).get(ticker, {})
		ms  = (state.get("mu_sigma")         or {}).get(ticker, {})
		fh  = (state.get("financial_health") or {}).get(ticker, {})
		ed  = (state.get("earnings_data")    or {}).get(ticker, {})
		val = (state.get("valuation")        or {}).get(ticker, {})
		bl_views     = (state.get("bl_views") or {}).get(ticker, {})
		posterior_mu = (state.get("posterior_mu") or {}).get(ticker, {})
		mu_sigma     = (state.get("mu_sigma") or {}).get(ticker, {})

		# Check page space — add new page if less than 70mm remaining
		if pdf.get_y() > PAGE_H - 90:
			pdf.add_page()
			pdf.set_bg()
			pdf.page_header_bar("5. Model Outputs Per Asset (continued)")

		pdf.h2(f"{ticker} — {_safe(val.get('sector', ''))} | {_safe(val.get('industry', ''))}")

		# Two-column layout using fixed positions
		left_x  = MARGIN
		right_x = MARGIN + 92

		start_y = pdf.get_y()

		# Left column — factor model and GARCH
		pdf.set_xy(left_x, start_y)
		pdf.set_font("Helvetica", "B", FS_SMALL)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(85, 5, "Factor Model", ln=True)
		pdf.set_text_color(*C_WHITE)

		factor_items = [
			("Alpha (daily)",    fr.get("alpha_daily"),  "{:.6f}"),
			("Beta MKT",         fr.get("b_MKT"),        "{:.3f}"),
			("Beta SMB",         fr.get("b_SMB"),        "{:.3f}"),
			("Beta HML",         fr.get("b_HML"),        "{:.3f}"),
			("R²",               fr.get("r2"),           "{:.3f}"),
			("p-val alpha",      fr.get("p_val_alpha"),  "{:.3f}"),
			("Mu (annual)",      ms.get("mu_annual"),    "{:.2%}"),
		]

		for label, val_f, fmt in factor_items:
			pdf.set_x(left_x)
			pdf.set_font("Helvetica", "", FS_SMALL)
			pdf.set_text_color(*C_GREY)
			pdf.cell(45, 5, label)
			pdf.set_text_color(*C_WHITE)
			text = fmt.format(val_f) if val_f is not None else "N/A"
			pdf.cell(40, 5, _safe(text), ln=True)

		# GARCH
		pdf.set_x(left_x)
		pdf.set_font("Helvetica", "B", FS_SMALL)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(85, 5, "GARCH(1,1)", ln=True)
		pdf.set_text_color(*C_WHITE)

		regime_color = C_RED if gr.get("vol_regime") == "elevated" else C_GREEN
		garch_items = [
			("Persistence",       gr.get("garch_persist"),     "{:.4f}", None),
			("Long-run vol",      gr.get("garch_long_run_vol"),"{:.2%}",  None),
			("Current vol (ann)", ms.get("sigma_annual"),      "{:.2%}",  None),
			("Vol regime",        gr.get("vol_regime"),        "{}",      regime_color),
		]

		for label, val_g, fmt, color in garch_items:
			pdf.set_x(left_x)
			pdf.set_font("Helvetica", "", FS_SMALL)
			pdf.set_text_color(*C_GREY)
			pdf.cell(45, 5, label)
			pdf.set_text_color(*(color if color else C_WHITE))
			text = fmt.format(val_g) if val_g is not None else "N/A"
			pdf.cell(40, 5, text, ln=True)
   
		# BL Views
		pdf.set_x(left_x)
		pdf.set_font("Helvetica", "B", FS_SMALL)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(85, 5, "Black Litterman Views", ln=True)
		pdf.set_text_color(*C_WHITE)
  
		bl_items = [ # Can add color later
			("Prior Return", 		mu_sigma.get("mu_annual"), 						"{:.2%}"),
			("Additional Return", 	bl_views.get("view_return", 0), 				"{:.2%}"),
			("Confidence",     		"*" * bl_views.get("confidence", 1), 			"{}"),
			("Conflict", 			bl_views.get("conflict", False), 				"{}"),
			("Sentiment", 			bl_views.get("sentiment_direction", "neutral"),	"{}"),
			("Valuation Signal", 	bl_views.get("valuation_signal", "neutral"), 	"{}"),
		]
		
		for label, val_f, fmt in bl_items:
			pdf.set_x(left_x)
			pdf.set_font("Helvetica", "", FS_SMALL)
			pdf.set_text_color(*C_GREY)
			pdf.cell(45, 5, label)
			pdf.set_text_color(*C_WHITE)
			text = fmt.format(val_f) if val_f is not None else "N/A"
			pdf.cell(40, 5, _safe(text), ln=True)

		# Full reasoning
		reasoning = bl_views.get("reasoning", "")
		if reasoning:
			pdf.set_font("Helvetica", "B", FS_SMALL)
			pdf.set_text_color(*C_CYAN)
			pdf.cell(0, 5, "Reasoning", ln=True)
			pdf.set_font("Helvetica", "", FS_SMALL)
			pdf.set_text_color(*C_WHITE)
			pdf.multi_cell(0, 5, _safe(reasoning))
			pdf.ln(2)

		end_y = pdf.get_y()

		# Right column — financial health and earnings
		right_y = start_y
		pdf.set_xy(right_x, right_y)
		pdf.set_font("Helvetica", "B", FS_SMALL)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(85, 5, "Financial Health", ln=True)
		pdf.set_text_color(*C_WHITE)

		health_items = [
			("Revenue Growth 1yr",  fh.get("revenue_growth_1yr"),  "{:.1%}"),
			("Revenue Growth 3yr",  fh.get("revenue_growth_3yr"),  "{:.1%}"),
			("Gross Margin",        fh.get("gross_margin_current"),"{:.1%}"),
			("Margin Trend",        fh.get("gross_margin_trend"),  "{}"),
			("FCF Margin",          fh.get("fcf_margin"),          "{:.1%}"),
			("Earnings Quality",    fh.get("earnings_quality"),    "{:.2f}"),
			("Debt / FCF",          fh.get("debt_to_fcf"),         "{:.1f}x"),
			("ROIC",                fh.get("roic"),                "{:.1%}"),
		]

		for label, val_h, fmt in health_items:
			pdf.set_x(right_x)
			pdf.set_font("Helvetica", "", FS_SMALL)
			pdf.set_text_color(*C_GREY)
			pdf.cell(50, 5, label)
			pdf.set_text_color(*C_WHITE)
			text = fmt.format(val_h) if val_h is not None else "N/A"
			pdf.cell(35, 5, _safe(text), ln=True)

		# Earnings summary
		pdf.set_x(right_x)
		pdf.set_font("Helvetica", "B", FS_SMALL)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(85, 5, "Earnings", ln=True)

		next_e   = ed.get("next_earnings_date", "N/A")
		avg_surp = ed.get("avg_eps_surprise_pct")
		consec   = ed.get("consecutive_beats", 0)

		earnings_items = [
			("Next Earnings",      next_e,   "{}"),
			("Avg EPS Surprise",   avg_surp, "{:.1f}%"),
			("Consecutive Beats",  consec,   "{:.0f}"),
		]

		for label, val_e, fmt in earnings_items:
			pdf.set_x(right_x)
			pdf.set_font("Helvetica", "", FS_SMALL)
			pdf.set_text_color(*C_GREY)
			pdf.cell(50, 5, label)
			pdf.set_text_color(*C_WHITE)
			text = fmt.format(val_e) if val_e is not None else "N/A"
			pdf.cell(35, 5, _safe(text), ln=True)
   
		# Valuation summary
		pdf.set_x(right_x)
		pdf.set_font("Helvetica", "B", FS_SMALL)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(85, 5, "Valuation", ln=True)

		valuation_items = [
			("PEG",   			val.get("peg", 0), 				"{:.2f}"),
			("Fwd PE",  		val.get("fwd_pe", 0),   		"{:.1f}x"),
   			("Ttm PE",  		val.get("ttm_pe", 0),   		"{:.1f}x"),
			("EV EBITDA",  		val.get("ev_ebitda", 0),   		"{:.1f}x"),
			("Target Price",	val.get("target_price", 0),   	"{:.2f}"),
			("Earnings Growth",	val.get("earnings_growth", 0),	"{:.2%}"),
			("Revenue Growth",	val.get("revenue_growth", 0),   "{:.2%}"),
		]

		for label, val_e, fmt in valuation_items:
			pdf.set_x(right_x)
			pdf.set_font("Helvetica", "", FS_SMALL)
			pdf.set_text_color(*C_GREY)
			pdf.cell(50, 5, label)
			pdf.set_text_color(*C_WHITE)
			text = fmt.format(val_e) if val_e is not None else "N/A"
			pdf.cell(35, 5, _safe(text), ln=True)
   
		# ── Per-ticker charts ─────────────────────────────────────────
		pdf.set_xy(left_x, end_y)
		pdf.set_font("Helvetica", "B", FS_SMALL)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(85, 5, f"Monte Carlo Simulation", ln=True)
		chart = charts_dir / f"mc_{ticker}.png" if charts_dir else None
		if not chart or not chart.exists():
			continue

		# Key stats below chart
		mc = (state.get("mc_current") or {}).get(ticker, {})
		ms = (state.get("mu_sigma")   or {}).get(ticker, {})
		if mc:
			pdf.ln(3)
			stat_items = [
				("Current Price",  f"${ms.get('s_current', 0):.2f}"),
				("Posterior Mu",   f"{(state.get('posterior_mu') or {}).get(ticker, 0):.2%}"),
				("Sigma (annual)", f"{ms.get('sigma_annual', 0):.2%}"),
				("Median S_T",     f"${mc.get('pct', 0)[3]:.2f}"),
				("VaR 95%",        f"${mc.get('VaR_95', 0):.2f}"),
				("P(Profit)",        f"{mc.get('prob_up', 0):.1%}"),
			]
			# Print in two rows of three
			pdf.set_font("Helvetica", "", FS_SMALL)
			for j, (label, value) in enumerate(stat_items):
				if j % 3 == 0 and j > 0:
					pdf.ln()
					pdf.set_x(MARGIN)
				pdf.set_text_color(*C_GREY)
				pdf.cell(28, 5, label)
				pdf.set_text_color(*C_WHITE)
				pdf.cell(32, 5, value)
			pdf.ln()

		pdf.image(chart, x=MARGIN, w=CONTENT_W)

		# Move to below both columns
		new_y = max(pdf.get_y(), start_y + len(factor_items) * 5 + len(garch_items) * 5 + 20)
		pdf.set_y(new_y + 4)
		pdf.divider()


def _section_simulation_charts(pdf: PortfolioReport, state: dict, charts_dir: dict):
    """
    Section 3 — Monte Carlo Charts.
    Current portfolio chart, per-strategy charts, per-ticker charts.
    Includes a simulation statistics comparison table.
    """
    pdf.add_page()
    pdf.set_bg()
    pdf.h1("3. Portfolio Simulation")

    # ── Simulation statistics comparison table ────────────────────
    pdf.h2("Portfolio Simulation Summary")
    pdf.set_font("Helvetica", "", FS_SMALL)
    pdf.multi_cell(0, 5,
        "All simulations use 1-year horizon (252 trading days), "
        "Student-t innovations (df=6), and GARCH-derived volatility. "
        "Dollar values in CAD."
    )
    pdf.ln(3)

    # Build comparison table — current + all strategies
    mc_curr  = state.get("mc_port_current") or {}
    mc_strat = state.get("mc_port_rebalanced") or {}
    total    = state.get("total_portfolio_value", 0)

    all_sims = {"Current": mc_curr}
    all_sims.update({
        s.replace("_", " ").title(): v
        for s, v in mc_strat.items() if v
    })    
    cols = [
        ("Strategy",      40),
        ("Median",        25),
        ("E[Value]",      25),
        ("VaR 95% $",     25),
        ("CVaR 95% $",    25),
        ("P(Profit)",       20),
        ("5th Pct",       20),
    ]

    pdf.table_header(cols)

    for label, sim in all_sims.items():
        is_current = label == "Current"
        pdf.set_font("Helvetica", "B" if is_current else "", FS_TINY)
        pdf.set_text_color(*C_CYAN if is_current else C_WHITE)
        pdf.cell(40, 6, label[:20], border=1)

        pdf.set_font("Helvetica", "", FS_TINY)
        pdf.set_text_color(*C_WHITE)
        pdf.cell(25, 6, f"${sim.get('pct', 0)[3]:,.0f}",           border=1, align="R")
        pdf.cell(25, 6, f"${sim.get('E_ST', 0):,.0f}", border=1, align="R")

        var  = sim.get("VaR_95", 0)
        cvar = sim.get("CVaR_95", 0)
        pdf.set_text_color(*C_RED)
        pdf.cell(25, 6, f"${var:,.0f}",  border=1, align="R")
        pdf.cell(25, 6, f"${cvar:,.0f}", border=1, align="R")

        p_up = sim.get("prob_up", 0)
        pdf.set_text_color(*C_RED if p_up < 0 else C_WHITE)
        pdf.cell(20, 6, f"{p_up:.1%}", border=1, align="R")

        pdf.set_text_color(*C_WHITE)
        pdf.cell(20, 6, f"${sim.get('pct', 0)[1]:,.0f}", border=1, align="R")
        pdf.ln()

    pdf.ln(6)

    # ── Current portfolio chart ───────────────────────────────────
    curr_chart = charts_dir / "mc_portfolio_current.png" if charts_dir else None
    if curr_chart and curr_chart.exists():
        pdf.add_page()
        pdf.set_bg()
        pdf.page_header_bar("3. Portfolio Simulation")
        pdf.h2("Current Portfolio — 1-Year Simulation")
        pdf.image(curr_chart, x=MARGIN, w=CONTENT_W)
        pdf.ln(4)

    # ── Per-strategy charts ───────────────────────────────────────
    strategies = list(mc_strat.keys())
    for i, strategy in enumerate(strategies):
        chart_key = f"mc_portfolio_{strategy}"
        chart     = charts_dir / f"{chart_key}.png" if charts_dir else None
        if not chart or not chart.exists():
            continue
        pdf.add_page()
        pdf.set_bg()
        pdf.page_header_bar("4. Monte Carlo Simulation")
        pdf.h2(f"{strategy.replace('_', ' ').title()} Portfolio — 1-Year Simulation")
        pdf.image(chart, x=MARGIN, w=CONTENT_W)
        pdf.ln(4)        


def _section_recommendation_table(pdf: PortfolioReport, state: dict):
	"""
	Section 2 — Recommendation Table.
	One row per ticker, one column per strategy plus consensus.
	"""
	pdf.add_page()
	pdf.set_bg()
	pdf.h1("2. Recommendation Table")

	rec_table = state.get("recommendation_table") or []
	strategies = list((state.get("recommended_weights") or {}).keys())

	# Header row — ticker + current + one col per strategy + consensus
	strategy_w = min(28, int((CONTENT_W - 20 - 20 - 20) / (len(strategies) + 1)))
	ticker_w   = 20
	current_w  = 20
	consensus_w= 20

	header_cols = [("Ticker", ticker_w), ("Current", current_w)]
	for s in strategies:
		label = _safe(s.replace("_", " ").replace("max", "Max").replace("min", "Min"))
		header_cols.append((label, strategy_w))
	header_cols.append(("Consensus", consensus_w))
	pdf.table_header(header_cols)

	for row in rec_table:
		ticker = row["ticker"]

		pdf.set_font("Helvetica", "B", FS_TINY)
		pdf.set_text_color(*C_CYAN)
		pdf.cell(ticker_w, 6, ticker, border=1)

		pdf.set_font("Helvetica", "", FS_TINY)
		pdf.set_text_color(*C_WHITE)
		pdf.cell(current_w, 6,
					f"{row.get('current_weight', 0):.1%}", border=1, align="R")

		for strategy in strategies:
			w      = row.get(f"{strategy}_weight", 0)
			action = row.get(f"{strategy}_action", "HOLD")
			delta  = row.get(f"{strategy}_delta", 0)

			# Color based on action
			color = C_GREEN if action == "BUY" else C_RED if action == "SELL" else C_GREY
			pdf.set_text_color(*color)
			pdf.set_font("Helvetica", "B" if action != "HOLD" else "", FS_TINY)
			pdf.cell(strategy_w, 6,
						f"{w:.1%} ({delta:+.1%})", border=1, align="R")

		# Consensus
		consensus = row.get("consensus_action", "HOLD")
		pdf.set_font("Helvetica", "B", FS_TINY)
		pdf.action_cell(consensus, width=consensus_w)
		pdf.ln()
        
		if pdf.get_y() > PAGE_H - 90:
			pdf.add_page()
			pdf.set_bg()
			pdf.page_header_bar("2. Recommendation Table (continued)")

	# Legend
	pdf.ln(6)
	pdf.set_font("Helvetica", "", FS_TINY)
	pdf.set_text_color(*C_GREY)
	pdf.cell(0, 5,
		"Format: weight (delta from current) | "
		f"BUY threshold: >{state.get('action_threshold', 0.02):.0%} | "
		f"SELL threshold: <-{state.get('action_threshold', 0.02):.0%}",
		ln=True
	)


def _section_advisory_commentary(pdf: PortfolioReport, state: dict):
    """
    Section 4 — Commentary.
    Quantitative anomaly flags, risk commentary, advisory commentary.
    """
    pdf.add_page()
    pdf.set_bg()
    pdf.h1("4. Advisory Commentary")

    # Advisory commentary — main section
    advisory = state.get("advisory_commentary")
    if advisory:
        pdf.h2("Portfolio Advisory")
        pdf.set_font("Helvetica", "", FS_BODY)
        pdf.set_text_color(*C_WHITE)
        pdf.multi_cell(0, 6, _safe(advisory))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(final_state: dict, report_dir, charts_dir=None, divergence_data=None) -> str:
    """
    Generates the weekly portfolio PDF report.

    Args:
        final_state:  Complete state dict from pipeline_graph.invoke()
        user_path:    Path to user's folder
        charts_dir:   Path to directory containing chart PNGs from generate_all_charts()
                      If None, charts sections are skipped

    Returns:
        Path to generated PDF as string
    """
    charts_dir = Path(charts_dir) if charts_dir else None
    report_dir.mkdir(parents=True, exist_ok=True)

    run_date    = final_state.get("run_date", str(date.today()))

    print(f"\n  [Report] Building PDF report for {final_state.get('user_name')}...")

    pdf = PortfolioReport()

    _section_cover(pdf, final_state)
    print(f"    [ok] Cover page")

    _section_portfolio_snapshot(pdf, final_state)
    print(f"    [ok] Portfolio snapshot")
    
    _section_recommendation_table(pdf, final_state)
    print(f"    [ok] Recommendation table")
    
    _section_simulation_charts(pdf, final_state, charts_dir)
    print(f"    [ok] Simulation charts")

    _section_advisory_commentary(pdf, final_state)
    print(f"    [ok] Commentary")

    _section_model_outputs(pdf, final_state, charts_dir)
    print(f"    [ok] Model outputs")
    
    _section_virtual_portfolio(pdf, divergence_data)
    print(f"    [ok] Virtual portfolio")

    pdf_path = report_dir / f"report_{final_state["user_name"]}_{run_date}.pdf"
    pdf.output(str(pdf_path))
    print(f"  [Report] Saved to {pdf_path}")

    return str(pdf_path)