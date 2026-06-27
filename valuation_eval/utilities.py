# ═════════════════════════════════════════════════════════════════════════════
# SECTOR ETF MAP
# Each sector maps to a SPDR Select ETF whose trailing PE yfinance exposes.
# ═════════════════════════════════════════════════════════════════════════════
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology":              "XLK",
    "Healthcare":              "XLV",
    "Financials":              "XLF",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Energy":                  "XLE",
    "Industrials":             "XLI",
    "Basic Materials":         "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
    "Communication Services":  "XLC",
}


# ── Source C: Peer sampling ───────────────────────────────────────────────

# Top 20 S&P 500 components by market cap, organised by yfinance sector name.
# Updated periodically — constituent changes are infrequent.
SECTOR_PEERS: dict[str, list[str]] = {
	"Technology": [
		"MSFT", "AAPL", "NVDA", "AVGO", "ORCL",
		"CRM", "AMD", "QCOM", "TXN", "AMAT",
		"INTC", "MU", "KLAC", "LRCX", "ADI",
		"PANW", "SNPS", "CDNS", "MRVL", "FTNT",
	],
	"Healthcare": [
		"LLY", "JNJ", "UNH", "ABBV", "MRK",
		"TMO", "ABT", "DHR", "BMY", "AMGN",
		"PFE", "SYK", "MDT", "ISRG", "GILD",
		"VRTX", "REGN", "CI", "HCA", "MCK",
	],
	"Financials": [
		"BRK-B", "JPM", "V", "MA", "BAC",
		"WFC", "GS", "MS", "AXP", "BLK",
		"SCHW", "C", "CB", "PGR", "MMC",
		"ICE", "CME", "AON", "TRV", "AFL",
	],
	"Consumer Discretionary": [
		"AMZN", "TSLA", "HD", "MCD", "NKE",
		"LOW", "SBUX", "TJX", "BKNG", "ORLY",
		"AZO", "GM", "F", "MAR", "HLT",
		"YUM", "DHI", "LEN", "PHM", "CCL",
	],
	"Consumer Staples": [
		"WMT", "PG", "COST", "KO", "PEP",
		"PM", "MO", "MDLZ", "CL", "KMB",
		"STZ", "GIS", "K", "HSY", "TSN",
		"SJM", "CAG", "CHD", "CLX", "EL",
	],
	"Energy": [
		"XOM", "CVX", "COP", "EOG", "SLB",
		"MPC", "PSX", "VLO", "PXD", "OXY",
		"KMI", "WMB", "HAL", "DVN", "BKR",
		"FANG", "HES", "APA", "MRO", "OKE",
	],
	"Industrials": [
		"GE", "CAT", "UPS", "HON", "DE",
		"RTX", "LMT", "NOC", "GD", "BA",
		"MMM", "EMR", "ETN", "ITW", "PH",
		"FDX", "CSX", "NSC", "UNP", "CARR",
	],
	"Basic Materials": [
		"LIN", "APD", "SHW", "ECL", "NEM",
		"FCX", "NUE", "VMC", "MLM", "CF",
		"MOS", "IFF", "PPG", "ALB", "RPM",
	],
	"Real Estate": [
		"PLD", "AMT", "EQIX", "CCI", "PSA",
		"O", "WELL", "DLR", "EXR", "AVB",
		"EQR", "VTR", "ARE", "BXP", "KIM",
	],
	"Utilities": [
		"NEE", "DUK", "SO", "D", "AEP",
		"EXC", "XEL", "PCG", "SRE", "ED",
		"ES", "WEC", "DTE", "ETR", "FE",
	],
	"Communication Services": [
		"META", "GOOGL", "GOOG", "NFLX", "DIS",
		"CMCSA", "T", "VZ", "TMUS", "EA",
		"ATVI", "TTWO", "WBD", "PARA", "FOXA",
	],
}

# Sector median PEG values — analogous to Damodaran PE averages.
# These reflect typical PEG ranges per sector based on growth profiles.
# High-margin, slow-growth sectors (staples, utilities) carry higher PEGs.
# High-growth sectors (tech) carry lower PEGs because growth is less durable.
SECTOR_PEG_LONGRUN: dict[str, dict] = {
    "Technology":              {"median": 1.80, "p25": 1.20, "p75": 2.50},
    "Healthcare":              {"median": 1.70, "p25": 1.10, "p75": 2.30},
    "Financials":              {"median": 1.20, "p25": 0.80, "p75": 1.70},
    "Consumer Discretionary":  {"median": 1.50, "p25": 1.00, "p75": 2.20},
    "Consumer Staples":        {"median": 2.20, "p25": 1.60, "p75": 3.00},
    "Energy":                  {"median": 0.90, "p25": 0.60, "p75": 1.40},
    "Industrials":             {"median": 1.60, "p25": 1.10, "p75": 2.20},
    "Basic Materials":         {"median": 1.20, "p25": 0.80, "p75": 1.70},
    "Real Estate":             {"median": 2.50, "p25": 1.80, "p75": 3.50},
    "Utilities":               {"median": 2.80, "p25": 2.00, "p75": 3.80},
    "Communication Services":  {"median": 1.50, "p25": 1.00, "p75": 2.10},
}

# Sectors where EV/EBITDA is structurally unreliable.
# For these, the signal is skipped rather than approximated.
EV_EBITDA_UNRELIABLE_SECTORS = {
    "Energy",         # EBITDA swings with commodity price, not business quality
    "Financials",     # interest income/expense makes EBITDA meaningless for banks
    "Real Estate",    # valued on cap rates and FFO, not EBITDA
    "Basic Materials" # commodity cycle dominates over operating performance
}

CACHE_CONFIG = {
    "peer_pe": {
        "ttl_days":          30,     # monthly refresh baseline
        "market_move_pct":    8.0,   # invalidate if SPY moved > 8% since last fetch
        "description":       "Forward/trailing PE from peer sampling"
    },
    "peer_ev_ebitda": {
        "ttl_days":          90,     # quarterly
        "market_move_pct":   12.0,   # less sensitive to market moves than PE
        "description":       "EV/EBITDA from peer sampling"
    },
    "peer_peg": {
        "ttl_days":          90,     # quarterly
        "market_move_pct":   None,   # PEG barely responds to market moves
        "description":       "PEG from peer sampling"
    },
    "damodaran": {
        "ttl_days":         365,     # annual — he only publishes once a year
        "market_move_pct":   None,   # long-run averages, market moves irrelevant
        "description":       "Damodaran industry-level PE table"
    },
}