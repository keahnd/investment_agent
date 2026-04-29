from agents.state import PipelineState




def fetch_sec_filings(ticker: str, filing_types: list = ["8-K", "10-Q"]) -> list[dict]:
    """
    Fetches recent SEC filings for a ticker via the EDGAR full-text search API.
    8-K contains material events, earnings releases, and guidance.
    10-Q contains quarterly financial statements.
    Both are primary source data — highest credibility.
    """
    results = []
    
    # First get the CIK number for this ticker
    cik_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={_ninety_days_ago()}&forms=8-K"
    
    try:
        # EDGAR search API
        search_url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": _ninety_days_ago(),
            "forms": ",".join(filing_types),
        }
        
        response = requests.get(
            search_url,
            params=params,
            headers={**HEADERS, "User-Agent": "Keahn Divecha keahnd@gmail.com"},
            timeout=10
        )
        data = response.json()
        
        for hit in data.get("hits", {}).get("hits", [])[:5]:
            source = hit.get("_source", {})
            results.append({
                "form_type":   source.get("form_type", ""),
                "filed_date":  source.get("file_date", ""),
                "company":     source.get("entity_name", ""),
                "description": source.get("period_of_report", ""),
                "url":         f"https://www.sec.gov/Archives/edgar/data/{source.get('entity_id', '')}/{source.get('file_num', '')}",
            })
        
    except Exception as e:
        print(f"    [warn] SEC EDGAR fetch failed for {ticker}: {e}")
    
    return results


def _ninety_days_ago() -> str:
    """Returns a date string 90 days ago in YYYY-MM-DD format."""
    return (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")


def fetch_sec_report_data():
    """
    Fetches the information from the SEC fillings.
    """
    print(f"Not Implemented yet.")


def fetch_earnings_data(ticker: str) -> dict:
    """
    Fetches earnings history and upcoming estimates via yfinance.
    Returns actual vs estimated EPS, surprise percentage, and
    next quarter estimates.
    
    This is separate from valuation metrics (handled by Agent 2) —
    this is specifically for beat/miss history and forward guidance context.
    """
    try:        
        # Earnings history — actual vs estimated EPS per quarter
        earnings_hist = yf.Ticker(ticker).earnings_dates
        
        if earnings_hist is not None and not earnings_hist.empty:
            # Most recent 12 quarters
            recent = earnings_hist.head(13).reset_index()
            
            quarters = []
            for _, row in recent.iterrows():
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
            surprises = [q["surprise_pct"] for q in quarters[1:] if q["surprise_pct"] is pd.notna]
            avg_surprise = round(sum(surprises) / len(surprises), 2) if surprises else None
            
            return {
                "recent_quarters": quarters,
                "avg_eps_surprise_pct": avg_surprise,
                "consecutive_beats": _count_consecutive_beats(quarters),
            }
    
    except Exception as e:
        print(f"    [warn] Earnings data fetch failed for {ticker}: {e}")
    
    return {"recent_quarters": [], "avg_eps_surprise_pct": None, "consecutive_beats": None}


def _count_consecutive_beats(quarters: list) -> int:
    """
    Counts how many consecutive quarters a company has beaten
    EPS estimates, starting from the most recent.
    A consistent beat pattern is a mild positive signal.
    """
    count = 0
    for q in quarters:
        if q["surprise_pct"] is not None and q["surprise_pct"] > 0:
            count += 1
        else:
            break
    return count

def fetch_earnings_dates(tickers: list[str]) -> dict:
    earnings = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).calendar
            if info is not None and not info.empty:
                # calendar returns a DataFrame with earnings dates
                earnings[ticker] = {
                    "next_earnings": str(info.columns[0].date()) if len(info.columns) > 0 else None
                }
        except Exception:
            earnings[ticker] = {"next_earnings": None}
        time.sleep(0.5)
    return earnings


def agent2_quant(state: PipelineState) -> dict:
    """
    Agent 2 — Quantitative Modeller
    Runs factor regression, GARCH, and valuation metric fetching per ticker.
    Wraps existing Phase 1 modelling code as LangChain tools.
    """
    print(f"  [Agent 2] Quant running")
    
    # SEC Filings
    # sec_results = fetch_sec_filings(ticker)
    # if sec_results:
    # 	yf_text = "\n".join(
    # 		f"{r['published']} {r['title']} [{r['snipped']}] {r['full_text']}"
    # 		for r in yf_results
    # 	)
    # 	write_raw(raw_dir, ticker, "yahoofinance", yf_text)
    # 	ticker_text[ticker].append(f"=== Yahoo Finance Articles ===\n{yf_text}")
    # 	print(f"	[ok]	Yahoo Finance: {len(yf_text)} articles")
    # else:
    # 	errors.append(f"{ticker}: Yahoo Finance returned no results")
    
    # --- Earnings calendar ---
    # earnings = fetch_earnings_dates(tickers)

    return {
        "factor_results": {
            t: {"alpha": 0.0, "b_mkt": 1.0, "b_smb": 0.0, "b_hml": 0.0, "r2": 0.5}
            for t in state["tickers"]
        },
        "garch_results": {
            t: {"omega": 1e-6, "alpha": 0.08, "beta": 0.90, "persist": 0.98, "sigma": 0.20}
            for t in state["tickers"]
        },
        "valuation": {
            t: {
                "peg": None, "fwd_pe": None, "ttm_pe": None, "ev_ebitda": None,
                "target_price": None, "analyst_rec": None,
                "200MA": None, "50MA": None,
                "earnings_growth": None, "revenue_growth": None,
                "sector": None, "industry": None,
            }
            for t in state["tickers"]
        },
        "quant_commentary": "[STUB] Quantitative analysis not yet implemented.",
    }