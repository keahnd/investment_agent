"""
Agent 1 — Data Harvester
=========================
Scrapes financial news, sentiment surveys, and podcast transcripts.
Produces per-ticker text summaries for Agent 4's Black-Litterman
view generation.

Sources:
  - Finviz          : per-ticker news headlines
  - Seeking Alpha   : per-ticker news headlines
  - Yahoo Finance	: per-ticker news articles
  - SEC EDGAR		: per-ticker 8-K filings
  - AAII            : weekly investor sentiment survey
  - CNN Fear & Greed: market fear/greed index
  - YouTube         : podcast transcripts via auto-captions
  - yfinance        : earnings data and calendar

All raw text is written to users/{name}/data/raw/ as dated files.
Summaries are written to users/{name}/data/summaries/.
"""

import os
import json
import time
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

from agents.state import PipelineState
from agents.llm import get_llm


# ── Constants ────────────────────────────────────────────────────────────────

# Realistic browser headers — without these Finviz returns 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Seconds to wait between requests to the same domain
# Finviz will block you without this
FINVIZ_DELAY    = 2.5
SEEKALPHA_DELAY = 3.5
GENERAL_DELAY   = 1.0


# ═════════════════════════════════════════════════════════════════════════════
# SCRAPER FUNCTIONS — each returns a list of dicts or a single dict
# Each is independently try/excepted so one failure doesn't kill the others
# ═════════════════════════════════════════════════════════════════════════════

def scrape_finviz(ticker: str) -> list[dict]:
    """
    Scrapes the news table from Finviz for a given ticker.
    Returns a list of {date, time, source, headline} dicts.
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"    [warn] Finviz request failed for {ticker}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    news_table = soup.find(id="news-table")

    if not news_table:
        print(f"    [warn] Finviz news table not found for {ticker}")
        return []

    results = []
    current_date = None

    for row in news_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Finviz date cell format:
        #   "Apr-21-25 10:32AM"  — first headline of the day (date + time)
        #   "10:32AM"            — subsequent headlines (time only)
        date_cell = cells[0].text.strip()
        parts = date_cell.split()

        if len(parts) == 2:
            current_date = parts[0]   # e.g. "Apr-21-25"
            time_str     = parts[1]
        elif len(parts) == 1:
            time_str = parts[0]       # time only, date carries forward
        else:
            continue

        headline_tag = cells[1].find("a")
        url = headline_tag["href"]
        source_tag   = cells[1].find("span")

        if not headline_tag:
            continue

        results.append({
            "date":     current_date or "",
            "time":     time_str,
            "source":   source_tag.text.strip() if source_tag else "",
            "headline": headline_tag.text.strip(),
            "url":		url,
            "full_text":fetch_article_text(url)
        })

    return results


def scrape_seeking_alpha(ticker: str) -> list[dict]:
    """
    Fetches recent headlines from Seeking Alpha's internal API endpoint.
    Falls back to empty list on any failure — SA is more aggressive about
    bot detection than Finviz.
    """
    url = f"https://seekingalpha.com/api/v3/symbols/{ticker}/news"
    params = {"filter[until]": "", "filter[since]": "", "isMounting": "true"}

    try:
        response = requests.get(
            url, headers=HEADERS, params=params, timeout=10
        )
        data = response.json()
    except Exception as e:
        print(f"    [warn] Seeking Alpha fetch failed for {ticker}: {e}")
        return []

    results = []
    for article in data.get("data", [])[:20]:
        attrs = article.get("attributes", {})

        # Paywalled articles: store title only, no full text available
        results.append({
            "title":       attrs.get("title", ""),
            "published":   attrs.get("publishOn", ""),
            "paywalled":   attrs.get("isPaywalled", True),
        })

    return results


def fetch_aaii_sentiment() -> dict:
    """
    Scrapes the AAII weekly sentiment survey table.
    Returns {bullish, bearish, neutral, bull_bear_spread} as strings.
    Published every Thursday at aaii.com.
    """
    url = "https://www.aaii.com/sentimentsurvey/sent_results"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", {"id": "sentimentResultsTable"})
        if not table:
            # Table ID may have changed — try finding any table with
            # Bullish/Bearish headers as a fallback
            for t in soup.find_all("table"):
                if "Bullish" in t.text and "Bearish" in t.text:
                    table = t
                    break

        if not table:
            return {"error": "AAII table not found"}

        rows = table.find_all("tr")
        headers = [th.text.strip() for th in rows[0].find_all(["th", "td"])]
        values  = [td.text.strip() for td in rows[1].find_all("td")]

        row = dict(zip(headers, values))

        return {
            "date":             row.get("Date", ""),
            "bullish":          row.get("Bullish", ""),
            "neutral":          row.get("Neutral", ""),
            "bearish":          row.get("Bearish", ""),
            "bull_bear_spread": row.get("Bull-Bear Spread", ""),
        }

    except Exception as e:
        print(f"    [warn] AAII sentiment fetch failed: {e}")
        return {"error": str(e)}


def fetch_fear_greed() -> dict:
    """
    Fetches the CNN Fear & Greed Index via their public data endpoint.
    Returns {score, rating} where score is 0-100.
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        data = response.json()
        current = data.get("fear_and_greed", {})

        return {
            "score":  current.get("score"),
            "rating": current.get("rating"),   # e.g. "Fear", "Extreme Greed"
        }

    except Exception as e:
        print(f"    [warn] Fear & Greed fetch failed: {e}")
        return {"score": None, "rating": "unknown", "error": str(e)}


def _resolve_channel_video_urls(channel_url: str, n: int = 3) -> list[str]:
    """
    Given a YouTube channel URL (/@Handle or /channel/ID), fetches the channel's
    /videos page and returns the URLs of the most recent n uploads.
    Returns empty list on failure.
    """
    videos_url = channel_url.rstrip("/") + "/videos"
    try:
        response = requests.get(videos_url, headers=HEADERS, timeout=10)
        video_ids = list(dict.fromkeys(re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', response.text)))
        return [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids[:n]]
    except Exception as e:
        print(f"    [warn] Could not resolve videos from {channel_url}: {e}")
    return []


def fetch_youtube_transcript(url: str, name: str) -> str:
    """
    Fetches auto-captions for a YouTube video URL (watch?v=... or youtu.be/...).
    Returns the full transcript as a single string, or empty string on failure.
    """
    # Extract video ID — handles both youtube.com/watch?v= and youtu.be/ formats
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        print(f"    [warn] Could not extract video ID from: {url}")
        return ""

    video_id = match.group(1)

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        full_text = " ".join(segment.text for segment in transcript)
        print(f"    [ok]   Transcript fetched for {name} ({len(full_text):,} chars)")
        return full_text

    except Exception as e:
        print(f"    [warn] Transcript fetch failed for {name}: {e}")
        return ""
    

def scrape_yahoo_finance(ticker: str) -> list[dict]:
    """
    Fetches full article text from Yahoo Finance news for a ticker.
    Yahoo Finance is free and aggregates from Reuters, AP, and others.
    """
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "xml")
        items = soup.find_all("item")
        
        results = []
        for item in items[:10]:   # cap at 10 most recent
            article_url = item.find("link")
            title = item.find("title")
            pub_date = item.find("pubDate")
            description = item.find("description")
            
            results.append({
                "title":       title.text if title else "",
                "url":         article_url.text if article_url else "",
                "published":   pub_date.text if pub_date else "",
                # description is often a snippet — better than nothing
                # if full article fetch fails
                "snippet":     description.text if description else "",
            })
        
        # Now fetch full article text for each
        for result in results:
            if result["url"]:
                full_text = fetch_article_text(result["url"])
                result["full_text"] = full_text
                time.sleep(GENERAL_DELAY)
        
        return results
        
    except Exception as e:
        print(f"    [warn] Yahoo Finance fetch failed for {ticker}: {e}")
        return []


def fetch_article_text(url: str) -> str:
    """
    Fetches the full text of a news article from its URL.
    Extracts paragraph text, ignoring navigation and boilerplate.
    Returns empty string on failure or if paywalled.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script, style, and nav elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        
        # Extract all paragraph text
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
        
        # Basic paywall detection — if the article is very short
        # after extraction, it's probably gated
        if len(text) < 200:
            return ""
        
        return text[:5000]   # cap at 5000 chars per article
        
    except Exception as e:
        return ""


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
            # Most recent 4 quarters
            recent = earnings_hist.head(12).reset_index()
            
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
            surprises = [q["surprise_pct"] for q in quarters if q["surprise_pct"] is not None]
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


def extract_ticker_mentions(transcript: str, ticker: str, company_name: int, window: int = 600) -> str:
    """
    Extracts text windows around each mention of a ticker or its company name.
    Searches case-insensitively. Returns a concatenated string of all relevant passages.
    A window of 600 characters captures roughly 2-3 sentences of context.
    """
    lower = transcript.lower()
    terms = [ticker.lower()]
    if company_name:
        # Also search for the first word of the company name (e.g. "Amazon" from "Amazon.com Inc.")
        first_word = company_name.split()[0].lower().rstrip(".,")
        if len(first_word) > 3:   # skip short words like "the", "inc"
            terms.append(first_word)
        terms.append(company_name.lower())

    seen_ranges: list[tuple[int, int]] = []
    mentions = []

    for term in terms:
        start = 0
        while True:
            idx = lower.find(term, start)
            if idx == -1:
                break
            begin = max(0, idx - window // 2)
            end   = min(len(transcript), idx + window // 2)
            # Deduplicate overlapping windows
            if not any(b <= idx < e for b, e in seen_ranges):
                mentions.append(transcript[begin:end])
                seen_ranges.append((begin, end))
            start = idx + 1

    return "\n...\n".join(mentions)


# ═════════════════════════════════════════════════════════════════════════════
# FILE I/O HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def write_raw(raw_dir: Path, run_date: str, label: str, source: str, text: str):
    """Writes raw scraped text to a dated file. Never overwrites."""
    filename = f"{run_date}_{label}_{source}.txt"
    filepath = raw_dir / filename
    filepath.write_text(text, encoding="utf-8")


def write_summary(summary_dir: Path, run_date: str, ticker: str, text: str):
    """Writes a summary paragraph to a dated file."""
    filename = f"{run_date}_{ticker}_summary.txt"
    filepath = summary_dir / filename
    filepath.write_text(text, encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# LLM SUMMARISATION
# ═════════════════════════════════════════════════════════════════════════════

SUMMARISE_PROMPT = """You are a financial analyst summarising market intelligence for {ticker}.

Below is collected text from multiple sources gathered this week.
Sources are labelled by type — weight them accordingly:
- Yahoo Finance articles: medium-high weight — aggregated from credible outlets
- Finviz articles: medium-high weight — aggregated from credible outlets
- Podcast commentary: medium weight if host is credentialed, lower otherwise
- Seeking Alpha headlines: low weight - useful for topic detection, limited depth
- Earnings data: high weight - useful to determine if company is generating consistent earnings

Summarise in 4-6 sentences covering:
- Overall sentiment direction and conviction level
- Specific catalysts, guidance changes, or material events mentioned
- Any analyst price targets or return expectations cited
- Risks or concerns raised by multiple sources
- Whether valuation or fundamentals are discussed explicitly

Collected text:
{text}

Write as plain prose. No bullet points. No preamble. Be specific — 
name the catalysts, name the figures cited, name the risks."""


def summarise_ticker_text(ticker: str, text: str) -> str:
    """
    Calls the LLM to produce a 3-5 sentence summary of all collected
    text for a ticker. Returns a fallback string if text is empty or
    the LLM call fails.
    """
    if not text.strip():
        return f"No data collected for {ticker} this week."

    # Truncate to ~8000 characters to stay within context limits for
    # smaller models. 8000 chars is roughly 2000 tokens — well within
    # gpt-4o-mini and claude-haiku limits.
    truncated = text[:8000]
    if len(text) > 8000:
        truncated += "\n[text truncated]"

    prompt = SUMMARISE_PROMPT.format(ticker=ticker, text=truncated)

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"    [warn] LLM summarisation failed for {ticker}: {e}")
        return f"Summarisation unavailable for {ticker} this week. Raw data collected."


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def agent1_harvester(state: PipelineState) -> dict:
	"""
	Agent 1 — Data Harvester

	Runs all scrapers, stores raw text to disk, produces per-ticker
	summaries via LLM, fetches macro sentiment indicators and earnings dates.

	Returns state updates for: raw_text, summaries, aaii_sentiment,
	fear_greed, earnings_dates, errors.
	"""
	user_name = state["user_name"]
	user_path = Path(state["user_path"])
	tickers   = state["tickers"]
	run_date  = state["run_date"]
	errors    = list(state.get("errors") or [])

	print(f"\n  [Agent 1] Data harvester running for {user_name}")
	print(f"            Tickers : {tickers}")

	# Create output directories
	raw_dir     = user_path / "data" / "raw"
	summary_dir = user_path / "data" / "summaries"
	raw_dir.mkdir(parents=True, exist_ok=True)
	summary_dir.mkdir(parents=True, exist_ok=True)

	# Load config for podcast sources
	config_path = user_path / "config.json"
	config = {}
	if config_path.exists():
		with open(config_path) as f:
			config = json.load(f)

	podcast_sources = config.get("podcast_sources", [])

	# ── Per-ticker scraping ───────────────────────────────────────────────────
	ticker_text = {t: [] for t in tickers}   # accumulates all text per ticker

	for ticker in tickers:
		print(f"\n    Scraping {ticker}...")

		# Finviz
		finviz_results = scrape_finviz(ticker)
		if finviz_results:
			finviz_text = "\n".join(
				f"{r['date']} {r['time']} [{r['source']}] {r['headline']}"
				for r in finviz_results
			)
			write_raw(raw_dir, run_date, ticker, "finviz", finviz_text)
			ticker_text[ticker].append(f"=== Finviz Headlines ===\n{finviz_text}")
			print(f"    [ok]   Finviz: {len(finviz_results)} headlines")
		else:
			errors.append(f"{ticker}: Finviz returned no results")

		time.sleep(FINVIZ_DELAY)

		# Seeking Alpha
		sa_results = scrape_seeking_alpha(ticker)
		if sa_results:
			sa_text = "\n".join(
				f"{r['published']} {r['title']}"
				+ (" [paywalled]" if r["paywalled"] else "")
				for r in sa_results
			)
			write_raw(raw_dir, run_date, ticker, "seekingalpha", sa_text)
			ticker_text[ticker].append(f"=== Seeking Alpha Headlines ===\n{sa_text}")
			print(f"    [ok]   Seeking Alpha: {len(sa_results)} headlines")
		else:
			errors.append(f"{ticker}: Seeking Alpha returned no results")

		time.sleep(SEEKALPHA_DELAY)
		
		# Yahoo Finance
		yf_results = scrape_yahoo_finance(ticker)
		if yf_results:
			yf_text = "\n".join(
				f"{r['published']} {r['title']} [{r['snipped']}] {r['full_text']}"
				for r in yf_results
			)
			write_raw(raw_dir, run_date, ticker, "yahoofinance", yf_text)
			ticker_text[ticker].append(f"=== Yahoo Finance Articles ===\n{yf_text}")
			print(f"	[ok]	Yahoo Finance: {len(yf_text)} articles")
		else:
			errors.append(f"{ticker}: Yahoo Finance returned no results")
   
		# SEC Filings
		# sec_results = fetch_sec_filings(ticker)
		# if sec_results:
		# 	yf_text = "\n".join(
		# 		f"{r['published']} {r['title']} [{r['snipped']}] {r['full_text']}"
		# 		for r in yf_results
		# 	)
		# 	write_raw(raw_dir, run_date, ticker, "yahoofinance", yf_text)
		# 	ticker_text[ticker].append(f"=== Yahoo Finance Articles ===\n{yf_text}")
		# 	print(f"	[ok]	Yahoo Finance: {len(yf_text)} articles")
		# else:
		# 	errors.append(f"{ticker}: Yahoo Finance returned no results")
  
		# Earnings Data
		print(f"\n    Fetching earnings data...")
		earnings = fetch_earnings_data(ticker)
		if earnings:
			earnings_text = "\n".join(
				f"{r['recent_quarters']} {r['avg_eps_surprise_pct']} [{r['consecutive_beats']}]"
				for r in earnings
			)
			write_raw(raw_dir, run_date, ticker, "earnings", earnings_text)
			ticker_text[ticker].append(f"=== Earnings Data ===\n{earnings_text}")
			print(f"    [ok]   {ticker} next earnings: {earnings["recent_quarters"][0]["date"] or 'unknown'}")
		else:
			errors.append(f"{ticker}: Earnings data returned no results")

	# ── Macro signals (portfolio-level, not per-ticker) ───────────────────────
	print(f"\n    Fetching macro signals...")

	aaii = fetch_aaii_sentiment()
	write_raw(raw_dir, run_date, "MARKET", "aaii", json.dumps(aaii, indent=2))
	if "error" not in aaii:
		print(f"    [ok]   AAII: bullish={aaii.get('bullish')} bearish={aaii.get('bearish')}")
	else:
		errors.append(f"AAII fetch failed: {aaii.get('error')}")

	fg = fetch_fear_greed()
	write_raw(raw_dir, run_date, "MARKET", "fear_greed", json.dumps(fg, indent=2))
	if "error" not in fg:
		print(f"    [ok]   Fear & Greed: {fg.get('score')} ({fg.get('rating')})")
	else:
		errors.append(f"Fear & Greed fetch failed: {fg.get('error')}")

	# ── Podcast transcripts ───────────────────────────────────────────────────
	# Build ticker → company name map from portfolio CSV
	company_names: dict[str, str] = {}
	portfolio_path = user_path / "portfolio.csv"
	if portfolio_path.exists():
		port_df = pd.read_csv(portfolio_path)
		company_names = dict(zip(port_df["Symbol"], port_df["Name"]))

	if podcast_sources:
		print(f"\n    Fetching {len(podcast_sources)} podcast transcript(s)...")

	for podcast in podcast_sources:
		url = podcast["url"]
		if "/@" in url or "/channel/" in url:
			video_urls = _resolve_channel_video_urls(url, n=1)
		else:
			video_urls = [url]

		for video_url in video_urls:
			transcript = fetch_youtube_transcript(video_url, podcast["name"])

		if transcript:
			# Find mentions of each ticker and append relevant passages
			for ticker in tickers:
				# Check for ticker symbol and company name mentions
				mentions = extract_ticker_mentions(transcript, ticker, company_name=company_names.get(ticker, ""))
				if mentions:
					# Store relevant information to disk
					write_raw(raw_dir, run_date, "PODCAST", podcast["name"], mentions)
					label = f"=== Podcast: {podcast['name']} (credibility: {podcast.get('credibility', 'medium')}) ==="
					ticker_text[ticker].append(f"{label}\n{mentions}")
					print(f"    [ok]   Found {ticker} mentions in {podcast['name']}")

		time.sleep(GENERAL_DELAY)

	# ── LLM summarisation ─────────────────────────────────────────────────────
	print(f"\n    Summarising collected text via LLM...")
	summaries = {}
	raw_text_out = {}

	for ticker in tickers:
		combined = "\n\n".join(ticker_text[ticker])
		raw_text_out[ticker] = combined

		summary = summarise_ticker_text(ticker, combined)
		summaries[ticker] = summary
		write_summary(summary_dir, run_date, ticker, summary)
		print(f"    [ok]   {ticker}: summary written ({len(summary)} chars)")

	print(f"\n  [Agent 1] Complete. Errors: {len(errors)}")

	return {
		"raw_text":       raw_text_out,
		"summaries":      summaries,
		"aaii_sentiment": aaii,
		"fear_greed":     fg,
		"earnings_dates": earnings,
		"errors":         errors,
	}