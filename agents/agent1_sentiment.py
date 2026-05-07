"""
Agent 1 — Sentiment Analysis
=========================
Scrapes financial news, sentiment surveys, and podcast transcripts.
Produces per-ticker text summaries for Agent 3's Black-Litterman
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

    Args:
        ticker: Plain ticker symbol (e.g. 'AAPL', 'LMT').

    Returns:
        List of dicts with keys: date, time, source, headline, url, full_text.
        Returns empty list on any request or parse failure.
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

    for row in news_table.find_all("tr"): # limit to lower number of articles??
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
            "url":      url,
            "full_text": "",
        })

    return results


def scrape_seeking_alpha(ticker: str) -> list[dict]:
    """
    Fetches recent headlines from Seeking Alpha's internal API endpoint.

    SA's /api/v3/ requires authentication cookies. Without them the data
    array is empty. Falls back to empty list on any failure.

    Args:
        ticker: Plain ticker symbol (e.g. 'AAPL').

    Returns:
        List of dicts with keys: title, published, paywalled.
        Returns empty list if unauthenticated or on request failure.
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
    Scrapes the AAII weekly sentiment survey table from aaii.com.

    Published every Thursday. Parses bullish/neutral/bearish percentages
    and the bull-bear spread from the results table.

    Returns:
        Dict with keys: date, bullish, neutral, bearish, bull_bear_spread.
        Returns {'error': <message>} on failure.
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
    Fetches the CNN Fear & Greed Index from their public data endpoint.

    Returns:
        Dict with keys: score (0–100 float), rating (str, e.g. 'Fear').
        Includes 'error' key on failure.
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
    Resolves the most recent video URLs from a YouTube channel page.

    Fetches the channel's /videos page and extracts video IDs from the
    embedded JSON, deduplicating in order of appearance.

    Args:
        channel_url: YouTube channel URL in /@Handle or /channel/ID format.
        n: Number of most recent videos to return. Defaults to 3.

    Returns:
        List of YouTube watch URLs (up to n). Returns empty list on failure.
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
    Fetches the auto-generated captions for a YouTube video.

    Handles both youtube.com/watch?v= and youtu.be/ URL formats.

    Args:
        url: Full YouTube video URL.
        name: Human-readable label for the video or podcast, used in log output.

    Returns:
        Full transcript as a single whitespace-joined string. Returns empty
        string if the video ID cannot be extracted or on any fetch failure.
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
    Fetches recent news items from the Yahoo Finance RSS headline feed.

    Yahoo Finance aggregates from Reuters, AP, and others and is free to use.
    Requires ticker.TO format for TSX-listed securities.

    Args:
        ticker: Ticker symbol as used by Yahoo Finance (e.g. 'AAPL', 'TD.TO').

    Returns:
        List of dicts with keys: title, url, published, snippet, full_text.
        Capped at 10 most recent items. Returns empty list on failure.
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
                "title":       	title.text if title else "",
                "url":         	article_url.text if article_url else "",
                "published":   	pub_date.text if pub_date else "",
                # description is often a snippet — better than nothing
                # if full article fetch fails
                "snippet":     	description.text if description else "",
                "full_text":	""
            })

        return results
        
    except Exception as e:
        print(f"    [warn] Yahoo Finance fetch failed for {ticker}: {e}")
        return []


def scrape_etf_dot_com(ticker: str) -> list[dict]:
    """
    Scrapes the ETF overview page on etf.com for fund summary and news links.

    Extracts a fund description block (if present) as a pseudo-article, then
    scrapes any news section anchor tags for additional items.

    Args:
        ticker: Plain ETF ticker symbol (e.g. 'ITA', 'NLR').

    Returns:
        List of dicts with keys: title, url, full_text.
        Returns empty list on request failure.
    """
    url = f"https://www.etf.com/{ticker}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"    [warn] etf.com request failed for {ticker}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    overview = soup.find("div", class_=re.compile(r"fund-description|etf-description|overview", re.I))
    if overview:
        text = overview.get_text(separator=" ", strip=True)
        if len(text) > 100:
            results.append({"title": f"{ticker} ETF Overview", "url": url, "full_text": text[:5000]})

    news_section = soup.find(["section", "div"], id=re.compile(r"news", re.I))
    if not news_section:
        news_section = soup.find(["section", "div"], class_=re.compile(r"news", re.I))
    if news_section:
        for a in news_section.find_all("a", href=True)[:10]:
            headline = a.get_text(strip=True)
            href = a["href"]
            if not href.startswith("http"):
                href = "https://www.etf.com" + href
            if headline:
                results.append({"title": headline, "url": href, "full_text": ""})

    return results


def scrape_globe_and_mail(ticker: str) -> list[dict]:
    """
    Scrapes recent news from the Globe and Mail stock page for a TSX ticker.

    Constructs the URL using Globe and Mail's exchange suffix convention
    (ticker + '-T', e.g. TD → TD-T, BEP-UN → BEP-UN-T). Falls back to
    scanning all article-like anchor tags if no <article> elements are found.

    Args:
        ticker: Plain TSX ticker symbol without exchange suffix (e.g. 'TD').

    Returns:
        List of dicts with keys: title, url, published, full_text.
        Returns empty list on request failure.
    """
    gm_ticker = ticker + "-T"
    url = f"https://www.theglobeandmail.com/investing/markets/stocks/{gm_ticker}/"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"    [warn] Globe and Mail request failed for {ticker}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for article in soup.find_all("article")[:10]:
        a_tag = article.find("a", href=True)
        if not a_tag:
            continue
        href = a_tag["href"]
        if not href.startswith("http"):
            href = "https://www.theglobeandmail.com" + href
        headline_tag = article.find(["h2", "h3", "h4"])
        time_tag = article.find("time")
        title = headline_tag.get_text(strip=True) if headline_tag else a_tag.get_text(strip=True)
        published = time_tag.get("datetime", "") if time_tag else ""
        results.append({"title": title, "url": href, "published": published, "full_text": ""})

    if not results:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/article/" in href or "/investing/" in href:
                title = a.get_text(strip=True)
                if len(title) > 20:
                    if not href.startswith("http"):
                        href = "https://www.theglobeandmail.com" + href
                    results.append({"title": title, "url": href, "published": "", "full_text": ""})
                    if len(results) >= 10:
                        break

    return results


def fetch_article_text(url: str) -> str:
    """
    Fetches and extracts paragraph text from a news article URL.

    Removes script, style, nav, header, footer, and aside tags before
    extracting paragraphs. Treats content under 200 characters as paywalled.

    Args:
        url: Full URL of the news article to fetch.

    Returns:
        Extracted article text capped at 5000 characters. Returns empty
        string on request failure or if content appears paywalled.
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


def filter_relevant_headlines(ticker: str, company_name: str, articles: list[dict], n: int = 5) -> list[dict]:
    """
    Uses an LLM to select the n most relevant articles for a given ticker.

    Passes headline text to the LLM and parses the returned comma-separated
    index list. Falls back to the first n articles if the LLM call fails.

    Args:
        ticker: Ticker symbol used as context for relevance scoring.
        company_name: Optional company name appended to the LLM prompt for context.
        articles: List of article dicts, each must have a 'headline' or 'title' key.
        n: Number of most relevant articles to return. Defaults to 5.

    Returns:
        Filtered list of up to n article dicts selected from the input list.
    """
    if not articles:
        return []
    name_hint = f" ({company_name})" if company_name else ""
    numbered = "\n".join(
        f"{i}. {a.get('headline') or a.get('title', '')}"
        for i, a in enumerate(articles)
    )
    prompt = (
        f"You are filtering financial news for relevance to {ticker}{name_hint}.\n"
        f"Return ONLY a comma-separated list of the indices (0-based) of the {n} most "
        f"relevant headlines below. No explanation.\n\n{numbered}"
    )
    try:
        llm = get_llm()
        response = llm.invoke(prompt).content.strip()
        indices = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]
        return [articles[i] for i in indices if i < len(articles)]
    except Exception as e:
        print(f"    [warn] Headline filter failed for {ticker}: {e}")
        return articles[:n]
    
    
def mention_is_relevant(ticker: str, mentions: list[str], company_name: str = "") -> list[str]:
    """
    Filters podcast transcript excerpts to remove false positives for a ticker.

    Some regex matches on ticker symbols or company names may not refer to the
    company (e.g. 'uber' as an adjective vs. Uber Inc.). Uses an LLM to verify
    each candidate excerpt.

    Args:
        ticker: Ticker symbol being searched for.
        mentions: List of transcript excerpt strings flagged as potential mentions.
        company_name: Optional company name for additional LLM context.

    Returns:
        Filtered list of excerpts genuinely discussing the ticker. Returns the
        full mentions list unchanged if the LLM call fails.
    """
    if not mentions:
        return []
    
    # print(f"Found {len(mentions)} mentions.")
    name_hint = f" ({company_name})" if company_name else ""
    numbered = "\n".join(f"{i}. {m}" for i, m in enumerate(mentions))
    prompt = (
        f"You are filtering podcast transcript excerpts for relevance to {ticker}{name_hint}.\n"
        f"Each excerpt below was flagged because it contains the ticker symbol or company name, "
        f"but some may be false positives (e.g. 'uber' used as an adjective, not Uber the company).\n"
        f"Return ONLY a comma-separated list of the 0-based indices of excerpts that are genuinely "
        f"discussing {ticker}{name_hint}. No explanation.\n\n"
        f"{numbered}"
    )
    try:
        llm = get_llm()
        response = llm.invoke(prompt).content.strip()
        indices = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]
        # print(f"Only {len(indices)} were relevant.")
        return [mentions[i] for i in indices if i < len(mentions)]
    except Exception as e:
        print(f"    [warn] Podcast relevance filter failed for {ticker}: {e}")
        return mentions
    

def extract_ticker_mentions(transcript: str, ticker: str, company_name: str = "", window: int = 600) -> str:
    """
    Extracts text windows around whole-word mentions of a ticker or company name.

    Uses regex word boundaries to avoid false positives (e.g. 'ITA' inside
    'capital'). Deduplicates overlapping windows and verifies results via
    mention_is_relevant.

    Args:
        transcript: Full transcript text to search.
        ticker: Ticker symbol to search for.
        company_name: Optional company name; the first word (>3 chars) and full
            name are also searched as additional terms.
        window: Characters of context to include around each mention. Defaults to 600.

    Returns:
        Relevant excerpts joined by '\\n...\\n'. Returns empty string if no
        relevant mentions are found.
    """
    terms = [re.escape(ticker)]
    if company_name:
        first_word = company_name.split()[0].rstrip(".,")
        if len(first_word) > 3:
            terms.append(re.escape(first_word))
        if company_name != first_word:
            terms.append(re.escape(company_name))

    seen_ranges: list[tuple[int, int]] = []
    mentions = []

    for term in terms:
        for m in re.finditer(rf"\b{term}\b", transcript, re.IGNORECASE):
            idx = m.start()
            begin = max(0, idx - window // 2)
            end   = min(len(transcript), idx + window // 2)
            if not any(b <= idx < e for b, e in seen_ranges):
                mentions.append(transcript[begin:end])
                seen_ranges.append((begin, end))

    if not mentions:
        return ""
    return "\n...\n".join(mention_is_relevant(ticker, mentions, company_name))


# ═════════════════════════════════════════════════════════════════════════════
# FILE I/O HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def write_raw(raw_dir: Path, label: str, source: str, text: str) -> None:
    """
    Writes raw scraped text to a labelled source file in the run's raw directory.

    Args:
        raw_dir: Base raw output directory for the current run.
        label: Subdirectory name, typically the ticker symbol.
        source: Source identifier used as the filename stem (e.g. 'finviz', 'yahoofinance').
        text: Raw text content to write.
    """
    filename = f"{source}.txt"
    ticker_dir = raw_dir / label
    ticker_dir.mkdir(parents=True, exist_ok=True)
    filepath = ticker_dir / filename
    filepath.write_text(text, encoding="utf-8")


def write_summary(summary_dir: Path, ticker: str, text: str) -> None:
    """
    Writes an LLM-generated summary to the run's summaries directory.

    Args:
        summary_dir: Summary output directory for the current run.
        ticker: Ticker symbol used as the filename stem.
        text: Summary text to write.
    """
    filename = f"sentiment_summary.txt"
    ticker_dir = summary_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    filepath = ticker_dir / filename
    filepath.write_text(text, encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# LLM SUMMARISATION
# ═════════════════════════════════════════════════════════════════════════════

SUMMARISE_PROMPT = """You are a financial analyst summarising market intelligence for {ticker}.

Below is collected text from multiple sources gathered this week.
Sources are labelled by type — weight them accordingly:
- Yahoo Finance articles: high weight — aggregated from credible outlets
- Finviz articles: high weight — aggregated from credible outlets
- Globe and Mail articles: high weight - aggregated from credible outlets
- ETF.com articles: high weight - aggregated from credible outlets
- Podcast commentary: medium weight if host is credentialed, lower otherwise
- Seeking Alpha headlines: low weight - useful for topic detection, limited depth

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
    Calls the LLM to summarise all collected news text for a ticker.

    Truncates input to 8000 characters before sending to stay within model
    context limits. Uses the SUMMARISE_PROMPT template.

    Args:
        ticker: Ticker symbol, used for prompt context and in fallback messages.
        text: Combined raw text from all scraped sources for this ticker.

    Returns:
        4–6 sentence plain prose summary covering sentiment, catalysts, analyst
        targets, and risks. Returns a fallback string if text is empty or the
        LLM call fails.
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

def agent1_sentiment(state: PipelineState) -> dict:
    """
    Agent 1 — Sentiment Analysis.

    Routes each ticker to appropriate news scrapers based on exchange and security
    type (US stock → Finviz + Seeking Alpha + Yahoo Finance; US ETF → etf.com +
    Yahoo Finance; TSX → Globe and Mail + Yahoo Finance). Fetches macro sentiment
    indicators (AAII, Fear & Greed) and podcast transcripts, then summarises all
    collected text via LLM.

    Args:
        state: Pipeline state dict containing user_name, user_path, tickers,
            run_date, and errors.

    Returns:
        Partial state update dict with keys: raw_text, summaries,
        aaii_sentiment, fear_greed, errors.
    """
    user_name = state["user_name"]
    user_path = Path(state["user_path"])
    tickers   = state["tickers"]
    run_date  = state["run_date"]
    errors    = list(state.get("errors") or [])

    print(f"\n  [Agent 1] Sentiment Analysis running for {user_name}")

    # Create output directories
    raw_dir     = user_path / "data" / run_date / "raw"
    summary_dir = user_path / "data" / run_date / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Build ticker metadata from portfolio CSV.
    # Keyed by base symbol (no .TO) since the CSV stores plain symbols.
    # Use base_ticker = ticker.removesuffix(".TO") to look up in the loop.
    company_names: dict[str, str] = {}
    ticker_meta:   dict[str, dict] = {}
    portfolio_path = user_path / "portfolio.csv"
    if portfolio_path.exists():
        port_df = pd.read_csv(portfolio_path)
        for _, row in port_df.iterrows():
            sym      = str(row["Symbol"]).strip()
            exchange = str(row.get("Exchange", "")).strip()
            sec_type = str(row.get("Security Type", "EQUITY")).strip()
            company_names[sym] = str(row.get("Name", ""))
            ticker_meta[sym]   = {
                "exchange": exchange,
                "is_etf":   "ETF" in sec_type.upper(),
                "is_tsx":   exchange == "TSX",
            }

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
        # ticker is the yfinance symbol (e.g. "TD.TO", "AAPL")
        # base_ticker strips .TO for scrapers, file paths, and transcript search
        base_ticker = ticker.removesuffix(".TO")
        print(f"\n    Scraping {ticker}...")
        meta   = ticker_meta.get(base_ticker, {"exchange": "", "is_etf": False, "is_tsx": False})
        is_tsx = meta["is_tsx"]
        is_etf = meta["is_etf"]
        cname  = company_names.get(base_ticker, "")

        if is_tsx:
            # ── Canadian stock or ETF → Globe and Mail ────────────────────
            gm_results = scrape_globe_and_mail(base_ticker)
            if gm_results:
                gm_results = filter_relevant_headlines(ticker, cname, gm_results)
                for r in gm_results:
                    r["full_text"] = fetch_article_text(r["url"])
                    time.sleep(GENERAL_DELAY)
                gm_text = "\n".join(
                    f"{r.get('published', '')} {r['title']}\n{r['full_text']}"
                    for r in gm_results
                )
                write_raw(raw_dir, base_ticker, "globeandmail", gm_text)
                ticker_text[ticker].append(f"=== Globe and Mail Articles ===\n{gm_text}")
                print(f"    [ok]   Globe and Mail: {len(gm_results)} relevant articles")
            else:
                errors.append(f"{ticker}: Globe and Mail returned no results")
            time.sleep(GENERAL_DELAY)

        elif is_etf:
            # ── US ETF → etf.com ──────────────────────────────────────────
            etf_results = scrape_etf_dot_com(base_ticker)
            if etf_results:
                etf_results = filter_relevant_headlines(ticker, cname, etf_results)
                for r in etf_results:
                    if not r.get("full_text"):
                        r["full_text"] = fetch_article_text(r["url"])
                        time.sleep(GENERAL_DELAY)
                etf_text = "\n".join(
                    f"{r['title']}\n{r['full_text']}"
                    for r in etf_results
                )
                write_raw(raw_dir, base_ticker, "etf_dot_com", etf_text)
                ticker_text[ticker].append(f"=== ETF.com ===\n{etf_text}")
                print(f"    [ok]   etf.com: {len(etf_results)} relevant articles")
            else:
                errors.append(f"{ticker}: etf.com returned no results")
            time.sleep(GENERAL_DELAY)

        else:
            # ── US Stock → Finviz + Seeking Alpha ─────────────────────────
            finviz_results = scrape_finviz(base_ticker)
            if finviz_results:
                finviz_results = filter_relevant_headlines(ticker, cname, finviz_results)
                for r in finviz_results:
                    r["full_text"] = fetch_article_text(r["url"])
                    time.sleep(GENERAL_DELAY)
                finviz_text = "\n".join(
                    f"{r['date']} {r['time']} [{r['source']}] {r['headline']}\n{r['full_text']}"
                    for r in finviz_results
                )
                write_raw(raw_dir, base_ticker, "finviz", finviz_text)
                ticker_text[ticker].append(f"=== Finviz Articles ===\n{finviz_text}")
                print(f"    [ok]   Finviz: {len(finviz_results)} relevant articles")
            else:
                errors.append(f"{ticker}: Finviz returned no results")
            time.sleep(FINVIZ_DELAY)

            sa_results = scrape_seeking_alpha(base_ticker)
            if sa_results:
                sa_text = "\n".join(
                    f"{r['published']} {r['title']}"
                    + (" [paywalled]" if r["paywalled"] else "")
                    for r in sa_results
                )
                write_raw(raw_dir, base_ticker, "seekingalpha", sa_text)
                ticker_text[ticker].append(f"=== Seeking Alpha Headlines ===\n{sa_text}")
                print(f"    [ok]   Seeking Alpha: {len(sa_results)} headlines")
            else:
                errors.append(f"{ticker}: Seeking Alpha returned no results")
            time.sleep(SEEKALPHA_DELAY)

        # ── Yahoo Finance — all tickers; ticker already has .TO for TSX ──
        yf_results = scrape_yahoo_finance(ticker)
        if yf_results:
            yf_results = filter_relevant_headlines(ticker, cname, yf_results)
            for r in yf_results:
                r["full_text"] = fetch_article_text(r["url"])
                time.sleep(GENERAL_DELAY)
            yf_text = "\n".join(
                f"{r['published']} {r['title']} [{r['snippet']}]\n{r['full_text']}"
                for r in yf_results
            )
            write_raw(raw_dir, base_ticker, "yahoofinance", yf_text)
            ticker_text[ticker].append(f"=== Yahoo Finance Articles ===\n{yf_text}")
            print(f"    [ok]   Yahoo Finance ({ticker}): {len(yf_results)} relevant articles")
        else:
            errors.append(f"{ticker}: Yahoo Finance returned no results")

        # break # For testing

    # ── Macro signals (portfolio-level, not per-ticker) ───────────────────────
    print(f"\n    Fetching macro signals...")

    # --- Market Sentiment ---
    aaii = fetch_aaii_sentiment()
    write_raw(raw_dir, "MARKET", "aaii", json.dumps(aaii, indent=2))
    if "error" not in aaii:
        print(f"    [ok]   AAII: bullish={aaii.get('bullish')} bearish={aaii.get('bearish')}")
    else:
        errors.append(f"AAII fetch failed: {aaii.get('error')}")

    fg = fetch_fear_greed()
    write_raw(raw_dir, "MARKET", "fear_greed", json.dumps(fg, indent=2))
    if "error" not in fg:
        print(f"    [ok]   Fear & Greed: {fg.get('score')} ({fg.get('rating')})")
    else:
        errors.append(f"Fear & Greed fetch failed: {fg.get('error')}")

    # ── Podcast transcripts ───────────────────────────────────────────────────
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
                podcast_raw_sections = []
                for ticker in tickers:
                    base_ticker = ticker.removesuffix(".TO")
                    mentions = extract_ticker_mentions(transcript, base_ticker, company_name=company_names.get(base_ticker, ""))
                    if mentions:
                        label = f"=== Podcast: {podcast['name']} (credibility: {podcast.get('credibility', 'medium')}) ==="
                        ticker_text[ticker].append(f"{label}\n{mentions}")
                        podcast_raw_sections.append(f"[{ticker}]\n{mentions}")
                        print(f"    [ok]   Found {ticker} mentions in {podcast['name']}")

                if podcast_raw_sections:
                    write_raw(raw_dir, "PODCAST", podcast["name"], "\n\n".join(podcast_raw_sections))

        time.sleep(GENERAL_DELAY)

    # ── LLM summarisation ─────────────────────────────────────────────────────
    print(f"\n    Summarising collected text via LLM...")
    summaries = {}
    raw_text_out = {}

    for ticker in tickers:
        # Only news text goes to the LLM — AAII, Fear & Greed, and
        # earnings dates are structured signals that Agent 3 reads directly
        combined = "\n\n".join(ticker_text[ticker])
        raw_text_out[ticker] = combined

        summary = summarise_ticker_text(ticker, combined)
        summaries[ticker] = summary
        write_summary(summary_dir, ticker.removesuffix(".TO"), summary)
        print(f"    [ok]   {ticker}: summary written ({len(summary)} chars)")

    print(f"\n  [Agent 1] Complete. Errors: {len(errors)}")

    return {
        "raw_text":       raw_text_out,
        "summaries":      summaries,
        "aaii_sentiment": aaii,
        "fear_greed":     fg,
        "errors":         errors,
    }