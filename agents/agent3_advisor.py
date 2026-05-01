from pathlib import Path
import json

from agents.state import PipelineState
from agents.llm import get_llm

def load_constraints(user_path: Path) -> dict:
    """
    Loads the user constriants from the config file
    
    Args:
        user_path: Path to config file
        
    Rerturns:
        dict of contraints
    """
    config_path = user_path / "config.json"
    constr = {}
    if config_path.exists():
        with open(config_path) as f:
            constr = {
                k: v for k, v in json.load(f).items()
                if k in ("risk_tolerance", "investment_horizon_years",
                    "max_allocation_per_asset", "max_allocation_per_sector", "min_cash_buffer")
            }
    return constr


# LLM Sentiment Analysis
SENTIMENT_PROMPT = """You are a quantitative analyst reviewing news, valuations and sentiment data for assets.

For each ticker below, analyse the summary data, the valuations, the next earnings call and the macro economic data
of the shiller cape value, the aaii sentiment (weekly investor sentiment survey), CNN's fear/greed index.

Data:
Macro context:
  CAPE: 
  Fear & Greed:
  AAII Sentiment:

Company context:
  Sector:
  Industry:
  Exchange:

  Sentiment summary: summary

  Quantitative signals: Valuation data

  Earnings proximity:  Next earnings
    Note: if earnings are within 7 days, reduce confidence by 1 and explicitly indicate earnings are upcoming.
    
{date}

Reference thresholds (use as guidelines, not hard rules):
  PEG:      < 0.5 strongly cheap, < 0.8 cheap, 1.0 fair, > 2.0 expensive
  Fwd P/E:  < 13 cheap, 16 market average, > 25 expensive
  TTM P/E:  < 14 cheap, 17 market average, > 30 expensive
  EV/EBITDA: evaluate relative to sector peers
  
Important context to apply:
  - Technology and growth companies typically trade at premium multiples
    — a 30x P/E for a high-growth tech company may not be expensive
  - Canadian TSX stocks typically trade at a 2-3 point P/E discount to US peers
  - Energy and financial companies have cyclical earnings — P/E is less reliable,
    weight EV/EBITDA and book value more heavily for these sectors
  - Negative P/E means the company is loss-making — do not apply P/E thresholds
  - Very high P/E (above 100) usually means near-zero earnings — treat as uninformative
  - If sentiment and valuation conflict, note this explicitly and explain your reasoning
  
Produce a single JSON object where each key is a ticker symbol and 
each value contains:
  view_return:         float  (annual return relative to equilibrium, e.g. 0.03 = 3 percent above)
  confidence:          integer 1-5
  sentiment_direction: string  (positive / negative / neutral)
  valuation_signal:    string  (cheap / expensive / neutral / not applicable)
  conflict:            boolean (true if sentiment and valuation disagree)
  reasoning:           string  (2-3 sentences explaining the view)
  
Return only valid JSON. No preamble, no markdown code fences.
Ticker keys must exactly match the ticker values passed in.
"""

def generate_sentiment_views(
        tickers: list[str],
        summaries: dict,
        earnings_dates: dict,
        valuations: dict,
        aaii_sentiment: dict,
        fear_greed: dict,
        cape: float,
        file = None,
    ) -> json:
    """
    Calls the LLM to interpret sentiment outputs, from agent 1 per ticker.

    Passes sentiment data to SENTIMENT_PROMPT, which asks the LLM LLM to produce a JSON structure per ticker 
    containing view_return_raw (estimated annual return relative to market equilibrium,
    e.g. 0.03 meaning 3% above), confidence (1 to 5), and reasoning (one sentence)

    Args:
        tickers: List of ticker symbols to include in the commentary.
        summaries: Dict of sentiment data gathered per ticker.
        earnings_dates: Dict of next earnings date for each ticker.
        aaii_sentiment: Dict of market sentiment.
        fear_greed: Dict of market fear/greed indication.
        cape: float the current shiller cape value
        file: Text file to store raw output.

    Returns:
        Plain prose commentary, one paragraph per ticker labelled with the
        ticker symbol. Returns a fallback string if the LLM call fails.
    """
    # Build a structured summary of all metrics per ticker
    data_lines = []
    
    data_lines.append(f"""Market Data:
        Fear/Greed: {fear_greed}
        aaii_sentiment: {aaii_sentiment}
        Shiller Cape: {cape}""")
    
    for ticker in tickers:
        sent  = summaries.get(ticker, {})
        earnings_date  = earnings_dates.get(ticker, {})
        valuation = valuations.get(ticker, {})

        data_lines.append(f"""Ticker = {ticker}:
        News Summary: {sent}, \
        Next Earnings Date: {earnings_date}
        Valuation Data: {valuation}""")

    prompt = SENTIMENT_PROMPT.format(
        data = "\n".join(data_lines),
    )

    try:
        llm      = get_llm()
        response = llm.invoke(prompt)
        print(f"\n LLM Commentary:{response.content.strip()}", file=file)
        return response.content.strip()
    except Exception as e:
        print(f"    [warn] LLM commentary failed: {e}")
        return "Quantitative commentary unavailable this run."


def agent3_advisor(state: PipelineState) -> dict:
    """
    Agent 3 — Portfolio Advisor
    Generates Black-Litterman views, runs optimisation,
    produces recommendation table and advisory commentary.
    """
    print(f"  [Agent 3] Advisor running")
    
    user_name = state["user_name"]
    user_path = Path(state["user_path"])
    tickers   = state["tickers"]
    run_date  = state["run_date"]
    errors    = list(state.get("errors") or [])
    prior_mu = [state["mu_sigma"][t]["mu_annual"] for t in tickers]
    
    constraints = load_constraints(user_path)
    
    bl_views = generate_sentiment_views(tickers, state["summaries"], state["earnings_dates"], 
                    state["valuation"], ["aaii_sentiment"], state["fear_greed"], state["cape"])
        

    n = len(state["tickers"])
    equal_weight = round(1.0 / n, 4)

    return {
        "bl_views": bl_views,
        "recommended_weights": {t: equal_weight for t in state["tickers"]},
        "recommendation_table": [
            {
                "ticker":               t,
                "current_weight":       None,
                "recommended_weight":   equal_weight,
                "delta":                None,
                "action":               "STUB",
            }
            for t in state["tickers"]
        ],
        "advisory_commentary": "[STUB] Advisory commentary not yet implemented.",
    }