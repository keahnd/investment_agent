from pathlib import Path
import json
import pandas as pd
import numpy as np
from pypfopt.black_litterman import BlackLittermanModel

from agents.state import PipelineState
from agents.llm import get_llm

UNCERTAINTY_SCALE = 0.05   # from config — tune this over time


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


def confidence_to_omega(confidence, asset_variance, uncertainty_scale):
    """
    Converts a 1-5 confidence score to an omega uncertainty value.
    Confidence 5 = low uncertainty = small omega = view pulls posterior strongly
    Confidence 1 = high uncertainty = large omega = view barely moves posterior
    """
    # Invert confidence so high confidence = small uncertainty
    uncertainty_factor = (6 - confidence) / 5.0
    # Scale 1.0 at confidence 1 down to 0.2 at confidence 5
    return uncertainty_factor * uncertainty_scale * asset_variance


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
    
{data}

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


def _parse_views(raw: str, tickers: list[str]) -> dict:
    """
    Parses and validates the LLM JSON response for sentiment views.
    Strips markdown fences, clips numeric fields, and fills neutral fallbacks
    for any tickers missing from the response.
    """
    def _neutral() -> dict:
        return {
            "view_return": 0.0,
            "confidence": 1,
            "sentiment_direction": "neutral",
            "valuation_signal": "neutral",
            "conflict": False,
            "reasoning": "No view generated — using neutral fallback.",
        }

    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        views = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    [warn] LLM returned invalid JSON: {e}")
        return {ticker: _neutral() for ticker in tickers}

    for ticker in tickers:
        if ticker not in views:
            views[ticker] = _neutral()
        else:
            views[ticker]["view_return"] = max(-0.30, min(0.30, float(views[ticker].get("view_return", 0.0))))
            views[ticker]["confidence"]  = max(1, min(5, int(views[ticker].get("confidence", 1))))

    return views


def generate_sentiment_views(
        tickers: list[str],
        summaries: dict,
        earnings_dates: dict,
        valuations: dict,
        aaii_sentiment: dict,
        fear_greed: dict,
        cape: float,
        file = None,
    ) -> dict:
    """
    Calls the LLM to interpret sentiment data per ticker and returns a parsed
    dict keyed by ticker. Falls back to neutral values if the call fails.
    """
    data_lines = []

    data_lines.append(f"""Market Data:
        Fear/Greed: {fear_greed}
        aaii_sentiment: {aaii_sentiment}
        Shiller Cape: {cape}""")

    for ticker in tickers:
        sent          = summaries.get(ticker, {})
        earnings_date = earnings_dates.get(ticker, {})
        valuation     = valuations.get(ticker, {})

        data_lines.append(f"""Ticker = {ticker}:
        News Summary: {sent}, \
        Next Earnings Date: {earnings_date}
        Valuation Data: {valuation}""")

    prompt = SENTIMENT_PROMPT.format(
        data="\n".join(data_lines),
    )

    try:
        llm      = get_llm()
        response = llm.invoke(prompt)
        raw = response.content.strip()
        return _parse_views(raw, tickers)
    except Exception as e:
        print(f"    [warn] LLM call failed: {e}")
        return _parse_views("", tickers)


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
    cov_df = pd.DataFrame(state["covariance_matrix"])
    cov_matrix = cov_df.loc[tickers, tickers]
    
    constraints = load_constraints(user_path)
    
    bl_views = generate_sentiment_views(tickers, state["summaries"], state["earnings_dates"],
                    state["valuation"], state["aaii_sentiment"], state["fear_greed"], state["cape"])
    
    views = {
        ticker: prior_mu[i] + bl_views[ticker]["view_return"]
        for i, ticker in enumerate(tickers)
    }
    variances = np.diag(cov_matrix.values)   # diagonal of covariance matrix
    omega_diag = np.array([
        confidence_to_omega(
            bl_views[t]["confidence"],
            variances[i],
            UNCERTAINTY_SCALE
        )
        for i, t in enumerate(tickers)
    ])
    omega = np.diag(omega_diag)
    print(f"omega: {omega}")
    bl = BlackLittermanModel(
        cov_matrix   = cov_matrix,
        pi           = np.array(prior_mu),          # factor model mu as prior
        absolute_views = views,           # {ticker: view_return}
        omega        = omega,             # uncertainty matrix
    )
    
    posterior_mu  = bl.bl_returns().to_dict()    # pandas Series, one value per ticker
    posterior_cov = bl.bl_cov()        # DataFrame, posterior covariance matrix  
    
    
    raw_dir = Path(state["user_path"]) / "data" / state["run_date"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, ticker in enumerate(tickers):
        (raw_dir / ticker).mkdir(parents=True, exist_ok=True)
        with open(raw_dir / ticker / "advisor.txt", "w", encoding="utf-8") as advisor_file:
            print(f"Historical Returns Estimate: {prior_mu[i]}")#, file=advisor_file)
            print(f"Views: {bl_views[ticker]}")#, file=advisor_file)
            print(f"posterior returns: {posterior_mu[ticker]}")#, file=advisor_file)

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