from agents.state import PipelineState


def agent1_harvester(state: PipelineState) -> dict:
    """
    Agent 1 — Data Harvester
    Scrapes news, sentiment surveys, podcast transcripts.
    Produces per-ticker text summaries for Agent 4's view generation.
    """
    print(f"  [Agent 1] Harvester running for {state['user_name']}")
    print(f"            Tickers : {state['tickers']}")
    print(f"            Date    : {state['run_date']}")

    return {
        "raw_text": {t: "" for t in state["tickers"]},
        "summaries": {
            t: f"[STUB] No data collected for {t} yet."
            for t in state["tickers"]
        },
        "aaii_sentiment": {"bullish": "N/A", "bearish": "N/A", "neutral": "N/A"},
        "fear_greed":     {"score": None, "rating": "N/A"},
        "earnings_dates": {t: None for t in state["tickers"]},
        "errors":         [],
    }