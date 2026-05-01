from agents.state import PipelineState


def agent3_advisor(state: PipelineState) -> dict:
    """
    Agent 3 — Portfolio Advisor
    Generates Black-Litterman views, runs optimisation,
    produces recommendation table and advisory commentary.
    """
    print(f"  [Agent 3] Advisor running")

    n = len(state["tickers"])
    equal_weight = round(1.0 / n, 4)

    return {
        "bl_views": {
            t: {"view_return": 0.0, "confidence": 1, "reasoning": "[STUB]"}
            for t in state["tickers"]
        },
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