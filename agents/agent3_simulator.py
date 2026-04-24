from agents.state import PipelineState


def agent3_simulator(state: PipelineState) -> dict:
    """
    Agent 3 — Monte Carlo Simulator
    Runs simulations for current and candidate rebalanced portfolios.
    """
    print(f"  [Agent 3] Simulator running")

    stub_mc = {
        "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0,
        "var_95": 0.0, "cvar_95": 0.0, "prob_loss": 0.0
    }

    return {
        "mc_current":     stub_mc,
        "mc_rebalanced":  stub_mc,
        "risk_commentary": "[STUB] Simulation not yet implemented.",
    }