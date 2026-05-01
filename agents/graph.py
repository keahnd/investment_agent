from langgraph.graph import StateGraph, END

from agents.state import PipelineState
from agents.agent1_sentiment import agent1_sentiment
from agents.agent2_quant import agent2_quant
from agents.agent4_simulator import agent4_simulator
from agents.agent3_advisor import agent3_advisor


def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    # Register the four nodes
    graph.add_node("setiment", agent1_sentiment)
    graph.add_node("quant",  agent2_quant)
    graph.add_node("advisor",   agent3_advisor)
    graph.add_node("simulator", agent4_simulator)

    # Wire them in sequence
    graph.set_entry_point("setiment")
    graph.add_edge("setiment", "quant")
    graph.add_edge("quant",  "advisor")
    graph.add_edge("advisor", "simulator")
    graph.add_edge("simulator",   END)

    return graph.compile()


pipeline_graph = build_pipeline_graph()