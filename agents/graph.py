from langgraph.graph import StateGraph, END

from agents.state import PipelineState
from agents.agent1_data_harvester import agent1_harvester
from agents.agent2_modeller import agent2_modeller
from agents.agent3_simulator import agent3_simulator
from agents.agent4_advisor import agent4_advisor


def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    # Register the four nodes
    graph.add_node("harvester", agent1_harvester)
    graph.add_node("modeller",  agent2_modeller)
    graph.add_node("simulator", agent3_simulator)
    graph.add_node("advisor",   agent4_advisor)

    # Wire them in sequence
    graph.set_entry_point("harvester")
    graph.add_edge("harvester", "modeller")
    graph.add_edge("modeller",  "simulator")
    graph.add_edge("simulator", "advisor")
    graph.add_edge("advisor",   END)

    return graph.compile()


pipeline_graph = build_pipeline_graph()