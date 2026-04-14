"""
LangGraph Pipeline
------------------
Wires all agents into a directed graph with conditional routing.

Graph flow:
  START
    → supervisor       (analyze query, create plan)
    → researcher       (web search via Tavily)
    → analyst          (synthesize draft answer)
    → critic           (validate, flag hallucinations)
    → [REVISE?]
        YES → reviser → critic (loop, max 2 iterations)
        NO  → END (finalized answer)
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.state import AgentState
from agents.supervisor import supervisor_node
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.critic import critic_node
from agents.reviser import reviser_node

load_dotenv()


def build_graph():
    """Build and compile the multi-agent LangGraph pipeline."""

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY not set. Please add your Gemini API key to .env file.\n"
            "Get one free at: https://aistudio.google.com/app/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.3
    )

    # Bind LLM to agents that need it
    def supervisor(state): return supervisor_node(state, llm)
    def analyst(state): return analyst_node(state, llm)
    def critic(state): return critic_node(state, llm)
    def reviser(state): return reviser_node(state, llm)

    # Routing function: after critic, go to reviser or END
    def route_after_critic(state: AgentState) -> str:
        if state.get("needs_revision") and state.get("iteration_count", 0) < 2:
            return "reviser"
        return END

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst)
    graph.add_node("critic", critic)
    graph.add_node("reviser", reviser)

    # Add edges
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"reviser": "reviser", END: END})
    graph.add_edge("reviser", "critic")  # Loop back to critic after revision

    return graph.compile()


def run_pipeline(query: str) -> AgentState:
    """Run the full research pipeline for a query."""
    graph = build_graph()

    initial_state: AgentState = {
        "query": query,
        "search_results": [],
        "document_analysis": "",
        "draft_answer": "",
        "critique": "",
        "final_answer": "",
        "confidence_scores": {},
        "agent_logs": [],
        "iteration_count": 0,
        "hallucination_flags": [],
        "needs_revision": False,
        "status": "started"
    }

    result = graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    # Quick CLI test
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is LangGraph and how does it differ from LangChain?"
    print(f"\n🔍 Query: {query}\n{'='*60}")
    result = run_pipeline(query)
    print(f"\n✅ Final Answer:\n{result['final_answer']}")
    print(f"\n📊 Confidence Scores: {result['confidence_scores']}")
    print(f"\n🔄 Agent Logs:")
    for log in result['agent_logs']:
        print(f"  {log}")
