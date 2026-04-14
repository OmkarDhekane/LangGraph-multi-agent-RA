"""
Supervisor Agent
----------------
Analyzes the incoming query, decides the execution plan,
and sets up initial state for the dispatcher.
"""

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState


SUPERVISOR_SYSTEM_PROMPT = """You are a Supervisor AI that analyzes research queries.
Your job is to:
1. Understand what the user is asking
2. Identify what kind of information is needed (factual, analytical, comparative, etc.)
3. Decompose the query into sub-tasks for specialized agents

Respond in this exact format:
QUERY_TYPE: <factual|analytical|comparative|technical>
SEARCH_NEEDED: <yes|no>
DOCUMENT_ANALYSIS_NEEDED: <yes|no>
SUB_TASKS:
- <sub-task 1>
- <sub-task 2>
- <sub-task 3>
COMPLEXITY: <simple|medium|complex>
"""


def supervisor_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """
    Supervisor node: parse query, create execution plan.
    """
    start = time.time()
    query = state["query"]

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze this research query: {query}")
    ]

    response = llm.invoke(messages)
    plan = response.content

    latency = round(time.time() - start, 2)

    # Parse complexity to set confidence baseline
    confidence = 0.9 if "simple" in plan.lower() else 0.75 if "medium" in plan.lower() else 0.65

    return {
        **state,
        "status": "supervised",
        "confidence_scores": {
            **state.get("confidence_scores", {}),
            "supervisor": confidence,
            "supervisor_latency_s": latency
        },
        "agent_logs": [
            f"[SUPERVISOR] Query analyzed in {latency}s | Plan:\n{plan}"
        ]
    }
