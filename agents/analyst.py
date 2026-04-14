"""
Analyst (Dispatcher) Agent
--------------------------
Synthesizes search results from the Researcher into a
structured draft answer. This is the core "answer builder."
"""

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState


ANALYST_SYSTEM_PROMPT = """You are an expert Research Analyst. Your job is to synthesize 
web search results into a clear, well-structured answer.

Rules:
1. Only use information present in the provided search results
2. Cite sources inline using [Source N] notation
3. Structure your answer with: Summary, Key Findings, Supporting Evidence
4. Flag any gaps where evidence is weak or missing
5. Be factual and precise — avoid speculation

Format your response as:
## Summary
<2-3 sentence overview>

## Key Findings
- Finding 1 [Source N]
- Finding 2 [Source N]
- ...

## Supporting Evidence
<detailed explanation with citations>

## Evidence Gaps
<any areas where more research may be needed>
"""


def analyst_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """
    Analyst node: synthesizes search results into a draft answer.
    """
    start = time.time()
    query = state["query"]
    search_results = state.get("search_results", [])

    # Format search results for the LLM
    formatted_results = "\n\n".join([
        f"[Source {i+1}] {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for i, r in enumerate(search_results)
    ])

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Query: {query}

Search Results:
{formatted_results}

Please synthesize these results into a comprehensive answer.
""")
    ]

    response = llm.invoke(messages)
    draft = response.content
    latency = round(time.time() - start, 2)

    # Heuristic confidence: longer, more cited answers score higher
    citation_count = draft.count("[Source")
    confidence = min(0.95, 0.6 + (citation_count * 0.05))

    return {
        **state,
        "document_analysis": draft,
        "draft_answer": draft,
        "status": "analyzed",
        "confidence_scores": {
            **state.get("confidence_scores", {}),
            "analyst": round(confidence, 2),
            "analyst_latency_s": latency,
            "analyst_citations": citation_count
        },
        "agent_logs": [
            f"[ANALYST] Draft generated in {latency}s | Citations: {citation_count} | Confidence: {round(confidence, 2)}"
        ]
    }
