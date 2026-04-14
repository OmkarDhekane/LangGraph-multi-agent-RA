"""
Critic Agent
------------
Validates the draft answer against source evidence.
Flags unsupported claims (hallucinations) and decides
whether the answer needs revision or can be finalized.
"""

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState

MAX_ITERATIONS = 2  # Max critic-revision loops before forcing finalization

CRITIC_SYSTEM_PROMPT = """You are a rigorous Fact-Checking Critic AI.
Your job is to validate a draft answer against the original source evidence.

For each claim in the draft answer:
1. Check if it is directly supported by the provided sources
2. Flag any claims that are speculative, unsupported, or potentially hallucinated
3. Assess overall answer quality

Respond in this EXACT format:

VERDICT: <APPROVE|REVISE>
CONFIDENCE_SCORE: <0.0-1.0>
HALLUCINATION_FLAGS:
- <flag 1 or "None">
- <flag 2>
QUALITY_ISSUES:
- <issue 1 or "None">
REVISION_INSTRUCTIONS:
<specific instructions for improvement, or "None" if approving>
REASONING:
<brief explanation of your verdict>
"""


def critic_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """
    Critic node: validates draft answer, flags hallucinations, routes to revise or finalize.
    """
    start = time.time()
    query = state["query"]
    draft = state.get("draft_answer", "")
    search_results = state.get("search_results", [])
    iteration = state.get("iteration_count", 0)

    # Force approval after MAX_ITERATIONS to prevent infinite loops
    if iteration >= MAX_ITERATIONS:
        latency = round(time.time() - start, 2)
        return {
            **state,
            "needs_revision": False,
            "final_answer": draft,
            "status": "finalized",
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "critic": 0.75,
                "critic_latency_s": latency,
                "critic_forced_approval": True
            },
            "agent_logs": [
                f"[CRITIC] Max iterations ({MAX_ITERATIONS}) reached. Force-approving answer."
            ]
        }

    # Format sources for critic context
    sources_text = "\n\n".join([
        f"[Source {i+1}]: {r['content'][:500]}"
        for i, r in enumerate(search_results)
    ])

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Original Query: {query}

Source Evidence:
{sources_text}

Draft Answer to Validate:
{draft}

Please validate this answer against the sources.
""")
    ]

    response = llm.invoke(messages)
    critique = response.content
    latency = round(time.time() - start, 2)

    # Parse verdict
    needs_revision = "VERDICT: REVISE" in critique.upper()

    # Parse confidence score
    try:
        score_line = [l for l in critique.split("\n") if "CONFIDENCE_SCORE:" in l][0]
        confidence = float(score_line.split(":")[1].strip())
    except (IndexError, ValueError):
        confidence = 0.8 if not needs_revision else 0.6

    # Parse hallucination flags
    hallucination_flags = []
    if "HALLUCINATION_FLAGS:" in critique:
        flags_section = critique.split("HALLUCINATION_FLAGS:")[1].split("QUALITY_ISSUES:")[0]
        flags = [f.strip("- ").strip() for f in flags_section.strip().split("\n") if f.strip() and f.strip() != "- None" and f.strip() != "None"]
        hallucination_flags = [f for f in flags if f]

    return {
        **state,
        "critique": critique,
        "needs_revision": needs_revision,
        "final_answer": "" if needs_revision else draft,
        "status": "needs_revision" if needs_revision else "finalized",
        "iteration_count": iteration + 1,
        "hallucination_flags": hallucination_flags,
        "confidence_scores": {
            **state.get("confidence_scores", {}),
            "critic": round(confidence, 2),
            "critic_latency_s": latency,
            "hallucinations_found": len(hallucination_flags)
        },
        "agent_logs": [
            f"[CRITIC] Iteration {iteration+1} | Verdict: {'REVISE' if needs_revision else 'APPROVE'} | "
            f"Confidence: {round(confidence, 2)} | Flags: {len(hallucination_flags)} | Latency: {latency}s"
        ]
    }
