"""
Reviser Agent
-------------
Takes the critic's feedback and rewrites the draft answer
to address hallucinations and quality issues.
"""

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState


REVISER_SYSTEM_PROMPT = """You are a precise Answer Reviser AI.
You will receive:
1. An original draft answer
2. Critic feedback with specific issues to fix
3. The original source evidence

Your job is to rewrite the answer to:
1. Remove or fix all flagged hallucinations
2. Address all quality issues mentioned
3. Maintain the same structure (Summary / Key Findings / Supporting Evidence)
4. Only include claims directly supported by sources
5. Be more precise and conservative where evidence is weak

Produce a revised, improved answer only. Do not include meta-commentary.
"""


def reviser_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """
    Reviser node: rewrites the draft based on critic feedback.
    """
    start = time.time()
    query = state["query"]
    draft = state.get("draft_answer", "")
    critique = state.get("critique", "")
    search_results = state.get("search_results", [])

    sources_text = "\n\n".join([
        f"[Source {i+1}]: {r['content'][:500]}"
        for i, r in enumerate(search_results)
    ])

    messages = [
        SystemMessage(content=REVISER_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Original Query: {query}

Source Evidence:
{sources_text}

Original Draft:
{draft}

Critic Feedback:
{critique}

Please produce a revised answer addressing all the critic's concerns.
""")
    ]

    response = llm.invoke(messages)
    revised = response.content
    latency = round(time.time() - start, 2)

    return {
        **state,
        "draft_answer": revised,
        "status": "revised",
        "confidence_scores": {
            **state.get("confidence_scores", {}),
            "reviser_latency_s": latency
        },
        "agent_logs": [
            f"[REVISER] Answer revised in {latency}s"
        ]
    }
