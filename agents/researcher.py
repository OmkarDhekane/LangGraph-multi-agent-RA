"""
Researcher Agent
----------------
Uses Tavily search API to gather web evidence for the query.
Returns structured search results with source URLs and snippets.
"""

import time
import os
from typing import List
from agents.state import AgentState

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


def _mock_search(query: str) -> List[dict]:
    """Fallback mock results when Tavily key is not configured."""
    return [
        {
            "title": f"Mock Result 1 for: {query}",
            "url": "https://example.com/result1",
            "content": f"This is a simulated search result about {query}. "
                       f"In production, real web results would appear here via Tavily API.",
            "score": 0.95
        },
        {
            "title": f"Mock Result 2 for: {query}",
            "url": "https://example.com/result2",
            "content": f"Additional context about {query}. "
                       f"Configure TAVILY_API_KEY in .env to get real search results.",
            "score": 0.88
        },
        {
            "title": f"Mock Result 3 for: {query}",
            "url": "https://example.com/result3",
            "content": f"Supporting information related to {query}.",
            "score": 0.81
        }
    ]


def researcher_node(state: AgentState) -> AgentState:
    """
    Researcher node: performs web search and returns structured results.
    """
    start = time.time()
    query = state["query"]
    api_key = os.getenv("TAVILY_API_KEY", "")

    if TAVILY_AVAILABLE and api_key and api_key != "your_tavily_api_key_here":
        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=True
            )
            results = response.get("results", [])
            # Normalize to standard format
            search_results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0)
                }
                for r in results
            ]
            source = "tavily"
        except Exception as e:
            search_results = _mock_search(query)
            source = f"mock (tavily error: {e})"
    else:
        search_results = _mock_search(query)
        source = "mock (no API key)"

    latency = round(time.time() - start, 2)
    confidence = min(1.0, sum(r["score"] for r in search_results) / len(search_results)) if search_results else 0.5

    return {
        **state,
        "search_results": search_results,
        "status": "researched",
        "confidence_scores": {
            **state.get("confidence_scores", {}),
            "researcher": round(confidence, 2),
            "researcher_latency_s": latency,
            "researcher_source": source
        },
        "agent_logs": [
            f"[RESEARCHER] Found {len(search_results)} results in {latency}s via {source}"
        ]
    }
