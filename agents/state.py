"""
Shared state schema for the multi-agent research assistant.
All agents read from and write to this TypedDict.
"""

from typing import TypedDict, List, Optional, Annotated
import operator


class AgentState(TypedDict):
    # Core query
    query: str

    # Intermediate outputs from each agent
    search_results: List[dict]        # Researcher agent output
    document_analysis: str            # Analyst agent output
    draft_answer: str                 # Dispatcher-assembled draft
    critique: str                     # Critic agent feedback
    final_answer: str                 # Final validated answer

    # Observability / metrics
    confidence_scores: dict           # Per-agent confidence scores
    agent_logs: Annotated[List[str], operator.add]  # Append-only log
    iteration_count: int              # How many critic loops have run
    hallucination_flags: List[str]    # Flagged unsupported claims

    # Routing control
    needs_revision: bool              # Critic says revise?
    status: str                       # current pipeline stage
