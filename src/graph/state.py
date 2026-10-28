"""Graph state: the single object every node reads and returns partial updates to.

Conversation-level fields persist across turns via the checkpointer (thread_id = session_id).
Turn-level fields are reset by `start_turn` at the beginning of every turn: their reducers
treat an update of `None` as "reset", otherwise they accumulate.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def reset_or_append(current: list | None, update: list | None) -> list:
    return [] if update is None else (current or []) + update


def reset_or_merge(current: dict | None, update: dict | None) -> dict:
    return {} if update is None else {**(current or {}), **update}


class RouteRecord(TypedDict):
    hop: int
    next: str  # agent name or "FINISH"
    task: str
    complexity: str  # low | medium | high
    reason: str
    decided_by: str  # "llm" | "guard: <why>"


class AgentResult(TypedDict, total=False):
    agent: str
    task: str
    status: str  # ok | error
    output: str
    error: str
    cited: list[str]
    provider: str
    model: str
    slot: str
    tools_used: list[str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    llm_calls: int


class GraphState(TypedDict, total=False):
    # --- conversation (persists across turns) ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- turn inputs ---
    query: str
    policy: str  # economy | balanced | quality
    forced_provider: str | None  # evaluation only

    # --- turn working state (reset each turn) ---
    routes: Annotated[list[RouteRecord], reset_or_append]
    results: Annotated[list[AgentResult], reset_or_append]
    sources: Annotated[dict[str, dict[str, Any]], reset_or_merge]  # label → Source
    hops: int

    # --- turn outputs ---
    final_answer: str
    turn_stats: dict[str, Any]
