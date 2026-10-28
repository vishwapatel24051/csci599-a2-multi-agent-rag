"""Trace events: the inspectable record of how a turn moved through the system.

Inside a LangGraph run, events go to the graph's `custom` stream (the API forwards
them to the browser as SSE `trace` frames) and are tagged with the emitting node.
Outside a graph (scripts, tests) they are only logged.

Event kinds: agent_route, llm_route, llm_call, llm_error, rag_retrieval, mcp_tool_call,
tool_call, agent_start, agent_result, agent_error.
"""

import logging
import time

log = logging.getLogger("trace")


def emit(kind: str, data: dict) -> None:
    event = {"kind": kind, "ts": round(time.time(), 3), **data}
    try:
        from langgraph.config import get_config, get_stream_writer

        event.setdefault("node", get_config().get("metadata", {}).get("langgraph_node"))
        get_stream_writer()(event)
    except Exception:  # noqa: BLE001 — not inside a graph run
        pass
    log.info("%s %s", kind, {k: v for k, v in event.items() if k not in ("kind", "ts")})
