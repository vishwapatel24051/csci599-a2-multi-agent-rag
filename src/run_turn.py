"""Run chat turns through the graph from the terminal, printing the live trace and streamed tokens.

Usage:
  python -m src.run_turn "How does ReAct differ from chain-of-thought?"
  python -m src.run_turn --policy economy "first question" "follow-up question"   # multi-turn, same thread
"""

import argparse
import asyncio
import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.graph.build import build_graph

COMPACT = {
    "agent_route": lambda e: f"→ {e['next']} [{e['decided_by']}] ({e['complexity']}) {e['reason']}",
    "llm_route": lambda e: f"  model {e['slot']} ({e['dimension']}) for {e['role']}: {e['reason']}",
    "llm_call": lambda e: f"  llm {e['model']} {e['latency_s']}s in={e['input_tokens']} out={e['output_tokens']} ${e['cost_usd']}",
    "llm_error": lambda e: f"  LLM ERROR {e['slot']}: {e['error'][:120]}",
    "rag_retrieval": lambda e: f"  rag[{e['agent']}] '{e['query'][:60]}' → " + ", ".join(
        f"{h['label']}:{h['chunk_id']}({h['score']})" for h in e["hits"]),
    "mcp_tool_call": lambda e: f"  mcp[{e['agent']}] {e['server']}.{e['tool']} {e['status']} "
                               f"{e.get('n_results', '')} results {e['latency_s']}s",
    "tool_call": lambda e: f"  tool[{e['agent']}] {e['tool']} {json.dumps(e.get('args', {}))[:80]} → {e.get('summary', e.get('error', ''))}",
    "agent_start": lambda e: f"▶ {e['agent']} starts: {e['task'][:100]}",
    "agent_result": lambda e: f"✔ {e['agent']} done via {e['model']} in {e['latency_s']}s, cited {e['cited']}",
    "agent_error": lambda e: f"✘ {e['agent']} FAILED: {e['error']}",
    "turn_complete": lambda e: f"■ turn: {json.dumps(e)}",
}


async def main(queries: list[str], policy: str, forced: str | None, show_tokens: bool) -> None:
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 40}
    for query in queries:
        print(f"\n=== USER: {query}")
        inputs = {"messages": [HumanMessage(query)], "policy": policy, "forced_provider": forced}
        async for mode, payload in graph.astream(inputs, config, stream_mode=["custom", "messages"]):
            if mode == "custom":
                fmt = COMPACT.get(payload["kind"])
                print(fmt(payload) if fmt else payload)
            elif show_tokens and mode == "messages":
                chunk, meta = payload
                if getattr(chunk, "content", None) and isinstance(chunk.content, str) and not chunk.tool_call_chunks:
                    print(chunk.content, end="", flush=True)
        state = (await graph.aget_state(config)).values
        print(f"\n=== FINAL ANSWER ({state['turn_stats']['mode']}):\n{state['final_answer']}")
        cited = {label: src for label, src in state.get("sources", {}).items() if f"[{label}]" in state["final_answer"]}
        for label, src in cited.items():
            page = f"p.{src['page_start']} " if src.get("page_start") else ""
            print(f"  [{label}] {src['title'][:70]} {src.get('section') or ''} {page}{src['url']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--policy", default="balanced", choices=["economy", "balanced", "quality"])
    parser.add_argument("--force", default=None, choices=["openai", "nvidia"])
    parser.add_argument("--tokens", action="store_true", help="also print streamed tokens")
    args = parser.parse_args()
    asyncio.run(main(args.queries, args.policy, args.force, args.tokens))
