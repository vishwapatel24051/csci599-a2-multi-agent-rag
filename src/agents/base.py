"""The worker-agent loop shared by all specialists: a bounded ReAct tool loop.

What differs between agents is their AgentSpec: system prompt, tool set, and limits.
Every LLM call goes through the model router, which re-checks the route as the context
grows (e.g. a Summary agent that fetches a 40k-token paper crosses the long-context
threshold and is re-routed, with a trace event explaining why).
"""

import re
import time
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.agents.tools import AgentContext
from src.llm.router import ModelRouter, TaskProfile, estimate_tokens
from src.rag.retriever import SourceRegistry, normalize_citations
from src.trace import emit

_CITATION = re.compile(r"\[((?:S|W)\d+)\]")
BUDGET_EXHAUSTED = "You have used your tool budget. Answer now from the information above, with citations."
SOURCE_TOOLS = {"search_corpus", "search_code_examples", "web_search", "web_extract", "fetch_document"}


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str  # what the supervisor is told this agent is for
    system_prompt: str
    make_tools: Callable[[AgentContext], list[BaseTool]]
    max_tool_rounds: int = 4
    max_tokens: int = 2048


def _cited_labels(text: str, registry: SourceRegistry) -> list[str]:
    return sorted({m for m in _CITATION.findall(text) if m in registry.sources}, key=lambda l: (l[0], int(l[1:])))


async def run_agent(spec: AgentSpec, task: str, *, router: ModelRouter, registry: SourceRegistry,
                    complexity: str = "medium", policy: str = "balanced", forced_provider: str | None = None) -> dict:
    """Run one agent on one task. Returns an AgentResult dict (raises on failure; the graph node catches)."""
    t0 = time.perf_counter()
    ctx = AgentContext(spec.name, registry)
    tools = spec.make_tools(ctx)
    by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [SystemMessage(spec.system_prompt), HumanMessage(task)]
    totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "llm_calls": 0}
    decision, final, tools_used = None, "", []
    sources_before = set(registry.sources)

    round_no = 0
    while True:
        profile = TaskProfile(role=spec.name, complexity=complexity, est_input_tokens=estimate_tokens(messages),
                              policy=policy, forced_provider=forced_provider)
        if decision is None or router.peek(profile).slot != decision.slot:
            decision = router.decide(profile)
        last_round = round_no >= spec.max_tool_rounds
        if last_round:
            messages.append(HumanMessage(BUDGET_EXHAUSTED))

        # Final round: no tools bound, so the model must answer in text. (Seen in testing: on its last
        # round Nemotron put the finished code into yet another check_python_syntax call instead.)
        result = await router.ainvoke(profile, messages, tools=None if last_round else tools, decision=decision,
                                      max_tokens=spec.max_tokens)
        decision = result.decision  # may now be a fallback on the other provider
        totals["input_tokens"] += result.input_tokens
        totals["output_tokens"] += result.output_tokens
        totals["cost_usd"] += result.cost_usd
        totals["llm_calls"] += 1

        msg = result.message
        if not msg.tool_calls or last_round:
            final = normalize_citations(msg.text.strip())
            searched = any(name in SOURCE_TOOLS for name in tools_used)
            if final and not last_round and searched and not _cited_labels(final, registry):
                # A reply that cites nothing after searching is a preamble, not an answer (seen live with
                # Nemotron's thinking mode: "Let me search for more specific announcements…" and no tool call).
                messages.append(msg)
                round_no = spec.max_tool_rounds
                continue
            if final or last_round:
                break
            round_no = spec.max_tool_rounds  # empty answer and no tool calls: go straight to the forced answer
            continue
        messages.append(msg)
        for call in msg.tool_calls:
            tools_used.append(call["name"])
            tool = by_name.get(call["name"])
            try:
                output = await tool.ainvoke(call["args"]) if tool else f"Unknown tool {call['name']!r}."
            except Exception as e:  # noqa: BLE001 — a failed tool is information for the model, not a crash
                output = f"Tool {call['name']} failed: {e}"
                emit("tool_call", {"agent": spec.name, "tool": call["name"], "ok": False, "error": str(e)[:200]})
            messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        round_no += 1

    if not final:
        raise RuntimeError(f"{spec.name} produced no answer")
    consulted = [label for label in registry.sources if label not in sources_before]
    if consulted and not _cited_labels(final, registry):
        # Rare (1 of ~40 live answers): still uncited after the forced round. List what was read, without
        # claiming which sentence came from where, so the answer still links back to its sources.
        final += "\n\nSources consulted: " + " ".join(f"[{label}]" for label in consulted[:8])

    return {
        "agent": spec.name,
        "task": task,
        "status": "ok",
        "output": final,
        "cited": _cited_labels(final, registry),
        "provider": decision.provider,
        "model": decision.model,
        "slot": decision.slot,
        "tools_used": tools_used,
        "latency_s": round(time.perf_counter() - t0, 3),
        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in totals.items()},
    }
