"""The orchestration graph (Supervisor pattern).

    START → start_turn → supervisor ─┬─► research ─┐
                             ▲       ├─► code     ─┤
                             │       ├─► data     ─┤  (every worker returns to the supervisor)
                             │       ├─► summary  ─┤
                             └───────┴─────────────┘
                                     └─► synthesize → END   (when the supervisor says FINISH)

Routing is state-driven: the supervisor writes a RouteRecord to `state["routes"]` and returns
`Command(goto=record["next"])`. Workers never crash the graph: any exception becomes an
AgentResult with status="error" that the supervisor sees on its next hop.
"""

import asyncio
import time
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agents.base import AgentSpec, run_agent
from src.agents.specialists import AGENTS
from src.config import get_settings
from src.graph.state import GraphState
from src.graph.supervisor import supervise
from src.llm.router import ModelRouter, TaskProfile
from src.rag.retriever import SourceRegistry, normalize_citations
from src.trace import emit

Worker = Literal["research", "code", "data", "summary"]

SYNTH_PROMPT = (
    "You combine the results of several specialist agents into one answer to the user's request.\n"
    "Keep every citation label exactly as it appears in the results ([S#] for corpus sources, [W#] for web "
    "sources) and attach it to the same claims; never add a label that is not in the results. "
    "If results disagree, say so. If an agent failed, answer with what is available and state what is missing. "
    "Be concise and well structured; keep code blocks and SQL as given."
)


def start_turn(state: GraphState) -> dict:
    """Reset turn-level fields (reducers treat None as reset)."""
    return {"query": state["messages"][-1].text, "routes": None, "results": None, "sources": None, "hops": 0,
            "final_answer": "", "turn_stats": {}}


def make_supervisor_node(router: ModelRouter):
    async def supervisor(state: GraphState) -> Command[Literal["research", "code", "data", "summary", "synthesize"]]:
        record = await supervise(state, router)
        goto = "synthesize" if record["next"] == "FINISH" else record["next"]
        return Command(goto=goto, update={"routes": [record], "hops": state.get("hops", 0) + 1})

    return supervisor


def make_worker_node(spec: AgentSpec, router: ModelRouter):
    async def worker(state: GraphState) -> Command[Literal["supervisor"]]:
        route = state["routes"][-1]
        registry = SourceRegistry(state.get("sources"))
        emit("agent_start", {"agent": spec.name, "task": route["task"], "complexity": route["complexity"]})
        try:
            result = await asyncio.wait_for(
                run_agent(spec, route["task"], router=router, registry=registry, complexity=route["complexity"],
                          policy=state.get("policy", "balanced"), forced_provider=state.get("forced_provider")),
                timeout=get_settings().worker_timeout_s,
            )
            emit("agent_result", {k: result[k] for k in ("agent", "status", "provider", "model", "cited", "tools_used",
                                                         "latency_s", "cost_usd")})
        except Exception as e:  # noqa: BLE001 — isolate the failure; the supervisor decides what next
            error = f"{type(e).__name__}: {e}"[:300] if str(e) else type(e).__name__
            result = {"agent": spec.name, "task": route["task"], "status": "error", "error": error, "output": "",
                      "cited": [], "latency_s": 0.0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
            emit("agent_error", {"agent": spec.name, "error": error})
        return Command(goto="supervisor", update={"results": [result], "sources": registry.as_state()})

    worker.__name__ = spec.name
    return worker


def make_synthesize_node(router: ModelRouter):
    async def synthesize(state: GraphState) -> dict:
        t0 = time.perf_counter()
        results = state.get("results", [])
        ok = [r for r in results if r["status"] == "ok"]
        failed = [r for r in results if r["status"] != "ok"]
        last_route = state["routes"][-1] if state.get("routes") else {}
        synth_cost, synth_model = 0.0, None

        if last_route.get("direct_answer"):
            answer, mode = last_route["direct_answer"], "direct"
        elif len(ok) == 1 and not failed:
            answer, mode = ok[0]["output"], "single-agent passthrough"
        elif ok:
            complexity = "high" if len(ok) > 2 or any(r.get("complexity") == "high" for r in state["routes"]) \
                else "medium"
            body = "\n\n".join(f"## {r['agent']} agent (task: {r['task']})\n{r['output']}" for r in ok)
            if failed:
                body += "\n\n## Failed agents\n" + "\n".join(f"- {r['agent']}: {r['error']}" for r in failed)
            profile = TaskProfile(role="synthesizer", complexity=complexity, policy=state.get("policy", "balanced"),
                                  forced_provider=state.get("forced_provider"))
            result = await router.ainvoke(
                profile, [SystemMessage(SYNTH_PROMPT),
                          HumanMessage(f"User request:\n{state['query']}\n\nAgent results:\n{body}")])
            answer, mode = normalize_citations(result.message.text.strip()), "llm synthesis"
            synth_cost, synth_model = result.cost_usd, result.decision.model
        else:
            reasons = "; ".join(f"{r['agent']}: {r['error']}" for r in failed) or "no agent was able to run"
            answer, mode = f"Sorry, I couldn't complete this request ({reasons}). Please try again.", "error"

        routes = state.get("routes", [])
        stats = {
            "mode": mode,
            "agents": [r["agent"] for r in results],
            "hops": state.get("hops", 0),
            "llm_cost_usd": round(sum(r.get("cost_usd", 0) for r in results) + synth_cost
                                  + sum(r.get("cost_usd", 0) for r in routes), 6),
            "input_tokens": sum(r.get("input_tokens", 0) for r in results) + sum(r.get("input_tokens", 0) for r in routes),
            "output_tokens": sum(r.get("output_tokens", 0) for r in results) + sum(r.get("output_tokens", 0) for r in routes),
            "providers": sorted({r["provider"] for r in results if r.get("provider")}
                                | {r["provider"] for r in routes if r.get("provider")}),
            "synth_model": synth_model,
            "synth_latency_s": round(time.perf_counter() - t0, 3) if mode == "llm synthesis" else 0.0,
        }
        emit("turn_complete", stats)
        return {"final_answer": answer, "messages": [AIMessage(answer)], "turn_stats": stats}

    return synthesize


def build_graph(router: ModelRouter | None = None, checkpointer=None):
    router = router or ModelRouter(emit=emit)
    builder = StateGraph(GraphState)
    builder.add_node("start_turn", start_turn)
    builder.add_node("supervisor", make_supervisor_node(router))
    for name, spec in AGENTS.items():
        builder.add_node(name, make_worker_node(spec, router))
    builder.add_node("synthesize", make_synthesize_node(router))
    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "supervisor")
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer)
