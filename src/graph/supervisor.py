"""Supervisor node: decides the next agent (or FINISH) from inspectable state.

The LLM proposes; deterministic guards dispose:
  * hop limit reached                      → FINISH
  * same (agent, task) already ran         → FINISH (no loops)
  * agent already failed twice this turn   → FINISH
  * supervisor LLM call itself failed      → research on the first hop, FINISH afterwards
Each decision is appended to `state["routes"]` and emitted as an `agent_route` trace event,
including who decided it (`llm` or `guard: <why>`).
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.specialists import AGENTS
from src.config import get_settings
from src.graph.state import GraphState, RouteRecord
from src.llm.router import ModelRouter, TaskProfile
from src.trace import emit

AgentName = Literal["research", "code", "data", "summary", "FINISH"]


class SupervisorDecision(BaseModel):
    next: AgentName = Field(description="The agent to run next, or FINISH when the request is fully answered.")
    task: str = Field(description="Self-contained instruction for that agent (resolve follow-ups and pronouns "
                                  "from the conversation). Empty when FINISH.")
    complexity: Literal["low", "medium", "high"] = Field(
        description="low: single fact or lookup; medium: explanation from one or two sources; "
                    "high: comparison or synthesis across several sources, or multi-step reasoning.")
    reason: str = Field(description="One sentence explaining the choice; shown in the trace log.")
    direct_answer: str | None = Field(
        default=None, description="Required when you FINISH before any agent has run this turn: the complete "
                                  "reply to the user (greetings, questions about this assistant, or questions "
                                  "answered by the conversation so far, e.g. 'what did I ask first?'). "
                                  "Otherwise null.")


def _system_prompt() -> str:
    agents = "\n".join(f"- {name}: {spec.description}" for name, spec in AGENTS.items())
    return (
        "You are the supervisor of a team of specialist agents. Each turn you choose the single next step.\n"
        f"Agents:\n{agents}\n"
        "- FINISH: the results so far fully answer the user's request, or nothing more can be done.\n\n"
        "Rules:\n"
        "1. One agent per step, with a self-contained task.\n"
        "2. Use several agents only when the request has several genuine parts, e.g. 'find the most-cited RAG "
        "paper and summarize it' → data, then summary with the paper it found.\n"
        "3. Never give the same task to the same agent twice. If an agent failed, route to another agent that "
        "can cover the task, or FINISH.\n"
        "4. Questions about current events, recent releases or 'latest' versions go to research (it has web search).\n"
        "5. A result marked 'ok, COMPLETE' is the agent's full answer even though you only see a preview; do not "
        "re-run agents to 'finish' a preview.\n"
        "6. For greetings, questions about this assistant, or questions the conversation so far already "
        "answers (e.g. 'what did I ask first?'), FINISH with the reply in direct_answer.\n"
        "7. A specific question about a paper or docs page (a result, a number, how its method works) goes to "
        "research, even when it names the document: research retrieves the relevant passages. Use summary only "
        "when the user asks to summarize or give an overview of a whole document.\n"
        "8. FINISH as soon as the results cover every part of the user's request. Agent answers reach the user "
        "as written, so never route to another agent to summarize, shorten, reformat, verify or illustrate a "
        "result, and never add work the user did not ask for (code, charts, extra summaries)."
    )


def _history(state: GraphState, max_messages: int = 6, max_chars: int = 600) -> str:
    previous = state.get("messages", [])[:-1][-max_messages:]
    lines = [f"{m.type}: {m.text[:max_chars]}" for m in previous]
    return "\n".join(lines) if lines else "(none)"


def _results_so_far(state: GraphState, max_chars: int = 6000) -> str:
    """Each result in full up to max_chars (answers run median ~1.7k, p90 ~3k chars), so the supervisor can see
    that a request is covered. An 800-char preview made it re-run agents or add a summary to "finish" answers
    whose key facts sat past the cut (only 9 of 24 replayed post-research decisions were FINISH)."""
    lines = []
    for r in state.get("results", []):
        if r["status"] != "ok":
            lines.append(f"[{r['agent']}] task: {r['task']}\nstatus: FAILED: {r.get('error', '')}")
            continue
        out = r["output"]
        status = (f"status: ok, COMPLETE ({len(out):,} chars; showing a preview of the first {max_chars})"
                  if len(out) > max_chars else "status: ok, complete")
        lines.append(f"[{r['agent']}] task: {r['task']}\n{status}\n{out[:max_chars]}")
    return "\n\n".join(lines) if lines else "(no agent has run yet this turn)"


def _guard(state: GraphState, d: SupervisorDecision) -> str | None:
    results = state.get("results", [])
    if d.next == "FINISH":
        return None
    if any(r["agent"] == d.next and r["task"].strip().lower() == d.task.strip().lower() for r in results):
        return f"guard: {d.next} already ran this exact task"
    if sum(1 for r in results if r["agent"] == d.next and r["status"] == "error") >= 2:
        return f"guard: {d.next} failed twice this turn"
    return None


async def supervise(state: GraphState, router: ModelRouter) -> RouteRecord:
    s = get_settings()
    hop = state.get("hops", 0)
    record: RouteRecord

    if hop >= s.max_hops:
        record = {"hop": hop, "next": "FINISH", "task": "", "complexity": "medium",
                  "reason": f"hop limit ({s.max_hops}) reached", "decided_by": "guard: max_hops"}
    else:
        messages = [
            SystemMessage(_system_prompt()),
            HumanMessage(f"Conversation so far:\n{_history(state)}\n\n"
                         f"Current user request:\n{state['query']}\n\n"
                         f"Results so far this turn:\n{_results_so_far(state)}"),
        ]
        profile = TaskProfile(role="supervisor", policy=state.get("policy", "balanced"),
                              forced_provider=state.get("forced_provider"))
        try:
            # The cap includes gpt-6-luna's hidden reasoning tokens: decisions run median ~110, p90 ~330 tokens,
            # and 600 hit the length limit on ~6% of hops in evaluation.
            result = await router.ainvoke(profile, messages, schema=SupervisorDecision, max_tokens=2000)
            d: SupervisorDecision = result.parsed
            cost, decided_by = result.cost_usd, "llm"
            first = not state.get("results")
            if first and d.next == "FINISH" and not d.direct_answer:
                # FINISH with nothing to show: seen live for "what was my first question?" (2 of 5 tries).
                # Ask once more; if it still has no answer, run research rather than return an empty turn.
                messages.append(HumanMessage("You chose FINISH before any agent ran but left direct_answer empty, "
                                             "so the user would get no reply. Put the full reply in "
                                             "direct_answer, or choose an agent."))
                result = await router.ainvoke(profile, messages, schema=SupervisorDecision, max_tokens=2000)
                d, cost = result.parsed, cost + result.cost_usd
                if d.next == "FINISH" and not d.direct_answer:
                    d = d.model_copy(update={"next": "research", "task": state["query"]})
                    decided_by = "guard: FINISH without an answer"
            guard = _guard(state, d)
            record = {"hop": hop, "next": "FINISH" if guard else d.next, "task": d.task,
                      "complexity": d.complexity, "reason": d.reason, "decided_by": guard or decided_by,
                      "provider": result.decision.provider, "model": result.decision.model,
                      "latency_s": round(result.latency_s, 3), "cost_usd": round(cost, 6),
                      "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
            if d.direct_answer and d.next == "FINISH" and first:
                record["direct_answer"] = d.direct_answer
        except Exception as e:  # noqa: BLE001 — the supervisor failing must not crash the turn
            first = not state.get("results")
            record = {"hop": hop, "next": "research" if first else "FINISH", "task": state["query"] if first else "",
                      "complexity": "medium", "reason": f"supervisor LLM failed ({type(e).__name__}); "
                      + ("defaulting to research" if first else "finishing with results so far"),
                      "decided_by": "guard: supervisor_error"}

    emit("agent_route", dict(record))
    return record
