"""Offline orchestration tests: a scripted router stands in for the LLMs."""

import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import src.graph.build as build
from src.graph.supervisor import SupervisorDecision
from src.llm.router import CallResult, RouteDecision


def _decision(slot="openai:fast"):
    provider = slot.split(":")[0]
    return RouteDecision(slot, provider, "fake-model", "cost", "test", "test", 0.0)


class ScriptedRouter:
    """Returns the next scripted supervisor decision; synthesizer calls get a fixed merged answer."""

    def __init__(self, decisions: list[SupervisorDecision]):
        self.decisions = list(decisions)
        self.calls = []

    def peek(self, profile):
        return _decision()

    def decide(self, profile):
        return _decision()

    async def ainvoke(self, profile, messages, *, schema=None, tools=None, max_tokens=2048, decision=None):
        self.calls.append(profile.role)
        if schema is SupervisorDecision:
            parsed = self.decisions.pop(0)
            return CallResult(AIMessage(""), _decision(), 0.01, 10, 5, 0.0001, parsed=parsed)
        return CallResult(AIMessage("merged answer [S1]"), _decision("openai:strong"), 0.01, 10, 5, 0.001)


def _run(graph, query: str, thread: str) -> dict:
    events = []

    async def go():
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
        async for mode, payload in graph.astream({"messages": [HumanMessage(query)], "policy": "balanced"}, config,
                                                 stream_mode=["custom", "updates"]):
            if mode == "custom":
                events.append(payload)
        return (await graph.aget_state(config)).values

    state = asyncio.run(go())
    state["_events"] = events
    return state


@pytest.fixture
def fake_agents(monkeypatch):
    async def fake_run_agent(spec, task, **kw):
        if spec.name == "research":
            raise ConnectionError("simulated provider outage")
        return {"agent": spec.name, "task": task, "status": "ok", "output": f"{spec.name} result", "cited": [],
                "provider": "openai", "model": "fake", "slot": "openai:fast", "tools_used": [], "latency_s": 0.1,
                "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.0, "llm_calls": 1}

    monkeypatch.setattr(build, "run_agent", fake_run_agent)


def test_worker_failure_does_not_crash_graph(fake_agents):
    router = ScriptedRouter([
        SupervisorDecision(next="research", task="explain X", complexity="medium", reason="conceptual"),
        SupervisorDecision(next="data", task="count X", complexity="low", reason="research failed; data can cover"),
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="done"),
    ])
    graph = build.build_graph(router=router, checkpointer=InMemorySaver())
    state = _run(graph, "tell me about X", str(uuid.uuid4()))

    assert [r["status"] for r in state["results"]] == ["error", "ok"]
    assert "simulated provider outage" in state["results"][0]["error"]
    assert state["final_answer"] == "merged answer [S1]"  # synthesized despite the failure
    assert state["turn_stats"]["mode"] == "llm synthesis"
    kinds = [e["kind"] for e in state["_events"]]
    assert "agent_error" in kinds and kinds.count("agent_route") == 3


def test_all_workers_fail_gives_explicit_error_answer(fake_agents):
    router = ScriptedRouter([
        SupervisorDecision(next="research", task="explain X", complexity="medium", reason="r"),
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="nothing else can help"),
    ])
    graph = build.build_graph(router=router, checkpointer=InMemorySaver())
    state = _run(graph, "tell me about X", str(uuid.uuid4()))
    assert state["final_answer"].startswith("Sorry, I couldn't complete this request")
    assert "research" in state["final_answer"]


def test_guard_blocks_repeating_the_same_task(fake_agents):
    router = ScriptedRouter([
        SupervisorDecision(next="data", task="count X", complexity="low", reason="r"),
        SupervisorDecision(next="data", task="count X", complexity="low", reason="again?"),
    ])
    graph = build.build_graph(router=router, checkpointer=InMemorySaver())
    state = _run(graph, "how many X", str(uuid.uuid4()))
    assert state["routes"][-1]["decided_by"].startswith("guard: data already ran")
    assert state["final_answer"] == "data result"  # single-agent passthrough, no synthesis call
    assert router.calls == ["supervisor", "supervisor"]


def test_hop_limit_forces_finish(fake_agents, monkeypatch):
    from src.config import get_settings

    monkeypatch.setattr(get_settings(), "max_hops", 2)
    router = ScriptedRouter([
        SupervisorDecision(next="data", task="a", complexity="low", reason="r"),
        SupervisorDecision(next="code", task="b", complexity="low", reason="r"),
    ])
    graph = build.build_graph(router=router, checkpointer=InMemorySaver())
    state = _run(graph, "loop forever", str(uuid.uuid4()))
    assert state["routes"][-1]["decided_by"] == "guard: max_hops"


def test_turn_state_resets_but_conversation_persists(fake_agents):
    router = ScriptedRouter([
        SupervisorDecision(next="data", task="q1", complexity="low", reason="r"),
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="done"),
        SupervisorDecision(next="code", task="q2", complexity="low", reason="r"),
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="done"),
    ])
    graph = build.build_graph(router=router, checkpointer=InMemorySaver())
    thread = str(uuid.uuid4())
    _run(graph, "first", thread)
    state = _run(graph, "second", thread)
    assert [r["agent"] for r in state["results"]] == ["code"]  # turn-level reset
    assert [m.type for m in state["messages"]] == ["human", "ai", "human", "ai"]  # conversation kept


def test_finish_without_answer_on_first_hop_is_retried(fake_agents):
    # Live bug: "what was my first question?" → FINISH with direct_answer=None → "Sorry, no agent was able to run".
    router = ScriptedRouter([
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="answered by history"),
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="answered by history",
                           direct_answer="Your first question was about X."),
    ])
    graph = build.build_graph(router=router, checkpointer=InMemorySaver())
    state = _run(graph, "what was my first question?", str(uuid.uuid4()))
    assert state["final_answer"] == "Your first question was about X."
    assert state["turn_stats"]["mode"] == "direct"


def test_finish_without_answer_twice_falls_back_to_an_agent(fake_agents):
    router = ScriptedRouter([
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="nothing"),
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="still nothing"),
        SupervisorDecision(next="FINISH", task="", complexity="low", reason="done"),
    ])
    graph = build.build_graph(router=router, checkpointer=InMemorySaver())
    state = _run(graph, "tell me about X", str(uuid.uuid4()))
    assert state["routes"][0]["next"] == "research"
    assert state["routes"][0]["decided_by"] == "guard: FINISH without an answer"
