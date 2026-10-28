"""Agent loop edge cases seen in live runs, reproduced with a scripted router."""

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agents.base import AgentSpec, run_agent
from src.llm.router import CallResult, RouteDecision
from src.rag.retriever import SourceRegistry

DECISION = RouteDecision("nvidia:strong", "nvidia", "fake", "cost", "test", "test", 0.0)


@tool
def noop(x: str) -> str:
    """Do nothing."""
    return "ok"


class Router:
    def __init__(self, replies):
        self.replies, self.tools_seen = list(replies), []

    def peek(self, profile):
        return DECISION

    def decide(self, profile):
        return DECISION

    async def ainvoke(self, profile, messages, *, tools=None, decision=None, max_tokens=0, schema=None):
        self.tools_seen.append(bool(tools))
        return CallResult(self.replies.pop(0), DECISION, 0.0, 1, 1, 0.0)


def _call(i):
    return AIMessage("", tool_calls=[{"name": "noop", "args": {"x": str(i)}, "id": f"c{i}", "type": "tool_call"}])


SPEC = AgentSpec("code", "test", "system", lambda ctx: [noop], max_tool_rounds=2)


def test_last_round_has_no_tools_and_returns_text():
    router = Router([_call(1), _call(2), AIMessage("final code")])
    result = asyncio.run(run_agent(SPEC, "task", router=router, registry=SourceRegistry()))
    assert result["output"] == "final code"
    assert router.tools_seen == [True, True, False]  # tools withheld on the budget-exhausted round
    assert result["tools_used"] == ["noop", "noop"]


def test_empty_answer_jumps_to_forced_answer_round():
    router = Router([AIMessage(""), AIMessage("recovered answer")])
    result = asyncio.run(run_agent(SPEC, "task", router=router, registry=SourceRegistry()))
    assert result["output"] == "recovered answer"
    assert router.tools_seen == [True, False]


def test_uncited_preamble_after_searching_is_not_accepted_as_the_answer():
    # Live failure: after 3 web searches Nemotron replied "Let me search for more…" with no tool call.
    def make_tools(ctx):
        @tool
        def web_search(query: str) -> str:
            """Search the web."""
            src = ctx.registry.add_web("https://example.com/news", "News", "Agent X launched on Monday.")
            return f"[{src.label}] News\nAgent X launched on Monday."
        return [web_search]

    spec = AgentSpec("research", "test", "system", make_tools, max_tool_rounds=4)
    search = AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "c1", "type": "tool_call"}])
    router = Router([search, AIMessage("Let me search for more specific announcements."),
                     AIMessage("Agent X launched on Monday [W1].")])
    result = asyncio.run(run_agent(spec, "task", router=router, registry=SourceRegistry()))
    assert result["output"] == "Agent X launched on Monday [W1]."
    assert result["cited"] == ["W1"]
    assert router.tools_seen == [True, True, False]  # the preamble triggers the forced, tool-less answer round


def test_uncited_answer_without_searching_is_accepted():
    router = Router([AIMessage("Hello! I can help with agent questions.")])
    result = asyncio.run(run_agent(SPEC, "task", router=router, registry=SourceRegistry()))
    assert result["output"].startswith("Hello") and router.tools_seen == [True]


def test_answer_still_uncited_after_forced_round_lists_sources_consulted():
    def make_tools(ctx):
        @tool
        def search_corpus(query: str) -> str:
            """Search the corpus."""
            src = ctx.registry.add_web("https://example.com/p", "Paper", "Self-RAG retrieves on demand.")
            return f"[{src.label}] Paper\nSelf-RAG retrieves on demand."
        return [search_corpus]

    spec = AgentSpec("research", "test", "system", make_tools, max_tool_rounds=4)
    search = AIMessage("", tool_calls=[{"name": "search_corpus", "args": {"query": "q"}, "id": "c1", "type": "tool_call"}])
    router = Router([search, AIMessage("Self-RAG retrieves on demand."), AIMessage("Self-RAG retrieves on demand.")])
    result = asyncio.run(run_agent(spec, "task", router=router, registry=SourceRegistry()))
    assert result["output"].endswith("Sources consulted: [W1]")
    assert result["cited"] == ["W1"]
