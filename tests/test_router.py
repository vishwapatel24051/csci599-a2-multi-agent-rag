import asyncio

import pytest

from src.llm.providers import ModelSlot
from src.llm.router import LONG_CONTEXT_TOKENS, ModelRouter, ProviderHealth, RateLimiter, TaskProfile, choose

SLOTS = {
    "openai:fast": ModelSlot("openai:fast", "openai", "fast", "gpt-6-luna"),
    "openai:strong": ModelSlot("openai:strong", "openai", "strong", "gpt-6.1-sol"),
    "nvidia:fast": ModelSlot("nvidia:fast", "nvidia", "fast", "nvidia/nemotron-3.5-lightning-30b-a3b"),
    "nvidia:strong": ModelSlot("nvidia:strong", "nvidia", "strong", "nvidia/nemotron-3-ultra-550b-a55b"),
}


def route(**kw):
    return choose(TaskProfile(**kw), ProviderHealth(), SLOTS)


def test_balanced_policy_uses_both_providers():
    """Papa's rule: a router that always picks the same model is not a router."""
    providers = {route(role=r, complexity=c).provider
                 for r in ("supervisor", "research", "code", "data", "summary", "synthesizer")
                 for c in ("low", "medium", "high")}
    assert providers == {"openai", "nvidia"}


@pytest.mark.parametrize("kw, slot, dimension", [
    (dict(role="supervisor"), "openai:fast", "latency"),
    (dict(role="research", complexity="low"), "nvidia:strong", "cost"),
    (dict(role="research", complexity="high"), "nvidia:strong", "capability"),
    (dict(role="code"), "nvidia:strong", "cost"),
    (dict(role="data"), "openai:fast", "capability"),
    (dict(role="summary", est_input_tokens=LONG_CONTEXT_TOKENS + 1), "nvidia:strong", "capability"),
    (dict(role="summary", est_input_tokens=5_000), "openai:fast", "cost"),
    (dict(role="synthesizer", complexity="high"), "nvidia:strong", "capability"),
])
def test_balanced_table(kw, slot, dimension):
    d = route(**kw)
    assert (d.slot, d.dimension) == (slot, dimension)
    assert d.reason and d.rule


def test_policy_changes_the_route():
    assert route(role="research", policy="economy").provider == "nvidia"
    assert route(role="research", policy="quality").slot == "openai:strong"
    assert route(role="supervisor", policy="economy").slot == "nvidia:fast"  # free routing in economy
    assert route(role="supervisor", policy="quality").slot == "openai:fast"


def test_open_circuit_falls_back_to_other_provider_with_reason():
    health = ProviderHealth()
    health.trip("nvidia", "RateLimitError")
    d = choose(TaskProfile(role="code"), health, SLOTS)
    assert d.slot == "openai:fast" and d.dimension == "availability"  # cheapest OpenAI model, not gpt-6.1-sol
    assert "nvidia circuit open (RateLimitError)" in d.skipped[0]
    assert "fallback from nvidia:strong" in d.reason


def test_context_window_guard_skips_small_window():
    # 920k tokens: over 90% of Nemotron's 1M window, within 90% of OpenAI's 1.05M
    d = choose(TaskProfile(role="summary", est_input_tokens=920_000), ProviderHealth(), SLOTS)
    assert d.slot == "openai:fast" and "context window" in d.skipped[0]


def test_forced_provider_for_evaluation():
    d = route(role="research", forced_provider="openai")
    assert (d.slot, d.dimension) == ("openai:strong", "forced")
    assert route(role="data", forced_provider="nvidia").slot == "nvidia:fast"


def test_cost_estimate_is_zero_on_free_tier_and_positive_on_openai():
    assert route(role="code").est_cost_usd == 0
    assert route(role="data").est_cost_usd > 0


def test_rate_limiter_refuses_when_window_is_full():
    limiter = RateLimiter(rpm=2)

    async def run():
        assert await limiter.acquire(max_wait_s=0)
        assert await limiter.acquire(max_wait_s=0)
        return await limiter.acquire(max_wait_s=0)

    assert asyncio.run(run()) is False


class _RateLimited(Exception):
    status_code = 429


class _FakeLLM:
    def __init__(self, fail: bool, name: str):
        self.fail, self.name = fail, name

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        if self.fail:
            raise _RateLimited("429 Too Many Requests")
        return AIMessage(content=f"answer from {self.name}",
                         usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def test_router_fails_over_to_other_provider_and_traces_it(monkeypatch):
    events = []
    router = ModelRouter(emit=lambda kind, data: events.append((kind, data)))
    router.slots = SLOTS
    monkeypatch.setattr(router, "_model", lambda d, tools, max_tokens, schema=None: _FakeLLM(d.provider == "nvidia", d.slot))

    result = asyncio.run(router.ainvoke(TaskProfile(role="code"), []))

    assert result.decision.slot == "openai:fast"
    assert result.message.content == "answer from openai:fast"
    kinds = [k for k, _ in events]
    assert kinds == ["llm_route", "llm_error", "llm_route", "llm_call"]
    assert events[2][1]["dimension"] == "availability"
    assert router.health.is_open("nvidia")


def test_reasoning_mode_only_for_hard_tasks_on_nvidia_strong():
    assert route(role="code", complexity="high").reasoning is True
    assert "thinking mode on" in route(role="code", complexity="high").reason
    assert route(role="code", complexity="medium").reasoning is False
    assert route(role="summary", est_input_tokens=50_000).reasoning is False  # long but not hard
    assert route(role="research", complexity="high").reasoning is True  # hard research thinks on Nemotron
    assert route(role="research", complexity="medium").reasoning is False


def test_fallback_is_capability_aware():
    health = ProviderHealth()
    health.trip("openai", "APIConnectionError")
    # tool-using role on openai:fast must not fall back to nvidia:fast (0% tool accuracy in benchmark)
    assert choose(TaskProfile(role="data"), health, SLOTS).slot == "nvidia:strong"
    # the supervisor has no tools: nvidia:fast is a fine fallback for routing
    assert choose(TaskProfile(role="supervisor"), health, SLOTS).slot == "nvidia:fast"


def test_summary_reroutes_when_context_grows():
    short = route(role="summary", est_input_tokens=800)
    long = route(role="summary", est_input_tokens=42_000)
    assert (short.slot, long.slot) == ("openai:fast", "nvidia:strong")


def test_dropped_connections_are_retryable():
    import httpx
    from src.llm.router import _is_retryable

    assert _is_retryable(httpx.ReadError(""))[0]
    assert _is_retryable(TimeoutError("x timed out"))[0]
    assert not _is_retryable(ValueError("bad schema"))[0]


def test_nvidia_structured_output_without_include_raw(monkeypatch):
    # ChatNVIDIA can't return the raw message with structured output; the router must accept the parsed object
    # alone (estimating usage) instead of failing every NVIDIA supervisor call (bug found in evaluation).
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel

    class Decision(BaseModel):
        next: str

    class _ParsedOnly:
        async def ainvoke(self, messages):
            return Decision(next="research")

    router = ModelRouter(emit=lambda kind, data: None)
    router.slots = SLOTS
    monkeypatch.setattr(router, "_model", lambda d, tools, max_tokens, schema=None: _ParsedOnly())

    result = asyncio.run(router.ainvoke(TaskProfile(role="supervisor", forced_provider="nvidia"),
                                        [HumanMessage("route this")], schema=Decision))

    assert result.parsed == Decision(next="research")
    assert result.decision.provider == "nvidia" and result.cost_usd == 0
    assert result.input_tokens > 0 and result.output_tokens > 0


def test_empty_completion_fails_over_to_other_provider(monkeypatch):
    # NVIDIA NIM sometimes answers 200 OK with nothing in it; that must count as a provider failure.
    from langchain_core.messages import AIMessage

    class _Empty:
        async def ainvoke(self, messages):
            return AIMessage(content="", usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    events = []
    router = ModelRouter(emit=lambda kind, data: events.append((kind, data)))
    router.slots = SLOTS
    monkeypatch.setattr(router, "_model", lambda d, tools, max_tokens, schema=None:
                        _Empty() if d.provider == "nvidia" else _FakeLLM(False, d.slot))

    result = asyncio.run(router.ainvoke(TaskProfile(role="research"), []))

    assert result.decision.provider == "openai"
    assert [k for k, _ in events] == ["llm_route", "llm_error", "llm_route", "llm_call"]
    assert "empty completion" in events[1][1]["error"]


def test_balanced_policy_never_selects_the_expensive_openai_model():
    # gpt-6.1-sol only runs when a user picks the quality policy, never by default or on an NVIDIA outage.
    nvidia_down = ProviderHealth()
    nvidia_down.trip("nvidia", "TimeoutError")
    for health in (ProviderHealth(), nvidia_down):
        for role in ("supervisor", "research", "code", "data", "summary", "synthesizer"):
            for complexity in ("low", "medium", "high"):
                for tokens in (2_000, 50_000):
                    d = choose(TaskProfile(role=role, complexity=complexity, est_input_tokens=tokens), health, SLOTS)
                    assert d.slot != "openai:strong", (role, complexity, tokens, d)
