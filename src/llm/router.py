"""Model router: decides which provider/model runs each LLM call, and says why.

Two layers:
  * `choose(profile)` is a pure function over a TaskProfile: a policy table keyed by
    (policy, role, complexity, input size), filtered by provider health and context
    windows. Same input, same decision, so the routing is testable and explainable.
  * `ModelRouter.ainvoke / astream` execute a decision. On a retryable provider failure
    (429, 5xx, timeout) they open that provider's circuit breaker, re-route, and retry
    once on the other provider, emitting a trace event for each step.

Agent routing (which agent works next) is the supervisor's job; this module only
decides which model computes a call that has already been assigned.
"""

import asyncio
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import AsyncIterator, Callable, Literal

from langchain_core.messages import AIMessage, BaseMessage

from src.config import get_settings
from src.llm.pricing import cost_usd
from src.llm.providers import ModelSlot, Provider, make_chat_model, model_slots

Role = Literal["supervisor", "research", "code", "data", "summary", "synthesizer"]
Complexity = Literal["low", "medium", "high"]
Policy = Literal["economy", "balanced", "quality"]
Dimension = Literal["cost", "capability", "latency", "availability", "forced"]

LONG_CONTEXT_TOKENS = 30_000  # above this, prefer the 1M-context free model for summaries
EXPECTED_OUTPUT_TOKENS = {"supervisor": 150, "research": 600, "code": 900, "data": 300, "summary": 800, "synthesizer": 700}


@dataclass(frozen=True)
class TaskProfile:
    role: Role
    complexity: Complexity = "medium"
    est_input_tokens: int = 2_000
    policy: Policy = "balanced"
    forced_provider: Provider | None = None  # evaluation shootout only


@dataclass
class RouteDecision:
    slot: str
    provider: Provider
    model: str
    dimension: Dimension
    reason: str
    rule: str
    est_cost_usd: float
    skipped: list[str] = field(default_factory=list)  # slots ruled out, with why
    reasoning: bool = False  # NVIDIA thinking mode: only when the task is complex enough to pay for it

    def as_trace(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- policy table
# Each rule: (id, predicate, primary slot, dimension, reason). First match wins.
# Reasons cite the Phase 0 benchmark (data/bench/bench_2026-10-26_*.json, medians of 3 runs,
# NVIDIA thinking off): openai:fast routing 100% @ 1.7 s; nvidia:strong tools 100% @ 1.6 s and
# RAG TTFT 1.5 s; nvidia:fast routing 92% @ 0.7 s but tool selection 0% (answers without calling).
Rule = tuple[str, Callable[[TaskProfile], bool], str, Dimension, str]

TOOL_ROLES = {"research", "code", "data", "summary"}

_SUPERVISOR: Rule = (
    "supervisor", lambda p: p.role == "supervisor", "openai:fast", "latency",
    "runs on every hop: fastest model with 100% routing accuracy in the benchmark")

_BALANCED: list[Rule] = [
    _SUPERVISOR,
    ("summary-long", lambda p: p.role == "summary" and p.est_input_tokens > LONG_CONTEXT_TOKENS, "nvidia:strong",
     "capability", f"input > {LONG_CONTEXT_TOKENS // 1000}k tokens: free 1M-context model (OpenAI cost scales with input)"),
    ("summary-short", lambda p: p.role == "summary", "openai:fast", "cost",
     "short input: cheapest reliable tool-caller to locate and fetch the document"),
    ("research-high", lambda p: p.role == "research" and p.complexity == "high", "nvidia:strong", "capability",
     "multi-hop question combining sources: strong free model with its thinking mode on"),
    ("research", lambda p: p.role == "research", "nvidia:strong", "cost",
     "grounded lookup: free model with 100% tool accuracy, 1.5 s TTFT in benchmark"),
    ("code", lambda p: p.role == "code", "nvidia:strong", "cost",
     "code generation: strong free model; OpenAI only if NVIDIA is unavailable"),
    ("data", lambda p: p.role == "data", "openai:fast", "capability",
     "text-to-SQL needs precise, short structured output; cheapest reliable model"),
    ("synth-high", lambda p: p.role == "synthesizer" and p.complexity == "high", "nvidia:strong", "capability",
     "merging several agents' results on a hard question: strong free model with its thinking mode on"),
    ("synth", lambda p: p.role == "synthesizer", "nvidia:strong", "cost",
     "merging results on a routine question: strong free model"),
]

_ECONOMY: list[Rule] = [
    ("economy-supervisor", lambda p: p.role == "supervisor", "nvidia:fast", "cost",
     "economy policy: free fast model for routing (92% accuracy @ 0.7 s in benchmark)"),
    ("economy", lambda p: True, "nvidia:strong", "cost", "economy policy: free strong model for all agent work"),
]

_QUALITY: list[Rule] = [
    _SUPERVISOR,
    ("quality", lambda p: True, "openai:strong", "capability", "quality policy: strongest model for all agent work"),
]

POLICIES: dict[str, list[Rule]] = {"balanced": _BALANCED, "economy": _ECONOMY, "quality": _QUALITY}


def _fallback(slot_key: str, profile: TaskProfile) -> str:
    """The other provider. Falling back to OpenAI uses gpt-6-luna (100% tool accuracy in the benchmark, 20x
    cheaper than gpt-6.1-sol), so an NVIDIA outage can't drain a prepaid OpenAI credit; gpt-6.1-sol runs only
    when the quality policy asks for it. Tool-using roles never fall back to nvidia:fast."""
    provider, tier = slot_key.split(":")
    if provider == "nvidia":
        return "openai:fast"
    if tier == "fast" and profile.role in TOOL_ROLES:
        return "nvidia:strong"
    return f"nvidia:{tier}"


def _estimate_cost(slot: ModelSlot, profile: TaskProfile) -> float:
    return cost_usd(slot.model_id, profile.est_input_tokens, EXPECTED_OUTPUT_TOKENS[profile.role])


# ---------------------------------------------------------------- provider health

class ProviderHealth:
    """Circuit breaker per provider: open for `cooldown_s` after a retryable failure."""

    def __init__(self, cooldown_s: float = 60.0):
        self.cooldown_s = cooldown_s
        self._open_until: dict[str, float] = {}
        self._why: dict[str, str] = {}

    def trip(self, provider: str, why: str, retry_after_s: float | None = None) -> None:
        self._open_until[provider] = time.monotonic() + (retry_after_s or self.cooldown_s)
        self._why[provider] = why

    def is_open(self, provider: str) -> bool:
        return time.monotonic() < self._open_until.get(provider, 0.0)

    def why(self, provider: str) -> str:
        return self._why.get(provider, "")

    def reset(self) -> None:
        self._open_until.clear()
        self._why.clear()


class RateLimiter:
    """Sliding-window limiter (requests per minute) for the NVIDIA free tier."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    def wait_time(self) -> float:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        return 0.0 if len(self._calls) < self.rpm else 60 - (now - self._calls[0])

    async def acquire(self, max_wait_s: float) -> bool:
        async with self._lock:
            wait = self.wait_time()
            if wait > max_wait_s:
                return False
            if wait > 0:
                await asyncio.sleep(wait)
            self._calls.append(time.monotonic())
            return True


def _wants_reasoning(slot: ModelSlot, profile: TaskProfile) -> bool:
    """Nemotron thinking costs 5-10x the output tokens and seconds of TTFT; pay it only for hard, strong-tier work."""
    return slot.provider == "nvidia" and slot.tier == "strong" and profile.complexity == "high" \
        and profile.role in ("research", "code", "synthesizer")


def choose(profile: TaskProfile, health: ProviderHealth | None = None,
           slots: dict[str, ModelSlot] | None = None) -> RouteDecision:
    slots = slots or model_slots()
    health = health or ProviderHealth()

    if profile.forced_provider:
        tier = "fast" if profile.role in ("supervisor", "data") else "strong"
        slot = slots[f"{profile.forced_provider}:{tier}"]
        return RouteDecision(slot.key, slot.provider, slot.model_id, "forced",
                             f"evaluation run forced provider={profile.forced_provider}", "forced",
                             _estimate_cost(slot, profile))

    rules = POLICIES[profile.policy]
    rule_id, _, primary, dimension, reason = next(r for r in rules if r[1](profile))
    skipped = []
    for candidate in (primary, _fallback(primary, profile)):
        slot = slots[candidate]
        if health.is_open(slot.provider):
            skipped.append(f"{candidate}: {slot.provider} circuit open ({health.why(slot.provider)})")
            continue
        if slot.context_window and profile.est_input_tokens > 0.9 * slot.context_window:
            skipped.append(f"{candidate}: input exceeds context window ({slot.context_window:,})")
            continue
        if candidate != primary:
            dimension, reason = "availability", f"fallback from {primary}: {skipped[-1].split(': ', 1)[1]}"
        reasoning = _wants_reasoning(slot, profile)
        if reasoning:
            reason += "; thinking mode on (high complexity)"
        return RouteDecision(slot.key, slot.provider, slot.model_id, dimension, reason, rule_id,
                             _estimate_cost(slot, profile), skipped, reasoning)

    # both providers unhealthy: try the primary anyway rather than fail closed
    slot = slots[primary]
    return RouteDecision(slot.key, slot.provider, slot.model_id, "availability",
                         "all providers degraded; trying primary", rule_id, _estimate_cost(slot, profile), skipped)


# ---------------------------------------------------------------- execution

_TRANSIENT_NAMES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
                    "ReadError", "WriteError", "ConnectError", "RemoteProtocolError", "ReadTimeout",
                    "ConnectTimeout", "PoolTimeout", "TimeoutError", "ServiceUnavailableError",
                    "EmptyCompletionError"}


class EmptyCompletionError(RuntimeError):
    """The provider answered 200 OK with no text, no tool calls and no output tokens.

    Seen from NVIDIA NIM in evaluation (~0.3 s, 0 tokens in/out, twice in a row for the same agent). Raising it
    lets the router fail over like any other transient provider error instead of the agent failing.
    """


def _is_empty(msg: AIMessage) -> bool:
    return not msg.text and not msg.tool_calls and not (msg.usage_metadata or {}).get("output_tokens")


def _is_retryable(e: Exception) -> tuple[bool, float | None]:
    """Rate limits, 5xx, timeouts and dropped connections are worth retrying on the other provider."""
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    names = {cls.__name__ for cls in type(e).__mro__}
    text = repr(e).lower()
    transient = (status in (429, 500, 502, 503, 504) or names & _TRANSIENT_NAMES
                 or any(s in text for s in ("429", "rate limit", "timed out", "503", "502", "overloaded")))
    if not transient:
        return False, None
    retry_after = None
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    if "retry-after" in headers:
        try:
            retry_after = float(headers["retry-after"])
        except ValueError:
            pass
    return True, retry_after


@dataclass
class CallResult:
    message: AIMessage
    decision: RouteDecision
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ttft_s: float | None = None
    parsed: object | None = None  # structured-output result, when a schema was requested


Emit = Callable[[str, dict], None]


class ModelRouter:
    def __init__(self, emit: Emit | None = None):
        s = get_settings()
        self.slots = model_slots(s)
        self.health = ProviderHealth()
        self.limits = {"nvidia": RateLimiter(max(1, s.nvidia_rpm_limit - 5))}
        self.emit: Emit = emit or (lambda kind, data: None)
        self.timeout_s = s.llm_call_timeout_s

    def peek(self, profile: TaskProfile) -> RouteDecision:
        """What would be chosen now, without emitting a trace event."""
        return choose(profile, self.health, self.slots)

    def decide(self, profile: TaskProfile) -> RouteDecision:
        decision = choose(profile, self.health, self.slots)
        self.emit("llm_route", {"role": profile.role, "complexity": profile.complexity,
                                "policy": profile.policy, "est_input_tokens": profile.est_input_tokens,
                                **decision.as_trace()})
        return decision

    async def _admit(self, decision: RouteDecision) -> bool:
        limiter = self.limits.get(decision.provider)
        if limiter is None:
            return True
        if await limiter.acquire(max_wait_s=5.0):
            return True
        self.health.trip(decision.provider, "local rate limit (rpm budget exhausted)", limiter.wait_time())
        return False

    def _model(self, decision: RouteDecision, tools: list | None, max_tokens: int, schema: type | None = None):
        llm = make_chat_model(self.slots[decision.slot], max_tokens=max_tokens, reasoning=decision.reasoning)
        if schema is not None:
            # include_raw keeps the AIMessage, so token usage (→ cost) is not lost. ChatNVIDIA raises
            # NotImplementedError for include_raw; it returns the parsed object alone (see ainvoke).
            if decision.provider == "nvidia":
                return llm.with_structured_output(schema)
            return llm.with_structured_output(schema, include_raw=True)
        return llm.bind_tools(tools) if tools else llm

    def _account(self, decision: RouteDecision, msg: AIMessage, t0: float, ttft: float | None) -> CallResult:
        usage = msg.usage_metadata or {}
        inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
        result = CallResult(msg, decision, time.perf_counter() - t0, inp, out,
                            cost_usd(decision.model, inp, out, cached), ttft)
        self.emit("llm_call", {"slot": decision.slot, "model": decision.model, "latency_s": round(result.latency_s, 3),
                               "ttft_s": round(ttft, 3) if ttft else None, "input_tokens": inp,
                               "output_tokens": out, "cost_usd": round(result.cost_usd, 6)})
        return result

    def _on_failure(self, decision: RouteDecision, e: Exception, profile: TaskProfile) -> RouteDecision | None:
        retryable, retry_after = _is_retryable(e)
        self.emit("llm_error", {"slot": decision.slot, "error": repr(e)[:300], "retryable": retryable})
        if not retryable or profile.forced_provider:
            return None
        self.health.trip(decision.provider, type(e).__name__, retry_after)
        return self.decide(profile)

    async def ainvoke(self, profile: TaskProfile, messages: list[BaseMessage], *, tools: list | None = None,
                      schema: type | None = None, max_tokens: int = 2048,
                      decision: RouteDecision | None = None) -> CallResult:
        decision = decision or self.decide(profile)
        for attempt in range(2):
            if not await self._admit(decision):
                decision = self.decide(profile)
            t0 = time.perf_counter()
            try:
                model = self._model(decision, tools, max_tokens, schema)
                out = await asyncio.wait_for(model.ainvoke(messages), self.timeout_s)
                if schema is None:
                    if _is_empty(out):
                        raise EmptyCompletionError(f"{decision.slot} returned an empty completion")
                    return self._account(decision, out, t0, None)
                if not isinstance(out, dict):  # parsed object only (NVIDIA): estimate usage; the model is free
                    if out is None:
                        raise ValueError("structured output did not parse")
                    text = out.model_dump_json()
                    usage = {"input_tokens": estimate_tokens(messages), "output_tokens": estimate_tokens([text])}
                    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
                    out = {"raw": AIMessage(content=text, usage_metadata=usage), "parsed": out}
                if out.get("parsed") is None:
                    raise ValueError(f"structured output did not parse: {out.get('parsing_error')}")
                result = self._account(decision, out["raw"], t0, None)
                result.parsed = out["parsed"]
                return result
            except Exception as e:  # noqa: BLE001 — classified below
                if attempt == 1 or isinstance(e, asyncio.CancelledError):
                    raise
                if isinstance(e, asyncio.TimeoutError):
                    e = TimeoutError(f"{decision.slot} timed out after {self.timeout_s}s")
                next_decision = self._on_failure(decision, e, profile)
                if next_decision is None or next_decision.provider == decision.provider:
                    raise
                decision = next_decision
        raise RuntimeError("unreachable")

    async def astream(self, profile: TaskProfile, messages: list[BaseMessage], *, max_tokens: int = 2048,
                      decision: RouteDecision | None = None) -> AsyncIterator[str | CallResult]:
        """Yield text tokens, then a final CallResult. Falls back only if no token was sent yet."""
        decision = decision or self.decide(profile)
        for attempt in range(2):
            if not await self._admit(decision):
                decision = self.decide(profile)
            t0, ttft, full = time.perf_counter(), None, None
            try:
                async for chunk in self._model(decision, None, max_tokens).astream(messages):
                    full = chunk if full is None else full + chunk
                    text = chunk.text  # str for chat-completions and NVIDIA; content blocks for the Responses API
                    if text:
                        ttft = ttft or time.perf_counter() - t0
                        yield text
                msg = AIMessage(content=full.content if full else "", usage_metadata=getattr(full, "usage_metadata", None))
                yield self._account(decision, msg, t0, ttft)
                return
            except Exception as e:  # noqa: BLE001
                if attempt == 1 or ttft is not None:
                    raise
                next_decision = self._on_failure(decision, e, profile)
                if next_decision is None or next_decision.provider == decision.provider:
                    raise
                decision = next_decision


def estimate_tokens(messages: list[BaseMessage] | list[str]) -> int:
    from src.rag.chunking import n_tokens

    return sum(n_tokens(m if isinstance(m, str) else str(m.content)) + 4 for m in messages)

