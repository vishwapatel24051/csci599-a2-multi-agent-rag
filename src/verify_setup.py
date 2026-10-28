"""Phase 0: does every key work, and does every configured model ID exist?

Usage: python -m src.verify_setup
Checks are independent; a missing or bad key fails only its own rows.
Secrets are masked in any error text (the Tavily key travels in the URL).
"""

import asyncio
import time
from dataclasses import dataclass

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel

from src.config import Settings, get_settings


@dataclass
class Row:
    service: str
    check: str
    ok: bool
    detail: str = ""
    ms: float | None = None


class RouteProbe(BaseModel):
    next_agent: str
    reason: str


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."


ROUTE_PROMPT = (
    "You are a router. Agents: research, code, data, summary. "
    "Pick the agent for: 'Write a Python function that parses a CSV file.' Give a one-sentence reason."
)
TOOL_PROMPT = "What's the weather in Paris right now? Use the tool."


def _secrets(s: Settings) -> list[str]:
    return [k.get_secret_value() for k in (s.openai_api_key, s.nvidia_api_key, s.pinecone_api_key, s.tavily_api_key) if k]


def _mask(text: str, s: Settings) -> str:
    for secret in _secrets(s):
        text = text.replace(secret, "***")
    return text[:300]


async def _timed(coro):
    t0 = time.perf_counter()
    result = await coro
    return result, (time.perf_counter() - t0) * 1000


async def _list_model_ids(base_url: str, api_key: str) -> set[str]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        return {m["id"] for m in resp.json()["data"]}


async def check_openai(s: Settings) -> list[Row]:
    if not s.openai_api_key:
        return [Row("openai", "key present", False, "OPENAI_API_KEY not set")]
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    key = s.openai_api_key.get_secret_value()
    rows = []
    ids = await _list_model_ids("https://api.openai.com/v1", key)
    for model in (s.openai_fast_model, s.openai_strong_model, s.openai_embedding_model):
        rows.append(Row("openai", f"model exists: {model}", model in ids))

    llm = ChatOpenAI(model=s.openai_fast_model, api_key=key)
    out, ms = await _timed(llm.with_structured_output(RouteProbe).ainvoke(ROUTE_PROMPT))
    rows.append(Row("openai", f"structured output ({s.openai_fast_model})", out.next_agent == "code", out.next_agent, ms))

    llm = ChatOpenAI(model=s.openai_strong_model, api_key=key)
    msg, ms = await _timed(llm.bind_tools([get_weather]).ainvoke(TOOL_PROMPT))
    rows.append(Row("openai", f"tool calling ({s.openai_strong_model})", bool(msg.tool_calls), str(msg.tool_calls)[:80], ms))

    emb = OpenAIEmbeddings(model=s.openai_embedding_model, api_key=key)
    vec, ms = await _timed(emb.aembed_query("hello world"))
    rows.append(Row("openai", f"embedding dim ({s.openai_embedding_model})", len(vec) == s.embedding_dim, str(len(vec)), ms))
    return rows


async def check_nvidia(s: Settings) -> list[Row]:
    if not s.nvidia_api_key:
        return [Row("nvidia", "key present", False, "NVIDIA_API_KEY not set")]
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    key = s.nvidia_api_key.get_secret_value()
    rows = []
    ids = await _list_model_ids(s.nvidia_base_url, key)
    nemotrons = sorted(i for i in ids if "nemotron" in i and "embed" not in i and "rerank" not in i)
    rows.append(Row("nvidia", "nemotron chat models in catalog", bool(nemotrons), ", ".join(nemotrons)[:300]))

    for model in (s.nvidia_strong_model, s.nvidia_fast_model):
        rows.append(Row("nvidia", f"model exists: {model}", model in ids))
        if model not in ids:
            continue
        llm = ChatNVIDIA(model=model, api_key=key, base_url=s.nvidia_base_url, max_completion_tokens=512)
        try:
            msg, ms = await _timed(llm.bind_tools([get_weather]).ainvoke(TOOL_PROMPT))
            rows.append(Row("nvidia", f"tool calling ({model})", bool(msg.tool_calls), str(msg.tool_calls)[:80], ms))
        except Exception as e:  # noqa: BLE001 — report and keep checking
            rows.append(Row("nvidia", f"tool calling ({model})", False, _mask(repr(e), s)))
        try:
            out, ms = await _timed(llm.with_structured_output(RouteProbe).ainvoke(ROUTE_PROMPT))
            rows.append(Row("nvidia", f"structured output ({model})", out is not None and out.next_agent == "code",
                            getattr(out, "next_agent", "None"), ms))
        except Exception as e:  # noqa: BLE001
            rows.append(Row("nvidia", f"structured output ({model})", False, _mask(repr(e), s)))
    return rows


async def check_pinecone(s: Settings) -> list[Row]:
    if not s.pinecone_api_key:
        return [Row("pinecone", "key present", False, "PINECONE_API_KEY not set")]
    from pinecone import Pinecone

    pc = Pinecone(api_key=s.pinecone_api_key.get_secret_value())
    names, ms = await _timed(asyncio.to_thread(lambda: pc.list_indexes().names()))
    exists = s.pinecone_index in names
    return [
        Row("pinecone", "auth + list indexes", True, ", ".join(names) or "(none yet)", ms),
        Row("pinecone", f"index '{s.pinecone_index}'", True, "exists" if exists else "not created yet (ingest creates it)"),
    ]


async def check_tavily(s: Settings) -> list[Row]:
    if not s.tavily_api_key:
        return [Row("tavily", "key present", False, "TAVILY_API_KEY not set")]
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({"tavily": {"transport": "streamable_http", "url": s.tavily_mcp_url}})
    tools, ms = await _timed(client.get_tools())
    names = [t.name for t in tools]
    rows = [Row("tavily", "MCP connect + list tools", bool(names), ", ".join(names), ms)]
    search = next((t for t in tools if "search" in t.name), None)
    if search:
        result, ms = await _timed(search.ainvoke({"query": "LangGraph latest release", "max_results": 1}))
        rows.append(Row("tavily", f"call {search.name}", bool(result), str(result)[:80].replace("\n", " "), ms))
    return rows


async def main() -> None:
    s = get_settings()
    checks = [check_openai, check_nvidia, check_pinecone, check_tavily]
    results = await asyncio.gather(*(c(s) for c in checks), return_exceptions=True)

    rows: list[Row] = []
    for check, res in zip(checks, results):
        if isinstance(res, BaseException):
            rows.append(Row(check.__name__.removeprefix("check_"), "unexpected error", False, _mask(repr(res), s)))
        else:
            rows.extend(res)

    for r in rows:
        mark = "PASS" if r.ok else "FAIL"
        ms = f"{r.ms:7.0f} ms" if r.ms is not None else " " * 10
        print(f"{mark}  {r.service:<9} {ms}  {r.check:<55} {_mask(r.detail, s)}")
    print(f"\n{sum(r.ok for r in rows)}/{len(rows)} checks passed")


if __name__ == "__main__":
    asyncio.run(main())
