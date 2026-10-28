"""FastAPI app: static UI + SSE chat endpoint.

POST /api/chat  {query, session_id, policy?, force_provider?}  →  text/event-stream with frames:
  trace     one orchestration event (agent_route, llm_route, llm_call, rag_retrieval, mcp_tool_call, ...)
  token     {"text", "node"}: a streamed LLM token and the agent/node that produced it
  citation  one cited source ([S#] corpus chunk or [W#] web page)
  done      {"final_answer", "turn_stats", "timing"}
  error     {"message"}
The browser consumes this with fetch() + ReadableStream (EventSource can't POST).
"""

import asyncio
import json
import logging
import re
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field

from src.agents.specialists import AGENTS
from src.config import ROOT, get_settings
from src.graph.build import build_graph
from src.llm.providers import model_slots
from src.persistence import checkpointer

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# httpx logs every request URL at INFO, and the Tavily MCP URL carries the API key as a query parameter.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("api")

STATIC = ROOT / "static"
HEARTBEAT_S = 15
STREAMED_NODES = set(AGENTS) | {"synthesize"}  # not the supervisor: its tokens are routing JSON
_LABEL = re.compile(r"\[((?:S|W)\d+)\]")


async def _warm_up() -> None:
    """Open the expensive clients at startup so the first request after a cold start is faster."""
    import openai.resources  # noqa: F401 — the openai client imports this lazily; load it once, on one thread

    from src.mcp_client import tavily
    from src.rag.retriever import get_vectorstore

    t0 = time.perf_counter()
    try:
        await asyncio.to_thread(get_vectorstore)  # Pinecone + OpenAI embeddings clients
        await tavily.tools()  # MCP handshake + tool discovery
        log.info("warm-up done in %.1fs", time.perf_counter() - t0)
    except Exception:  # noqa: BLE001 — warm-up is best effort; requests will retry lazily
        log.warning("warm-up failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with checkpointer() as saver:
        app.state.graph = build_graph(checkpointer=saver)
        app.state.saver = saver
        # Awaited, not a background task: uvicorn accepts connections only after startup, so no request can race
        # it. (On Cloud Run, a background warm-up importing openai modules in a worker thread while the first
        # request imported them on the event loop raised _DeadlockError.)
        await _warm_up()
        log.info("graph ready (state backend: %s)", get_settings().state_backend)
        yield


app = FastAPI(title="Multi-Agent RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    policy: Literal["economy", "balanced", "quality"] = "balanced"
    force_provider: Literal["openai", "nvidia"] | None = None  # evaluation shootout only


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _chat_events(req: ChatRequest, graph):
    t0 = time.perf_counter()
    ttft = None
    config = {"configurable": {"thread_id": req.session_id}, "recursion_limit": 40}
    inputs = {"messages": [HumanMessage(req.query)], "policy": req.policy, "forced_provider": req.force_provider}
    try:
        async for mode, payload in graph.astream(inputs, config, stream_mode=["custom", "messages"]):
            if mode == "custom":
                yield _sse("trace", payload)
            else:
                chunk, meta = payload
                # Only real LLM token chunks. "messages" mode also re-emits whole AIMessages that nodes write
                # to state (e.g. the synthesizer's final answer), which would duplicate the streamed text.
                if not isinstance(chunk, AIMessageChunk):
                    continue
                node = meta.get("langgraph_node")
                # .text, not .content: with tools bound, langchain-openai uses the Responses API, whose chunks
                # carry content as a list of typed blocks; NVIDIA's are plain strings. .text handles both.
                text = chunk.text
                if text and node in STREAMED_NODES and not chunk.tool_call_chunks:
                    if text.strip():  # a tool-calling round can stream a bare "\n" first; that isn't an answer
                        ttft = ttft or time.perf_counter() - t0
                    yield _sse("token", {"text": text, "node": node})

        state = (await graph.aget_state(config)).values
        answer = state.get("final_answer", "")
        sources = state.get("sources", {})
        for label in dict.fromkeys(_LABEL.findall(answer)):
            if label in sources:
                src = dict(sources[label])
                src["snippet"] = src.pop("text", "")[:400]
                yield _sse("citation", src)
        timing = {"ttft_s": round(ttft, 3) if ttft else None, "total_s": round(time.perf_counter() - t0, 3)}
        log.info("turn session=%s timing=%s stats=%s", req.session_id[:8], timing, state.get("turn_stats"))
        yield _sse("done", {"final_answer": answer, "turn_stats": state.get("turn_stats", {}), "timing": timing})
    except Exception as e:  # noqa: BLE001 — report to the client instead of dropping the stream
        log.exception("chat failed")
        yield _sse("error", {"message": f"{type(e).__name__}: {e}"[:500]})


async def _with_heartbeat(events):
    """Interleave SSE comments so proxies don't close an idle stream during long agent calls."""
    queue: asyncio.Queue = asyncio.Queue()

    async def pump():
        try:
            async for item in events:
                await queue.put(item)
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), HEARTBEAT_S)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if item is None:
                break
            yield item
    finally:
        task.cancel()


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    return StreamingResponse(
        _with_heartbeat(_chat_events(req, request.app.state.graph)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions/{session_id}")
async def session_history(session_id: str, request: Request):
    """Conversation restored from the checkpointer: proves state outlives the process/instance."""
    snapshot = await request.app.state.graph.aget_state({"configurable": {"thread_id": session_id}})
    messages = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    return {"session_id": session_id, "messages": [{"role": m.type, "content": m.text} for m in messages]}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    await request.app.state.saver.adelete_thread(session_id)
    return {"deleted": session_id}


@app.get("/api/info")
async def info():
    s = get_settings()
    return {
        "agents": {name: spec.description for name, spec in AGENTS.items()},
        "models": {k: {"provider": v.provider, "model": v.model_id} for k, v in model_slots().items()},
        "policies": ["economy", "balanced", "quality"],
        "default_policy": s.routing_policy,
        "state_backend": s.state_backend,
        "vector_store": f"pinecone:{s.pinecone_index}",
    }


@app.get("/health")  # not /healthz: Cloud Run's front end reserves paths ending in "z" and returns its own 404
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    path: Path = STATIC / "index.html"
    if not path.exists():
        raise HTTPException(404, "UI not built")
    return FileResponse(path)
