"""Run the fixed question set through the deployed chat API and record everything evaluation needs.

    python -m src.eval.collect --mode auto            # balanced routing (what users get)
    python -m src.eval.collect --mode openai          # every role forced to OpenAI
    python -m src.eval.collect --mode nvidia          # every role forced to NVIDIA NIM

Each question runs in a fresh session through POST /api/chat (the real SSE path), so timings include the
network, Cloud Run, the supervisor hop and the agents. Per question it records the answer, the retrieved
contexts (chunk texts, in retrieval order, for RAGAS), client-side TTFT/total, and a latency breakdown
summed from the trace: retrieval (Pinecone + query embedding), LLM calls, MCP tool calls, and the rest.
Output: src/eval/runs/<tag>.jsonl (one JSON object per question).
"""

import argparse
import asyncio
import json
import time
import uuid
from collections import Counter

import httpx

from src.config import ROOT

DEFAULT_URL = "https://multi-agent-rag-157472143408.us-east4.run.app"
QUESTIONS = ROOT / "src/eval/questions.json"
CHUNKS = ROOT / "data/processed/chunks.jsonl"
RUNS = ROOT / "src/eval/runs"


def load_chunk_texts() -> dict[str, str]:
    with open(CHUNKS) as f:
        return {c["id"]: c["text"] for c in map(json.loads, f)}


async def run_question(client: httpx.AsyncClient, url: str, q: dict, mode: str, policy: str,
                       chunk_texts: dict[str, str]) -> dict:
    session = f"eval-{mode}-{q['id']}-{uuid.uuid4().hex[:8]}"[:64].replace(".", "-")
    body = {"query": q["question"], "session_id": session, "policy": policy}
    if mode in ("openai", "nvidia"):
        body["force_provider"] = mode

    trace, citations, done, error = [], [], {}, None
    t0 = time.perf_counter()
    first_byte = first_token = None
    phases: dict[str, float] = {}  # client-side time of the first occurrence of each milestone
    event = None
    try:
        async with client.stream("POST", f"{url}/api/chat", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                first_byte = first_byte or time.perf_counter() - t0
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    now = round(time.perf_counter() - t0, 3)
                    if event == "trace":
                        trace.append(data)
                        milestone = {"agent_route": "routed_s", "agent_start": "agent_start_s",
                                     "rag_retrieval": "first_retrieval_s"}.get(data["kind"])
                        if milestone:
                            phases.setdefault(milestone, now)
                    elif event == "token" and data["text"].strip():  # a bare "\n" before a tool call isn't an answer
                        first_token = first_token or now
                    elif event == "citation":
                        citations.append(data)
                    elif event == "done":
                        done = data
                    elif event == "error":
                        error = data.get("message")
    except Exception as e:  # noqa: BLE001 — record the failure, keep the run going
        error = f"{type(e).__name__}: {e}"
    total = time.perf_counter() - t0

    # Retrieved contexts, in retrieval order, deduplicated across searches (what the agent actually saw).
    # Agents that read whole documents (summary's fetch_document) don't search; their cited chunks are appended.
    chunk_ids = list(dict.fromkeys(h["chunk_id"] for e in trace if e["kind"] == "rag_retrieval" for h in e["hits"]))
    retrieved = len(chunk_ids)
    chunk_ids += [c["chunk_id"] for c in citations if c.get("chunk_id") and c["chunk_id"] not in chunk_ids]
    retrieval_s = sum(e["latency_s"] for e in trace if e["kind"] == "rag_retrieval")
    llm_calls = [e for e in trace if e["kind"] == "llm_call"]
    llm_s = sum(e["latency_s"] for e in llm_calls)
    mcp_calls = [e for e in trace if e["kind"] == "mcp_tool_call"]
    mcp_s = sum(e.get("latency_s") or 0 for e in mcp_calls)
    stats = done.get("turn_stats", {})
    return {
        "id": q["id"], "topic": q["topic"], "mode": mode, "policy": policy, "session_id": session,
        "question": q["question"], "answer": done.get("final_answer", ""), "error": error,
        "contexts": [chunk_texts.get(cid, "") for cid in chunk_ids], "chunk_ids": chunk_ids,
        "n_retrieved": retrieved,
        "expected_docs": q["docs"],
        "cited": [c["label"] for c in citations],
        "web_used": bool(mcp_calls),
        "agents": stats.get("agents", []), "mode_used": stats.get("mode"), "providers": stats.get("providers", []),
        "models": dict(Counter(f"{e['node']}:{e['model']}" for e in llm_calls)),
        "routes": [{"next": e["next"], "complexity": e["complexity"], "reason": e["reason"]}
                   for e in trace if e["kind"] == "agent_route"],
        "timing": {
            "first_byte_s": round(first_byte or 0, 3), "ttft_s": round(first_token, 3) if first_token else None,
            "total_s": round(total, 3), "server_ttft_s": done.get("timing", {}).get("ttft_s"),
            "server_total_s": done.get("timing", {}).get("total_s"), **phases,
        },
        "latency": {
            "retrieval_s": round(retrieval_s, 3), "llm_s": round(llm_s, 3), "mcp_s": round(mcp_s, 3),
            "other_s": round(max(total - retrieval_s - llm_s - mcp_s, 0), 3),
        },
        "llm_calls": [{"node": e["node"], "model": e["model"], "latency_s": e["latency_s"],
                       "in": e["input_tokens"], "out": e["output_tokens"], "cost_usd": e["cost_usd"]} for e in llm_calls],
        "tokens": {"in": stats.get("input_tokens", 0), "out": stats.get("output_tokens", 0)},
        "llm_cost_usd": stats.get("llm_cost_usd", 0.0),
        "retrieval_queries": [e["query"] for e in trace if e["kind"] == "rag_retrieval"],
        "mcp_tools": dict(Counter(e["tool"] for e in mcp_calls)),
        "llm_errors": [e for e in trace if e["kind"] in ("llm_error", "agent_error")],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "openai", "nvidia"], default="auto")
    ap.add_argument("--policy", choices=["economy", "balanced", "quality"], default="balanced")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--tag", help="output name (default: <mode>_<date>)")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text())["questions"]
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    chunk_texts = load_chunk_texts()
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{args.tag or args.mode + '_' + time.strftime('%Y-%m-%d')}.jsonl"

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=60)) as client:
        async def one(q):
            async with sem:
                rec = await run_question(client, args.url, q, args.mode, args.policy, chunk_texts)
                t = rec["timing"]
                print(f"{rec['id']:<22} {args.mode:<6} ttft={t['ttft_s']} total={t['total_s']}s "
                      f"ctx={len(rec['contexts'])} agents={rec['agents']} ${rec['llm_cost_usd']}"
                      + (f" ERROR {rec['error']}" if rec["error"] else ""), flush=True)
                return rec
        records = await asyncio.gather(*(one(q) for q in questions))
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} records to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
