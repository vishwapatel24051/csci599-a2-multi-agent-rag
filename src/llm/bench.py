"""Phase 0 benchmark: measure each candidate model on the three call shapes the router will send it.

  route  structured output (the supervisor's job): latency + schema validity + correct choice
  tool   tool selection between two tools (worker agents' job): picks the right tool?
  rag    grounded answer over 5 real corpus chunks, streamed: TTFT, total time, cites [S#]?

Results (median over REPS) decide which slot gets which job; raw rows are saved to
data/bench/ as evidence for docs/evaluation.md.

Usage: python -m src.llm.bench [--reps 3] [--slots openai:fast nvidia:strong ...]
"""

import argparse
import asyncio
import json
import re
import statistics
import time
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import ROOT
from src.llm.providers import make_chat_model, model_slots
from src.rag.build_chunks import CHUNKS_PATH
from src.rag.chunking import read_jsonl

BENCH_DIR = ROOT / "data" / "bench"


class Route(BaseModel):
    next_agent: str = Field(description="one of: research, code, data, summary")
    reason: str


@tool
def search_corpus(query: str) -> str:
    """Search the indexed research papers and framework docs."""
    return ""


@tool
def run_sql(sql: str) -> str:
    """Run a read-only SQL query over the papers / authors / llm_models tables."""
    return ""


ROUTE_CASES = [
    ("Write a Python function that builds a LangGraph StateGraph with two nodes.", "code"),
    ("How many papers in the corpus were published in 2023?", "data"),
    ("Summarize the Reflexion paper.", "summary"),
    ("What does the ReAct paper say about hallucination in chain-of-thought?", "research"),
]
TOOL_CASES = [
    ("Which author appears on the most papers in the corpus?", "run_sql"),
    ("How does Self-RAG decide when to retrieve?", "search_corpus"),
]
RAG_QUESTION = "How does ReAct differ from chain-of-thought prompting, and why does it reduce hallucination?"


def _rag_prompt() -> list:
    chunks = [c for c in read_jsonl(CHUNKS_PATH) if c.doc_id == "arxiv:2210.03629"][:5]
    context = "\n\n".join(f"[S{i}] ({c.title}, p.{c.page_start})\n{c.text}" for i, c in enumerate(chunks, 1))
    return [
        SystemMessage("Answer using only the sources. Cite every claim with [S#]. Be concise (under 200 words)."),
        HumanMessage(f"Sources:\n{context}\n\nQuestion: {RAG_QUESTION}"),
    ]


async def bench_route(llm) -> dict:
    ok, times = 0, []
    for question, expected in ROUTE_CASES:
        t0 = time.perf_counter()
        try:
            out = await llm.with_structured_output(Route).ainvoke(
                f"Route this request to one agent (research, code, data, summary): {question}")
            ok += int(out is not None and out.next_agent.strip().lower() == expected)
        except Exception:  # noqa: BLE001 — a failure is a data point
            pass
        times.append(time.perf_counter() - t0)
    return {"accuracy": ok / len(ROUTE_CASES), "median_s": statistics.median(times)}


async def bench_tool(llm) -> dict:
    ok, times = 0, []
    bound = llm.bind_tools([search_corpus, run_sql])
    for question, expected in TOOL_CASES:
        t0 = time.perf_counter()
        try:
            msg = await bound.ainvoke(question)
            ok += int(bool(msg.tool_calls) and msg.tool_calls[0]["name"] == expected)
        except Exception:  # noqa: BLE001
            pass
        times.append(time.perf_counter() - t0)
    return {"accuracy": ok / len(TOOL_CASES), "median_s": statistics.median(times)}


async def bench_rag(llm, messages) -> dict:
    t0 = time.perf_counter()
    ttft, text, usage = None, "", None
    async for chunk in llm.astream(messages):
        if chunk.text and ttft is None:
            ttft = time.perf_counter() - t0
        text += chunk.text
        usage = chunk.usage_metadata or usage
    return {
        "ttft_s": ttft,
        "total_s": time.perf_counter() - t0,
        "cites": len(set(re.findall(r"\[S\d\]", text))),
        "input_tokens": usage.get("input_tokens") if usage else None,
        "output_tokens": usage.get("output_tokens") if usage else None,
    }


async def run(slot_keys: list[str], reps: int, reasoning: bool = False) -> list[dict]:
    slots = model_slots()
    rag_messages = _rag_prompt()
    rows = []
    for key in slot_keys:
        llm = make_chat_model(slots[key], max_tokens=1024, reasoning=reasoning)
        for rep in range(reps):
            row = {"slot": key, "model": slots[key].model_id, "rep": rep, "reasoning": reasoning}
            for name, coro in (("route", bench_route(llm)), ("tool", bench_tool(llm)), ("rag", bench_rag(llm, rag_messages))):
                try:
                    row[name] = await coro
                except Exception as e:  # noqa: BLE001
                    row[name] = {"error": repr(e)[:200]}
            rows.append(row)
            print(json.dumps(row))
    return rows


def summarize(rows: list[dict]) -> None:
    def med(values):
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else float("nan")

    print(f"\n{'slot':<15}{'route acc':>10}{'route s':>9}{'tool acc':>10}{'tool s':>8}{'rag TTFT':>10}{'rag total':>11}{'cites':>7}")
    for key in dict.fromkeys(r["slot"] for r in rows):
        rs = [r for r in rows if r["slot"] == key]
        print(f"{key:<15}"
              f"{med([r['route'].get('accuracy') for r in rs]):>10.2f}{med([r['route'].get('median_s') for r in rs]):>9.1f}"
              f"{med([r['tool'].get('accuracy') for r in rs]):>10.2f}{med([r['tool'].get('median_s') for r in rs]):>8.1f}"
              f"{med([r['rag'].get('ttft_s') for r in rs]):>10.1f}{med([r['rag'].get('total_s') for r in rs]):>11.1f}"
              f"{med([r['rag'].get('cites') for r in rs]):>7.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--slots", nargs="*", default=list(model_slots()))
    parser.add_argument("--reasoning", action="store_true", help="NVIDIA thinking mode on (default off)")
    parser.add_argument("--tag", default="", help="suffix for the output file")
    args = parser.parse_args()

    rows = asyncio.run(run(args.slots, args.reps, args.reasoning))
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / f"bench_{date.today().isoformat()}{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(rows, indent=2))
    summarize(rows)
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
