"""Tool factories. Each agent gets its own tool set: distinct tool access is what makes it an agent.

Tools are built per agent run, closed over an AgentContext so results land in the turn's
shared SourceRegistry ([S#] / [W#] labels) and every call is traced with the agent's name.
"""

import ast
import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool

from src.mcp_client import tavily
from src.rag.retriever import SourceRegistry, fetch_document_chunks, format_sources, get_vectorstore, retrieval_chain
from src.tools import sql_tools
from src.trace import emit

MAX_DOC_TOKENS = 120_000  # cap for whole-document fetches


@dataclass
class AgentContext:
    agent: str
    registry: SourceRegistry
    tool_log: list[dict] = field(default_factory=list)

    def log(self, tool: str, args: dict, ok: bool, latency_s: float, summary: str) -> None:
        entry = {"agent": self.agent, "tool": tool, "args": args, "ok": ok, "latency_s": round(latency_s, 3),
                 "summary": summary}
        self.tool_log.append(entry)
        emit("tool_call", entry)


# ---------------------------------------------------------------- corpus (Pinecone)

def search_corpus_tool(ctx: AgentContext, *, code_only: bool = False) -> BaseTool:
    name = "search_code_examples" if code_only else "search_corpus"

    async def run(query: str, source: Literal["any", "papers", "docs"] = "any") -> str:
        flt: dict = {}
        if code_only:
            flt["has_code"] = {"$eq": True}
        if source != "any":
            flt["source_type"] = {"$eq": "paper" if source == "papers" else "docs"}
        t0 = time.perf_counter()
        hits = await retrieval_chain.ainvoke({"query": query, "filter": flt or None})
        sources = [ctx.registry.add_chunk(doc, score) for doc, score in hits]
        emit("rag_retrieval", {
            "agent": ctx.agent, "tool": name, "query": query, "filter": flt or None,
            "latency_s": round(time.perf_counter() - t0, 3),
            "hits": [{"label": s.label, "chunk_id": s.chunk_id, "title": s.title, "section": s.section,
                      "page": s.page_start, "score": s.score} for s in sources],
        })
        return format_sources(sources)

    description = (
        "Search code-bearing passages (LangGraph/LangChain/MCP/A2A docs and papers) for API usage and examples."
        if code_only else
        "Search the indexed corpus: 58 research papers on LLM agents, multi-agent systems, tool use, RAG, "
        "routing and evaluation, plus LangGraph/LangChain, MCP and A2A documentation. "
        "Returns numbered sources [S#] to cite. `source` narrows to papers or docs."
    )
    return StructuredTool.from_function(coroutine=run, name=name, description=description)


# ---------------------------------------------------------------- web (Tavily Remote MCP, shared)

def web_search_tool(ctx: AgentContext) -> BaseTool:
    async def run(query: str, max_results: int = 4) -> str:
        payload = await tavily.call("tavily_search", {"query": query, "max_results": max(1, min(max_results, 5))},
                                    agent=ctx.agent)
        sources = [ctx.registry.add_web(r["url"], r.get("title", ""), r.get("content", ""),
                                        float(r["score"]) if r.get("score") else None)
                   for r in payload.get("results", [])]
        return format_sources(sources)

    return StructuredTool.from_function(
        coroutine=run, name="web_search",
        description="Live web search (Tavily, via MCP) for current events, recent releases, versions, or anything "
                    "newer than the corpus. Returns numbered sources [W#] to cite.")


def web_extract_tool(ctx: AgentContext, max_chars: int = MAX_DOC_TOKENS * 4) -> BaseTool:
    async def run(url: str) -> str:
        payload = await tavily.call("tavily_extract", {"urls": [url]}, agent=ctx.agent)
        results = payload.get("results", [])
        if not results:
            return f"Could not extract {url}: {payload.get('failed_results') or payload.get('raw', 'no content')}"
        r = results[0]
        content = r.get("raw_content") or r.get("content") or ""
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n[truncated: first {max_chars:,} of {len(content):,} characters]"
        source = ctx.registry.add_web(url, r.get("title") or url, content)
        return format_sources([source])

    return StructuredTool.from_function(
        coroutine=run, name="web_extract",
        description="Fetch the full text of a web page (Tavily extract, via MCP). Returns it as source [W#].")


# ---------------------------------------------------------------- documents (whole-document access)

def document_tools(ctx: AgentContext) -> list[BaseTool]:
    async def find_document(query: str) -> str:
        t0 = time.perf_counter()
        con = sql_tools.connect()
        like = f"%{query.strip()}%"
        rows = con.execute(
            "SELECT 'arxiv:' || arxiv_id, title, year FROM papers WHERE title LIKE ? OR arxiv_id = ? "
            "UNION ALL SELECT doc_id, title, NULL FROM doc_pages WHERE title LIKE ? LIMIT 8",
            (like, query.strip(), like)).fetchall()
        if not rows:  # fall back to semantic search over chunks, deduplicated to documents
            hits = await get_vectorstore().asimilarity_search_with_score(query, k=12)
            seen = {}
            for doc, _ in hits:
                seen.setdefault(doc.metadata["doc_id"], (doc.metadata["doc_id"], doc.metadata["title"],
                                                         doc.metadata.get("year")))
            rows = list(seen.values())[:6]
        ctx.log("find_document", {"query": query}, bool(rows), time.perf_counter() - t0, f"{len(rows)} candidates")
        if not rows:
            return "No matching documents."
        return "\n".join(f"- {doc_id} — {title}" + (f" ({year})" if year else "") for doc_id, title, year in rows)

    async def fetch_document(doc_id: str) -> str:
        t0 = time.perf_counter()
        docs = await asyncio.to_thread(fetch_document_chunks, doc_id.strip())
        if not docs:
            ctx.log("fetch_document", {"doc_id": doc_id}, False, time.perf_counter() - t0, "not found")
            return f"No document with id {doc_id!r}. Use find_document first."
        sources, budget = [], MAX_DOC_TOKENS
        for d in docs:
            budget -= int(d.metadata.get("n_tokens", 400))
            if budget < 0:
                break
            sources.append(ctx.registry.add_chunk(d, None))
        ctx.log("fetch_document", {"doc_id": doc_id}, True, time.perf_counter() - t0,
                f"{len(sources)}/{len(docs)} chunks as {sources[0].label}–{sources[-1].label}")
        header = f"Document {doc_id}: {docs[0].metadata['title']} ({len(sources)} sections)\n\n"
        return header + format_sources(sources)

    return [
        StructuredTool.from_function(
            coroutine=find_document, name="find_document",
            description="Find corpus documents by title words, arXiv id, or topic. Returns doc_ids."),
        StructuredTool.from_function(
            coroutine=fetch_document, name="fetch_document",
            description="Fetch a whole corpus document by doc_id (e.g. 'arxiv:2210.03629'), in order, as labeled "
                        "sections [S#] to cite."),
    ]


# ---------------------------------------------------------------- structured data (SQLite)

def sql_agent_tools(ctx: AgentContext) -> list[BaseTool]:
    def traced(name: str, fn, args: dict) -> str:
        t0 = time.perf_counter()
        con = sql_tools.connect()
        try:
            out = fn(con)
            rows = max(out.count("\n") - 1, 0) if out.startswith("|") else None
            ctx.log(name, args, True, time.perf_counter() - t0, f"{rows} rows" if rows is not None else "ok")
            return out
        except sql_tools.SQLToolError as e:
            ctx.log(name, args, False, time.perf_counter() - t0, str(e))
            return f"Error: {e}"
        finally:
            con.close()

    async def list_tables() -> str:
        return await asyncio.to_thread(traced, "list_tables", sql_tools.list_tables, {})

    async def describe_table(table: str) -> str:
        return await asyncio.to_thread(traced, "describe_table", lambda c: sql_tools.describe_table(c, table),
                                       {"table": table})

    async def run_sql(sql: str) -> str:
        return await asyncio.to_thread(traced, "run_sql", lambda c: sql_tools.run_sql(c, sql), {"sql": sql})

    return [
        StructuredTool.from_function(coroutine=list_tables, name="list_tables",
                                     description="List the tables in the database with row counts."),
        StructuredTool.from_function(coroutine=describe_table, name="describe_table",
                                     description="Show a table's CREATE statement (with column comments) and sample rows."),
        StructuredTool.from_function(coroutine=run_sql, name="run_sql",
                                     description="Run ONE read-only SELECT (SQLite dialect). Returns a markdown table."),
    ]


# ---------------------------------------------------------------- code

def python_syntax_tool(ctx: AgentContext) -> BaseTool:
    async def check_python_syntax(code: str) -> str:
        t0 = time.perf_counter()
        try:
            ast.parse(code)
            out = f"OK: valid Python syntax ({code.count(chr(10)) + 1} lines). The code was parsed, not executed."
            ok = True
        except SyntaxError as e:
            out = f"SyntaxError at line {e.lineno}, column {e.offset}: {e.msg}\n{(e.text or '').rstrip()}"
            ok = False
        ctx.log("check_python_syntax", {"lines": code.count("\n") + 1}, ok, time.perf_counter() - t0, out[:120])
        return out

    return StructuredTool.from_function(
        coroutine=check_python_syntax, name="check_python_syntax",
        description="Check that Python code parses (static check only; nothing is executed).")
