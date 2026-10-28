"""Stage 1 of ingestion (no API keys needed): raw files → data/processed/chunks.jsonl.

Usage: python -m src.rag.build_chunks
"""

import json
import os
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.rag.chunking import Chunk, chunk_document, clean_mdx, pdf_pages, write_jsonl
from src.rag.fetch_corpus import DOCS_DIR, PAPERS_DIR, ROOT

PROCESSED = ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED / "chunks.jsonl"
DOCUMENTS_PATH = PROCESSED / "documents.json"
EMBED_PRICE_PER_MTOK = 0.02  # text-embedding-3-small, USD


def _short_authors(authors: list[str]) -> str:
    last = authors[0].split()[-1] if authors else "Unknown"
    return last if len(authors) == 1 else f"{last} et al."


def _doc_label(source: str, path: str) -> str:
    if source == "mcp":
        return "MCP Specification" if "/specification/" in path else "MCP Docs"
    if source == "langgraph":
        return "LangChain Docs" if "/langchain/" in path else "LangGraph Docs"
    return "A2A Protocol"


def _paper_chunks(m: dict) -> list[Chunk]:
    doc = {
        "doc_id": f"arxiv:{m['arxiv_id']}",
        "title": m["title"],
        "source_type": "paper",
        "source": "arxiv",
        "url": m["pdf_url"],
        "topic": m["topic"],
        "year": int(m["published"][:4]),
        "authors": _short_authors(m["authors"]),
    }
    return chunk_document(doc, pdf_pages(PAPERS_DIR / f"{m['arxiv_id']}.pdf"))


def _doc_chunks(rec: dict) -> list[Chunk]:
    text, front = clean_mdx((ROOT / rec["local_path"]).read_text(encoding="utf-8"))
    page_title = front.get("title") or Path(rec["path"]).stem.replace("-", " ").title()
    slug = rec["path"].rsplit(".", 1)[0].removeprefix("src/").removeprefix("docs/")
    doc = {
        "doc_id": f"{rec['source']}:{slug}",
        "title": f"{_doc_label(rec['source'], rec['path'])}: {page_title}",
        "source_type": "docs",
        "source": rec["source"],
        "url": rec["url"],
        "topic": "protocols" if rec["source"] in ("mcp", "a2a") else "orchestration",
    }
    return chunk_document(doc, [(None, text)])


def main() -> None:
    papers = json.loads((PAPERS_DIR / "meta.json").read_text())
    docs = json.loads((DOCS_DIR / "meta.json").read_text())

    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        paper_results = list(pool.map(_paper_chunks, papers.values()))
    doc_results = [_doc_chunks(rec) for rec in docs]

    chunks = [c for group in paper_results + doc_results for c in group]
    write_jsonl(chunks, CHUNKS_PATH)

    registry: dict[str, dict] = {}
    for c in chunks:
        d = registry.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "source": c.source,
                                           "source_type": c.source_type, "url": c.url.split("#")[0],
                                           "topic": c.topic, "year": c.year, "authors": c.authors,
                                           "n_chunks": 0, "n_tokens": 0})
        d["n_chunks"] += 1
        d["n_tokens"] += c.n_tokens
    DOCUMENTS_PATH.write_text(json.dumps(registry, indent=2))

    sizes = [c.n_tokens for c in chunks]
    total = sum(sizes)
    print(f"documents: {len(registry)}  ({Counter(d['source'] for d in registry.values())})")
    print(f"chunks:    {len(chunks)}  (with code: {sum(c.has_code for c in chunks)})")
    print(f"tokens:    total {total:,}  mean {statistics.mean(sizes):.0f}  median {statistics.median(sizes):.0f}  "
          f"min {min(sizes)}  max {max(sizes)}")
    print(f"embedding cost estimate: ${total / 1e6 * EMBED_PRICE_PER_MTOK:.3f}")
    print(f"wrote {CHUNKS_PATH.relative_to(ROOT)} and {DOCUMENTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
