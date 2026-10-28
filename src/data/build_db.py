"""Build data/agentlab.db: the structured store queried by the Data agent.

Sources: arXiv metadata gathered by fetch_corpus, the chunk registry from build_chunks,
Semantic Scholar citation counts (best effort, cached), and src/llm/pricing.py.
Column docs live as SQL comments in the CREATE statements; SQLite keeps that text in
sqlite_master, and the Data agent's describe_table tool shows it to the model.

Usage: python -m src.data.build_db
"""

import json
import sqlite3
import time

import httpx

from src.config import get_settings
from src.llm.pricing import PRICES, PRICES_CHECKED_ON
from src.rag.build_chunks import DOCUMENTS_PATH
from src.rag.fetch_corpus import PAPERS_DIR, ROOT

CITATIONS_CACHE = PAPERS_DIR / "citations.json"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

SCHEMA = """
CREATE TABLE papers (
    arxiv_id TEXT PRIMARY KEY,         -- e.g. '2210.03629'; also the Pinecone doc_id 'arxiv:<arxiv_id>'
    title TEXT NOT NULL,
    topic TEXT NOT NULL,               -- one of: reasoning, agents, tool-use, multi-agent, rag, retrieval, routing, evaluation
    published DATE NOT NULL,           -- first arXiv submission date, YYYY-MM-DD
    year INTEGER NOT NULL,             -- year of first submission
    primary_category TEXT,             -- arXiv primary category, e.g. 'cs.CL'
    first_author TEXT,
    n_authors INTEGER,
    abstract TEXT,
    abs_url TEXT,
    n_chunks INTEGER,                  -- number of chunks indexed in the vector store
    n_tokens INTEGER,                  -- tokens indexed (tiktoken cl100k_base)
    citation_count INTEGER,            -- Semantic Scholar total citations at build time; NULL if unavailable
    influential_citation_count INTEGER -- Semantic Scholar 'influential' citations; NULL if unavailable
);

CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL          -- 'First Last' as listed on arXiv
);

CREATE TABLE paper_authors (
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    author_id INTEGER NOT NULL REFERENCES authors(author_id),
    position INTEGER NOT NULL,         -- 1 = first author
    PRIMARY KEY (arxiv_id, author_id)
);

CREATE TABLE doc_pages (
    doc_id TEXT PRIMARY KEY,           -- Pinecone doc_id, e.g. 'langgraph:oss/langgraph/persistence'
    source TEXT NOT NULL,              -- 'langgraph' (LangGraph + LangChain docs), 'mcp', or 'a2a'
    title TEXT NOT NULL,
    url TEXT NOT NULL,                 -- GitHub URL pinned to the indexed commit
    n_chunks INTEGER,
    n_tokens INTEGER
);

CREATE TABLE llm_models (
    model_id TEXT PRIMARY KEY,         -- exact API model ID
    provider TEXT NOT NULL,            -- 'openai' or 'nvidia'
    tier TEXT NOT NULL,                -- fast, mid, strong, flagship, embedding
    input_usd_per_mtok REAL,           -- USD per 1M input tokens (0 = free developer tier)
    output_usd_per_mtok REAL,          -- USD per 1M output tokens
    cached_input_usd_per_mtok REAL,    -- USD per 1M cached input tokens; NULL if not offered
    context_window INTEGER,            -- max tokens; NULL if not published
    used_by_router INTEGER NOT NULL,   -- 1 if this app's model router can send calls to it
    price_source TEXT,
    price_checked_on DATE
);
"""


def fetch_citations(arxiv_ids: list[str]) -> dict[str, dict]:
    """Best effort: the keyless Semantic Scholar pool is shared and often rate-limited."""
    if CITATIONS_CACHE.exists():
        return json.loads(CITATIONS_CACHE.read_text())
    for attempt in range(5):
        resp = httpx.post(S2_BATCH_URL, params={"fields": "citationCount,influentialCitationCount"},
                          json={"ids": [f"ARXIV:{i}" for i in arxiv_ids]}, timeout=60)
        if resp.status_code == 200:
            result = {aid: row for aid, row in zip(arxiv_ids, resp.json()) if row}
            CITATIONS_CACHE.write_text(json.dumps(result, indent=2))
            return result
        wait = 10 * 2 ** attempt
        print(f"  Semantic Scholar {resp.status_code}, retrying in {wait}s")
        time.sleep(wait)
    print("  citation counts unavailable; columns left NULL (re-run later to fill them)")
    return {}


def main() -> None:
    s = get_settings()
    papers = json.loads((PAPERS_DIR / "meta.json").read_text())
    docs = json.loads(DOCUMENTS_PATH.read_text())
    citations = fetch_citations(list(papers))
    router_models = {s.openai_fast_model, s.openai_strong_model, s.nvidia_fast_model, s.nvidia_strong_model}

    db_path = ROOT / "data" / "agentlab.db"
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    for aid, m in papers.items():
        d = docs.get(f"arxiv:{aid}", {})
        c = citations.get(aid, {})
        con.execute(
            "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, m["title"], m["topic"], m["published"], int(m["published"][:4]), m.get("primary_category"),
             m["authors"][0] if m["authors"] else None, len(m["authors"]), m["abstract"], m["abs_url"],
             d.get("n_chunks"), d.get("n_tokens"), c.get("citationCount"), c.get("influentialCitationCount")),
        )
        for pos, name in enumerate(m["authors"], 1):
            con.execute("INSERT OR IGNORE INTO authors(name) VALUES (?)", (name,))
            (author_id,) = con.execute("SELECT author_id FROM authors WHERE name = ?", (name,)).fetchone()
            con.execute("INSERT OR IGNORE INTO paper_authors VALUES (?,?,?)", (aid, author_id, pos))

    for d in docs.values():
        if d["source_type"] == "docs":
            con.execute("INSERT INTO doc_pages VALUES (?,?,?,?,?,?)",
                        (d["doc_id"], d["source"], d["title"], d["url"], d["n_chunks"], d["n_tokens"]))

    for p in PRICES.values():
        con.execute(
            "INSERT INTO llm_models VALUES (?,?,?,?,?,?,?,?,?,?)",
            (p.model_id, p.provider, p.tier, p.input_usd_per_mtok, p.output_usd_per_mtok,
             p.cached_input_usd_per_mtok, p.context_window, int(p.model_id in router_models),
             p.source, PRICES_CHECKED_ON),
        )

    con.commit()
    for table in ("papers", "authors", "paper_authors", "doc_pages", "llm_models"):
        (n,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table:<14} {n} rows")
    con.close()
    print(f"wrote {db_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
