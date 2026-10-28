"""Retrieval + augmentation (LCEL).

    {"query", "filter"} ──► pinecone_search (top RETRIEVAL_K by cosine)
                        ──► diversify (≤ PER_DOC_CAP chunks per document, keep CONTEXT_K)
                        ──► list[(Document, score)]

Augmentation is done by `SourceRegistry.format`, which turns hits into numbered source
blocks. Labels are stable for the whole turn: corpus chunks are [S1], [S2], …; web pages
from Tavily are [W1], [W2], … The registry maps each label back to its chunk ID, page and
URL, which is what the UI's citation panel shows.
"""

import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from src.config import get_settings

PER_DOC_CAP = 3  # diversity: no single paper can take over the context
STATE_TEXT_CHARS = 4000  # per-source text kept in graph state (see SourceRegistry.as_state)
_BLANK_RUNS = re.compile(r"\n\s*\n(\s*\n)+")  # PDF layout leaves runs of blank lines


@dataclass
class Source:
    label: str
    kind: str  # "corpus" | "web"
    title: str
    url: str
    text: str
    chunk_id: str | None = None
    doc_id: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None

    def as_event(self) -> dict:
        d = asdict(self)
        d["snippet"] = d.pop("text")[:300]
        return d


class SourceRegistry:
    """Per-turn label allocator shared by every agent in the turn."""

    def __init__(self, existing: dict[str, dict] | None = None):
        self.sources: dict[str, Source] = {k: Source(**v) for k, v in (existing or {}).items()}
        self._by_key = {(s.chunk_id or s.url): s for s in self.sources.values()}

    def _next(self, prefix: str) -> str:
        return f"{prefix}{sum(1 for k in self.sources if k.startswith(prefix)) + 1}"

    def add_chunk(self, doc: Document, score: float | None) -> Source:
        m = doc.metadata
        key = m.get("doc_id", "") + f"#{int(m.get('chunk_index', 0)):04d}"
        if key in self._by_key:
            return self._by_key[key]
        body = doc.page_content.split("\n\n", 1)[-1]  # drop the "Title — Section" header line
        src = Source(self._next("S"), "corpus", m.get("title", ""), m.get("url", ""), body, key, m.get("doc_id"),
                     m.get("section"), _int(m.get("page_start")), _int(m.get("page_end")),
                     round(score, 4) if score is not None else None)
        self.sources[src.label] = self._by_key[key] = src
        return src

    def add_web(self, url: str, title: str, content: str, score: float | None = None) -> Source:
        if url in self._by_key:
            return self._by_key[url]
        src = Source(self._next("W"), "web", title or url, url, content, score=score)
        self.sources[src.label] = self._by_key[url] = src
        return src

    def as_state(self) -> dict[str, dict]:
        # The model already read the full text in the tool result; state keeps an excerpt for citations and
        # evaluation. Bounded because the checkpoint is one Firestore document (1 MiB max) and a web_extract
        # page can be 100k+ characters. Corpus chunks (<= 864 tokens) fit whole.
        return {k: asdict(v) | {"text": v.text[:STATE_TEXT_CHARS]} for k, v in self.sources.items()}


_LENTICULAR = re.compile(r"【\s*((?:S|W)\d+)\s*】")


def normalize_citations(text: str) -> str:
    """Nemotron sometimes writes 【W2】 instead of [W2]; citation extraction and the UI expect [W2]."""
    return _LENTICULAR.sub(r"[\1]", text)


def format_sources(sources: list[Source]) -> str:
    """Augmentation template: one block per source, label first so the model can cite it."""
    blocks = []
    for s in sources:
        if s.kind == "corpus":
            pages = f", p.{s.page_start}" + (f"–{s.page_end}" if s.page_end and s.page_end != s.page_start else "") \
                if s.page_start else ""
            relevance = f" (relevance {s.score:.2f})" if s.score is not None else ""
            head = f"[{s.label}] {s.title} — {s.section}{pages}{relevance}"
        else:
            head = f"[{s.label}] {s.title} <{s.url}>"
        blocks.append(f"{head}\n{_BLANK_RUNS.sub(chr(10) * 2, s.text.strip())}")
    return "\n\n---\n\n".join(blocks) if blocks else "(no results)"


def _int(v) -> int | None:
    return int(v) if v is not None else None


@lru_cache
def get_index():
    from pinecone import Pinecone

    s = get_settings()
    return Pinecone(api_key=s.pinecone_api_key.get_secret_value()).Index(s.pinecone_index)


@lru_cache
def get_vectorstore():
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    s = get_settings()
    embeddings = OpenAIEmbeddings(model=s.openai_embedding_model, api_key=s.openai_api_key.get_secret_value())
    return PineconeVectorStore(index=get_index(), embedding=embeddings, text_key="text", namespace=s.pinecone_namespace)


def fetch_document_chunks(doc_id: str) -> list[Document]:
    """All chunks of one document, in order (deterministic IDs make this a prefix listing)."""
    s = get_settings()
    index = get_index()
    ids = [i for page in index.list(prefix=f"{doc_id}#", namespace=s.pinecone_namespace) for i in page]
    docs = []
    for i in range(0, len(ids), 100):
        fetched = index.fetch(ids=ids[i : i + 100], namespace=s.pinecone_namespace).vectors
        docs += [Document(page_content=v.metadata["text"], metadata=dict(v.metadata)) for v in fetched.values()]
    return sorted(docs, key=lambda d: int(d.metadata["chunk_index"]))


async def _search(inputs: dict) -> list[tuple[Document, float]]:
    s = get_settings()
    return await get_vectorstore().asimilarity_search_with_score(
        inputs["query"], k=inputs.get("k", s.retrieval_k), filter=inputs.get("filter"))


def _diversify(hits: list[tuple[Document, float]], limit: int | None = None) -> list[tuple[Document, float]]:
    """Best `limit` hits with at most PER_DOC_CAP per document, so one paper can't crowd out the others.

    The cap is a preference, not a filter: when it leaves fewer than `limit` chunks (all hits came from one
    or two documents, typical for a question about one paper), the best capped-out hits fill the free slots.
    A hard cap kept only 3 chunks for such questions and dropped the abstract with the key numbers (RAGAS
    context recall 0.33 on those questions). Rank order is preserved.
    """
    limit = limit or get_settings().context_k
    per_doc: dict[str, int] = {}
    chosen: set[int] = set()
    for i, (doc, _) in enumerate(hits):  # already sorted by score, best first
        doc_id = doc.metadata.get("doc_id", "")
        if per_doc.get(doc_id, 0) < PER_DOC_CAP:
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            chosen.add(i)
            if len(chosen) == limit:
                break
    for i in range(len(hits)):  # backfill
        if len(chosen) >= limit:
            break
        chosen.add(i)
    return [hits[i] for i in sorted(chosen)]


retrieval_chain: Runnable = (
    RunnableLambda(_search).with_config(run_name="pinecone_search")
    | RunnableLambda(_diversify).with_config(run_name="diversify")
)
