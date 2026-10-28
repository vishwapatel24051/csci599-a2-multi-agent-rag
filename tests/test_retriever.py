from langchain_core.documents import Document

from src.rag.retriever import STATE_TEXT_CHARS, SourceRegistry, _diversify


def _hits(doc_ids: str) -> list[tuple[Document, float]]:
    return [(Document(f"chunk {i}", metadata={"doc_id": d, "chunk_index": i}), 1 - i / 100)
            for i, d in enumerate(doc_ids)]


def ids(hits) -> str:
    return "".join(doc.metadata["doc_id"] for doc, _ in hits)


def test_diversify_caps_chunks_per_document_when_others_exist():
    assert ids(_diversify(_hits("AAAABACA"), limit=5)) == "AAABC"


def test_diversify_backfills_when_one_document_dominates():
    # A single-paper question: the cap must not shrink the context from 5 to 3 chunks.
    kept = _diversify(_hits("AAAAAAAA"), limit=5)
    assert ids(kept) == "AAAAA" and [d.metadata["chunk_index"] for d, _ in kept] == [0, 1, 2, 3, 4]


def test_diversify_backfill_keeps_rank_order():
    assert [d.metadata["chunk_index"] for d, _ in _diversify(_hits("AAAAAB"), limit=5)] == [0, 1, 2, 3, 5]


def test_state_keeps_bounded_excerpt_of_large_pages():
    # A checkpoint is one Firestore document (1 MiB max); an extracted web page can be 100k+ characters.
    reg = SourceRegistry()
    page = reg.add_web("https://example.com", "Example", "x" * 100_000)
    chunk = reg.add_chunk(Document("Title — Intro\n\nshort body", metadata={"doc_id": "arxiv:1", "chunk_index": 0}), 0.8)
    state = reg.as_state()
    assert len(state[page.label]["text"]) == STATE_TEXT_CHARS
    assert state[chunk.label]["text"] == "short body"
    assert len(page.text) == 100_000  # the in-turn Source (what the model saw) is untouched


def test_lenticular_citation_brackets_are_normalized():
    from src.rag.retriever import normalize_citations

    assert normalize_citations("version 1.2.18 【W2】 and 【 S10 】, not 【note】") == \
        "version 1.2.18 [W2] and [S10], not 【note】"
