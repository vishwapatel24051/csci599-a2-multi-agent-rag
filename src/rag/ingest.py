"""Stage 2 of ingestion: data/processed/chunks.jsonl → OpenAI embeddings → Pinecone.

Re-indexing logic:
  * Chunk IDs are deterministic (`doc_id#0007`), so an upsert overwrites in place.
  * A per-document content hash (data/processed/index_state.json) skips documents whose
    chunks haven't changed since the last run, so re-runs cost nothing.
  * When a document now has fewer chunks than before, the leftover IDs are deleted
    (found by listing the `doc_id#` prefix).

Usage:
  python -m src.rag.ingest            # incremental
  python -m src.rag.ingest --force    # re-embed everything
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from src.config import get_settings
from src.rag.build_chunks import CHUNKS_PATH, PROCESSED
from src.rag.chunking import Chunk, read_jsonl

STATE_PATH = PROCESSED / "index_state.json"
EMBED_BATCH = 96
UPSERT_BATCH = 96


def _doc_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()


def ensure_index(pc: Pinecone, name: str, dim: int, cloud: str, region: str):
    if not pc.has_index(name):
        print(f"creating serverless index '{name}' ({dim}-d, cosine, {cloud}/{region})")
        pc.create_index(name=name, dimension=dim, metric="cosine", spec=ServerlessSpec(cloud=cloud, region=region))
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
    return pc.Index(name)


def _with_retry(fn, *args, attempts: int = 5, **kwargs):
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — transient network/rate-limit errors
            if i == attempts - 1:
                raise
            wait = 2 ** (i + 1)
            print(f"  {type(e).__name__}: retrying in {wait}s")
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-embed every document")
    args = parser.parse_args()

    s = get_settings()
    if not (s.openai_api_key and s.pinecone_api_key):
        raise SystemExit("OPENAI_API_KEY and PINECONE_API_KEY must be set in .env")

    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in read_jsonl(CHUNKS_PATH):
        by_doc[c.doc_id].append(c)

    state: dict[str, dict] = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    todo = {d: cs for d, cs in by_doc.items() if args.force or state.get(d, {}).get("hash") != _doc_hash(cs)}
    print(f"{len(by_doc)} documents, {len(todo)} new or changed")
    if not todo:
        return

    pc = Pinecone(api_key=s.pinecone_api_key.get_secret_value())
    index = ensure_index(pc, s.pinecone_index, s.embedding_dim, s.pinecone_cloud, s.pinecone_region)
    embedder = OpenAIEmbeddings(model=s.openai_embedding_model, api_key=s.openai_api_key.get_secret_value())

    embedded_tokens = 0
    for n, (doc_id, chunks) in enumerate(todo.items(), 1):
        vectors = []
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            values = _with_retry(embedder.embed_documents, [c.text for c in batch])
            vectors += [{"id": c.id, "values": v, "metadata": c.metadata()} for c, v in zip(batch, values)]
            embedded_tokens += sum(c.n_tokens for c in batch)
        for i in range(0, len(vectors), UPSERT_BATCH):
            _with_retry(index.upsert, vectors=vectors[i : i + UPSERT_BATCH], namespace=s.pinecone_namespace)

        new_ids = {c.id for c in chunks}
        stale = [i for page in index.list(prefix=f"{doc_id}#", namespace=s.pinecone_namespace) for i in page
                 if i not in new_ids]
        if stale:
            index.delete(ids=stale, namespace=s.pinecone_namespace)

        state[doc_id] = {"hash": _doc_hash(chunks), "n_chunks": len(chunks)}
        STATE_PATH.write_text(json.dumps(state, indent=2))
        print(f"  [{n}/{len(todo)}] {doc_id}: {len(chunks)} chunks" + (f", deleted {len(stale)} stale" if stale else ""))

    print(f"embedded {embedded_tokens:,} tokens (≈ ${embedded_tokens / 1e6 * 0.02:.3f})")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
