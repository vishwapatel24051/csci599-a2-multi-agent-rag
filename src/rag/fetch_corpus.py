"""Download the corpus listed in data/corpus_manifest.json.

Papers: metadata from each arXiv abstract page's `citation_*` meta tags (each title is
checked against the manifest's `expect` fragment so a mistyped ID fails loudly), PDFs
from arxiv.org. (The export.arxiv.org API was rate-limiting/timing out; the abs and pdf
pages are CDN-served and reliable.)
Docs: individual files read from shallow git clones; the commit SHA is recorded so
every citation URL points at the exact text that was indexed.

Idempotent: anything already on disk is skipped.

Usage: python -m src.rag.fetch_corpus
"""

import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PAPERS_DIR = RAW / "papers"
DOCS_DIR = RAW / "docs"
REPOS_DIR = DOCS_DIR / "_repos"
MANIFEST = DATA / "corpus_manifest.json"

HEADERS = {"User-Agent": "csci599-assignment2-corpus-fetch/1.0"}
ARXIV_DELAY_S = 3.0  # arXiv asks automated clients to wait 3 s between requests
MAX_RETRIES = 6
_META_TAG = re.compile(r'<meta name="(citation_[a-z_]+)" content="([^"]*)"')
_PRIMARY_SUBJECT = re.compile(r'class="primary-subject">[^<(]*\(([^)]+)\)')


def _clean(text: str) -> str:
    return " ".join(html.unescape(text).split())


def _get_with_retry(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    for attempt in range(MAX_RETRIES):
        wait = ARXIV_DELAY_S * 2 ** (attempt + 1)
        try:
            resp = client.get(url, **kwargs)
        except httpx.TransportError as e:
            print(f"  {type(e).__name__} for {url}, retrying in {wait:.0f}s")
            time.sleep(wait)
            continue
        if resp.status_code not in (429, 500, 502, 503, 504):
            resp.raise_for_status()
            return resp
        wait = float(resp.headers.get("retry-after", wait))
        print(f"  {resp.status_code} for {url}, retrying in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url} after {MAX_RETRIES} attempts")


def fetch_abs_metadata(client: httpx.Client, arxiv_id: str) -> dict:
    page = _get_with_retry(client, f"https://arxiv.org/abs/{arxiv_id}", timeout=60, follow_redirects=True).text
    tags: dict[str, list[str]] = {}
    for name, value in _META_TAG.findall(page):
        tags.setdefault(name, []).append(_clean(value))
    subject = _PRIMARY_SUBJECT.search(page)
    # citation_author is "Last, First"; store "First Last"
    authors = [" ".join(reversed([p.strip() for p in a.split(",", 1)])) for a in tags.get("citation_author", [])]
    return {
        "arxiv_id": arxiv_id,
        "title": tags["citation_title"][0],
        "abstract": tags.get("citation_abstract", [""])[0],
        "authors": authors,
        "published": tags["citation_date"][0].replace("/", "-"),
        "primary_category": subject.group(1) if subject else None,
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def fetch_papers(client: httpx.Client, papers: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Metadata + PDF for each paper; returns (meta by id, list of problems)."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = PAPERS_DIR / "meta.json"
    meta: dict[str, dict] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    problems = []
    for i, p in enumerate(papers, 1):
        arxiv_id = p["arxiv_id"]
        if arxiv_id not in meta:
            meta[arxiv_id] = fetch_abs_metadata(client, arxiv_id)
            time.sleep(ARXIV_DELAY_S)
        m = meta[arxiv_id]
        m["topic"] = p["topic"]
        if p["expect"].lower() not in m["title"].lower():
            problems.append(f"{arxiv_id}: expected '{p['expect']}', got '{m['title']}'")
            del meta[arxiv_id]
            continue

        dest = PAPERS_DIR / f"{arxiv_id}.pdf"
        if not (dest.exists() and dest.stat().st_size > 10_000):
            resp = _get_with_retry(client, m["pdf_url"], timeout=120, follow_redirects=True)
            if not resp.content.startswith(b"%PDF"):
                problems.append(f"{arxiv_id}: response is not a PDF")
                continue
            dest.write_bytes(resp.content)
            time.sleep(ARXIV_DELAY_S)
        print(f"  [{i}/{len(papers)}] {arxiv_id}  {m['title'][:80]}")
        meta_path.write_text(json.dumps(meta, indent=2))
    return meta, problems


def _git(repo_dir: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo_dir), *args], check=True, capture_output=True, text=True).stdout


def fetch_docs(doc_sources: list[dict]) -> list[dict]:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for src in doc_sources:
        repo_dir = REPOS_DIR / src["repo"].replace("/", "__")
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
                 f"https://github.com/{src['repo']}", str(repo_dir)],
                check=True,
            )
        sha = _git(repo_dir, "rev-parse", "HEAD").strip()
        for path in src["files"]:
            dest = DOCS_DIR / src["source"] / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(_git(repo_dir, "show", f"HEAD:{path}"), encoding="utf-8")
            records.append({
                "source": src["source"],
                "repo": src["repo"],
                "commit": sha,
                "path": path,
                "local_path": str(dest.relative_to(ROOT)),
                "url": f"https://github.com/{src['repo']}/blob/{sha}/{path}",
            })
        print(f"  {src['source']}: {len(src['files'])} files @ {sha[:10]}")
    return records


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    papers = manifest["papers"]

    print("Fetching docs…")
    records = fetch_docs(manifest["docs"])
    (DOCS_DIR / "meta.json").write_text(json.dumps(records, indent=2))

    print(f"Fetching {len(papers)} papers (metadata + PDF)…")
    with httpx.Client(headers=HEADERS) as client:
        meta, problems = fetch_papers(client, papers)

    print(f"Done: {len(meta)} papers, {len(records)} doc pages.")
    if problems:
        print("Problems:\n  " + "\n  ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
