"""Turn raw corpus files into retrieval chunks.

Strategy (structure-aware, two pass):
  1. Convert each source to markdown and split it on headings into sections,
     remembering the heading path ("3 Method > 3.2 Prompting") and, for PDFs, the page.
  2. Sections that fit within CHUNK_TOKENS become one chunk; longer ones are split by
     a token-based recursive splitter with CHUNK_OVERLAP. Tiny sections are merged forward.

Every chunk's text starts with a short header (document title and section path), so its
embedding carries document context even when the body is generic ("we set k=5 …").
Reference lists are dropped: they are dense with paper titles and attract false matches.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100
MIN_SECTION_TOKENS = 80  # smaller sections are merged into the next one
MIN_BODY_TOKENS = 20  # split remainders smaller than this carry nothing retrievable

_ENC = tiktoken.get_encoding("cl100k_base")  # tokenizer of text-embedding-3-*


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TOKENS,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=n_tokens,
    separators=["\n```", "\n\n", "\n", ". ", " ", ""],
    keep_separator=True,
)


@dataclass
class Chunk:
    id: str
    doc_id: str
    chunk_index: int
    text: str
    title: str
    section: str
    source_type: str  # "paper" | "docs"
    source: str  # "arxiv" | "langgraph" | "mcp" | "a2a"
    url: str
    n_tokens: int
    has_code: bool
    topic: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    year: int | None = None
    authors: str | None = None

    def metadata(self) -> dict:
        """Pinecone metadata: no None values allowed."""
        return {k: v for k, v in asdict(self).items() if v is not None and k != "id"}


@dataclass
class Section:
    path: list[str]
    text: str
    page_start: int | None = None
    page_end: int | None = None
    lines: list[str] = field(default_factory=list)


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_REFERENCES = re.compile(r"^(references|bibliography|reference)$", re.I)
_CAPTION = re.compile(r"^(figure|fig\.|table|algorithm)\s*\d", re.I)  # captions misdetected as headings
_PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)
_INLINE_HTML = re.compile(r"</?(sup|sub|br|span|b|i|em|strong)\b[^>]*>", re.I)


def _strip_md(text: str) -> str:
    return re.sub(r"[*_`]", "", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _too_long_for_heading(text: str) -> bool:
    # PDF layout detection occasionally promotes a code listing or paragraph to a heading
    return len(text) > 120 or len(text.split()) > 15


_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")


def clean_pdf_markdown(md: str) -> str:
    """Drop layout-mode figure-text blocks (garbled axis labels) and inline HTML tags."""
    md = _PICTURE_TEXT.sub("", md)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    return _INLINE_HTML.sub("", md)


def strip_running_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Remove page numbers and running headers/footers (short lines repeated on many pages)."""
    counts: dict[str, int] = {}
    for _, md in pages:
        for line in {l.strip() for l in md.splitlines() if 0 < len(l.strip()) < 100}:
            counts[line] = counts.get(line, 0) + 1
    threshold = max(3, int(0.3 * len(pages)))
    running = {line for line, n in counts.items() if n >= threshold and not _HEADING.match(line)}
    return [
        (page_no, "\n".join(l for l in md.splitlines() if l.strip() not in running and not _PAGE_NUMBER.match(l)))
        for page_no, md in pages
    ]


def split_sections(pages: list[tuple[int | None, str]], doc_title: str = "") -> list[Section]:
    """Split (page_number, markdown) pages into heading-delimited sections.

    A heading that repeats the document title is not added to section paths (every chunk
    header already carries the title), and figure/table captions are kept as body text.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    current = Section(path=[], text="")
    skipping_refs = False
    title_key = _norm(doc_title)[:25]

    def flush():
        current.text = "\n".join(current.lines).strip()
        if current.text:
            sections.append(current)

    for page_no, md in pages:
        for line in md.splitlines():
            m = _HEADING.match(line)
            if m and (_CAPTION.match(_strip_md(m.group(2))) or _too_long_for_heading(m.group(2))):
                line, m = _strip_md(m.group(2)), None
            if m and title_key and _norm(m.group(2))[:25] == title_key:
                continue
            if m:
                level, heading = len(m.group(1)), _strip_md(m.group(2))
                flush()
                skipping_refs = bool(_REFERENCES.match(heading))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
                current = Section(path=[h for _, h in stack], text="", page_start=page_no, page_end=page_no)
                continue
            if skipping_refs:
                continue
            if current.page_start is None:
                current.page_start = page_no
            if line.strip():
                current.page_end = page_no
            current.lines.append(line)
    flush()
    return sections


def _with_heading(s: Section) -> str:
    return f"## {s.path[-1]}\n{s.text}" if s.path else s.text


def _merge_small(sections: list[Section]) -> list[Section]:
    """Fold sections under MIN_SECTION_TOKENS into the following section, keeping their headings."""
    merged: list[Section] = []
    carry: Section | None = None
    for s in sections:
        if carry is not None:
            s.text = f"{_with_heading(carry)}\n\n{_with_heading(s)}"
            s.page_start = carry.page_start or s.page_start
            carry = None
        if n_tokens(s.text) < MIN_SECTION_TOKENS:
            carry = s
            continue
        merged.append(s)
    if carry is not None:
        if merged:
            merged[-1].text += "\n\n" + _with_heading(carry)
            merged[-1].page_end = carry.page_end or merged[-1].page_end
        else:
            merged.append(carry)
    return merged


def chunk_document(doc: dict, pages: list[tuple[int | None, str]]) -> list[Chunk]:
    """doc carries doc_id, title, source_type, source, url and optional topic/year/authors."""
    chunks: list[Chunk] = []
    for section in _merge_small(split_sections(pages, doc["title"])):
        section_path = " > ".join(section.path) if section.path else "Introduction"
        header = f"{doc['title']} — {section_path}\n\n"
        for body in _SPLITTER.split_text(section.text):
            if n_tokens(body) < MIN_BODY_TOKENS:  # "Continued on next page", lone captions
                continue
            text = header + body.strip()
            i = len(chunks)
            chunks.append(Chunk(
                id=f"{doc['doc_id']}#{i:04d}",
                doc_id=doc["doc_id"],
                chunk_index=i,
                text=text,
                title=doc["title"],
                section=section_path,
                source_type=doc["source_type"],
                source=doc["source"],
                url=doc["url"] + (f"#page={section.page_start}" if section.page_start else ""),
                n_tokens=n_tokens(text),
                has_code="```" in body,
                topic=doc.get("topic"),
                page_start=section.page_start,
                page_end=section.page_end,
                year=doc.get("year"),
                authors=doc.get("authors"),
            ))
    return chunks


# ---------------------------------------------------------------- PDF (papers)

def pdf_pages(path: Path) -> list[tuple[int, str]]:
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    return strip_running_lines([(p["metadata"]["page_number"], clean_pdf_markdown(p["text"])) for p in pages])


# ---------------------------------------------------------------- MDX / MD (docs)

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)
_LANG_BLOCK = re.compile(r"^:::(python|js)\s*\n(.*?)^:::\s*$", re.S | re.M)
_JSX_LINE = re.compile(r"^\s*</?[A-Z][A-Za-z0-9.]*(\s[^>]*)?/?>\s*$")
_JSX_INLINE = re.compile(r"</?[A-Z][A-Za-z0-9.]*(\s[^>]*)?/?>")
_MDX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.S)
_IMPORT_EXPORT = re.compile(r"^(import|export)\s.*$", re.M)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_ANCHOR = re.compile(r'^\s*<a\s+(id|name)="[^"]*"\s*/?>(</a>)?\s*$', re.M)
_MD_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")  # [text](url) → text (images untouched)


def clean_mdx(raw: str) -> tuple[str, dict]:
    front = {}
    m = _FRONTMATTER.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                front[k.strip()] = v.strip().strip("\"'")
        raw = raw[m.end():]
    raw = _LANG_BLOCK.sub(lambda b: b.group(2) if b.group(1) == "python" else "", raw)
    raw = _MDX_COMMENT.sub("", raw)
    raw = _HTML_COMMENT.sub("", raw)
    raw = _IMPORT_EXPORT.sub("", raw)
    raw = _ANCHOR.sub("", raw)
    out, in_fence = [], False
    for line in raw.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            if _JSX_LINE.match(line):
                continue
            line = _MD_LINK.sub(r"\1", _JSX_INLINE.sub("", line))
        out.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text, front


def write_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def read_jsonl(path: Path) -> list[Chunk]:
    with path.open() as f:
        return [Chunk(**json.loads(line)) for line in f]
