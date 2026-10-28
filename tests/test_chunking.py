from src.rag.chunking import chunk_document, clean_mdx, clean_pdf_markdown, split_sections

DOC = {"doc_id": "arxiv:0000.00000", "title": "A Study of Agents", "source_type": "paper", "source": "arxiv",
       "url": "https://arxiv.org/pdf/0000.00000"}
FILLER = "word " * 120  # ~120 tokens, above MIN_SECTION_TOKENS


def test_sections_track_heading_path_and_pages():
    pages = [(1, f"# A Study of Agents\n## 1 Intro\n{FILLER}"), (2, f"### 1.1 Setup\n{FILLER}")]
    sections = split_sections(pages, DOC["title"])
    assert [s.path for s in sections] == [["1 Intro"], ["1 Intro", "1.1 Setup"]]  # title H1 dropped
    assert (sections[1].page_start, sections[1].page_end) == (2, 2)


def test_references_are_dropped_until_next_heading():
    pages = [(1, f"## Results\n{FILLER}\n## References\n[1] Someone. A paper.\n## A Appendix\n{FILLER}")]
    texts = " ".join(s.text for s in split_sections(pages))
    assert "Someone" not in texts and "Appendix" not in texts  # appendix heading kept in path, not text
    assert any(s.path == ["A Appendix"] for s in split_sections(pages))


def test_captions_are_not_headings():
    pages = [(1, f"## 2 Method\n{FILLER}\n## Figure 3: Scaling results\n{FILLER}")]
    sections = split_sections(pages)
    assert len(sections) == 1 and "Figure 3: Scaling results" in sections[0].text


def test_chunk_ids_are_deterministic_and_carry_context_header():
    pages = [(3, f"## 2 Method\n{FILLER}")]
    a, b = chunk_document(DOC, pages), chunk_document(DOC, pages)
    assert [c.id for c in a] == [c.id for c in b] == ["arxiv:0000.00000#0000"]
    assert a[0].text.startswith("A Study of Agents — 2 Method")
    assert a[0].url.endswith("#page=3")
    assert None not in a[0].metadata().values()


def test_small_sections_merge_forward_keeping_heading():
    pages = [(1, f"## Tiny\nshort text\n## Big\n{FILLER}")]
    chunks = chunk_document(DOC, pages)
    assert len(chunks) == 1 and "## Tiny" in chunks[0].text and "short text" in chunks[0].text


def test_pdf_markdown_cleanup():
    md = "x<sup>1</sup> <!-- Start of picture text -->axis 0.1 0.2<!-- End of picture text --> y"
    assert clean_pdf_markdown(md) == "x1  y"


def test_clean_mdx_keeps_python_drops_js_and_jsx():
    raw = (
        '---\ntitle: Persistence\n---\nimport X from "y"\n<a id="anchor" />\n<Tip>\nRead [the docs](/oss/x).\n</Tip>\n'
        ":::python\n```python\nprint('py')\n```\n:::\n:::js\n```js\nconsole.log('js')\n```\n:::\n"
    )
    text, front = clean_mdx(raw)
    assert front["title"] == "Persistence"
    assert "print('py')" in text and "console.log" not in text
    assert "<Tip>" not in text and "import X" not in text and "anchor" not in text
    assert "Read the docs." in text


def test_code_listing_promoted_to_heading_is_body_text():
    listing = "## Step 2: refine " + "agent = LLMAgentBase(['x'], 'Agent') " * 10
    pages = [(1, f"## 3 Method\n{FILLER}\n{listing}\n{FILLER}")]
    sections = split_sections(pages)
    assert [s.path for s in sections] == [["3 Method"]] and "LLMAgentBase" in sections[0].text


def test_running_headers_and_page_numbers_removed():
    from src.rag.chunking import strip_running_lines
    pages = [(i, f"Published as a conference paper at ICLR 2023\n## S{i}\nbody {i}\n{i}") for i in range(1, 6)]
    cleaned = strip_running_lines(pages)
    joined = "\n".join(md for _, md in cleaned)
    assert "Published as a conference paper" not in joined
    assert "body 3" in joined and "## S3" in joined and "\n3" not in joined.replace("## S3", "")
