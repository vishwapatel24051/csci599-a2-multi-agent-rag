# Build notes

A factual record of what happened during development: what broke, why, and what we changed. It is raw material for `PROCESS_LOG.md`, which you write in your own words.

## 2026-10-26: Phase 0/1 (scaffold, corpus)

- **RAGAS 0.4.3 does not import** next to LangChain 1.x. It imports `langchain_community.chat_models.vertexai`, which `langchain-community` 0.4.x removed, and the community releases that still have it require `langchain-core` < 1.0.
  - Decision: evaluation runs in a separate virtualenv (`requirements-eval.txt`). It scores answers and contexts that the app produces over HTTP, which also means it evaluates the real production path.
- **`pymupdf4llm` 2.0 installs as an empty package.** In PyMuPDF 2.x the 4llm code moved into `pymupdf` itself. Inspected the package metadata before trusting it, then installed `pymupdf==2.0`, which provides `pymupdf4llm.to_markdown`.
- **`ChatNVIDIA` defaults to `max_tokens=1024`.** That would silently truncate long summaries, so the cap must be set explicitly per call.
- **arXiv's export API returned 429, then timed out,** even for a single 58-ID query and then for batches of 10. Semantic Scholar's keyless pool also returned 429.
  - Fix: read metadata from each paper's arXiv abstract page (`citation_*` meta tags, CDN-served), with retry and backoff that honors `Retry-After`. Citation counts are best effort and cached.
- **PDF → markdown quirks found on the ReAct paper:**
  - The paper's title appeared as an H1 and prefixed every section path. Dropped by fuzzy-matching it to the doc title.
  - A figure caption was detected as a heading. Captions are now kept as body text.
  - Layout mode emits `<!-- picture text -->` blocks of garbled axis labels, which produced a 33-token junk chunk. Stripped.
- **MDX docs** contain JSX components, `:::python` / `:::js` language blocks, anchors and relative links. The cleaner keeps the Python variants and code fences and drops the rest.
- **First full build:** 4,154 chunks, but the outlier check found two problems.
  - **A code listing in AFlow (2410.10762) was detected as a heading.** That put a 450-token "section path" into every chunk header in its appendix. Fix: text over 120 characters or 15 words is never treated as a heading.
  - **15 tiny chunks were page furniture** ("21 / Published as a conference paper at ICLR 2023"). Fix: strip bare page numbers, plus short lines that repeat on at least 30% of a paper's pages.
  - **After the fixes:** 4,105 chunks, 43–864 tokens each, with a 67-token header ceiling. Every fix has a regression test in `tests/test_chunking.py`.
- `src/oss/langchain/retrieval.mdx` turned out to be a 96-character stub, so it was removed from the manifest (99 docs pages).
- Semantic Scholar answered on a later run, so all 58 papers have citation counts.
- **SQL tool safety, verified by tests:** a `WITH … DELETE` statement gets past the keyword check but is denied by the SQLite authorizer, and a runaway recursive CTE is aborted by the progress-handler budget.

## 2026-10-26: Phases 0, 2, 3, 4 (keys, router, agents, supervisor graph)

- **Phase 0 verification: 17/17 checks passed.**
  - All 4 configured model IDs exist and support tool calling and structured output.
  - Tavily's MCP tools are named `tavily_search` and `tavily_extract` (underscores; the docs show hyphens), so the names are read from the live tool list.
- **NVIDIA "thinking" mode is on by default, and it dominated latency.**
  - With it on, Lightning spent 559 of 639 output tokens on hidden reasoning (6.3 s). Nemotron Ultra's RAG time to first token was 8.8 s, and one tool call took 33 s.
  - With `chat_template_kwargs.enable_thinking=false`: Lightning answered in 0.9 s (0.23 s to first token); Ultra in 1.5 s to first token with 100% tool accuracy.
  - This became a router decision: thinking turns on only for high-complexity work on `nvidia:strong`.
- **Lightning with thinking off picks 0% of tools correctly.** Given a bare prompt, it answers from memory or says "Let me look at the data" but never calls a tool. Its structured routing is still 92% correct at 0.7 s.
  - Consequence: Lightning is never used for tool-calling roles, and fallbacks for tool-using roles skip it (`_fallback` in `router.py`).
- **`nvidia/nemotron-3-super-120b-a12b` is listed by `/models` but returns 404** "Function … Not found for account". Being in the catalog doesn't mean the model is deployed for your key; only a real call proves it.
- **Supervisor bug: re-running agents to "finish" truncated previews.** The supervisor saw worker outputs cut at 800 characters, decided a complete summary was "cut off", and ran an extra Research hop (+11 s, +$0.01).
  - Fix: previews are labeled `status: ok, COMPLETE (N chars; showing a preview)`, and a prompt rule says not to re-run agents to finish a preview.
- **Code-agent bug: no text answer.**
  - Nemotron spent 4 rounds on near-duplicate searches. On the budget-exhausted round it put the finished code into another `check_python_syntax` call instead of answering, so the agent produced nothing.
  - **The supervisor handled the failure correctly** by re-routing to Research. That's the failure isolation working live.
  - Fixes: the final round withholds tools (verified both providers accept tool history with no tools bound); an empty answer jumps straight to the forced-answer round; the Code prompt caps searches at 2. Tests are in `tests/test_agent_loop.py`.
- **The supervisor's OpenAI call stalled about 8 minutes, then raised `httpx.ReadError`.**
  - The guard finished the turn with the results it had, but the router hadn't failed over, because `ReadError` wasn't classified as retryable. Connection and timeout errors are now retryable (tested).
  - The 120 s `asyncio.wait_for` never fired. The likely cause is laptop sleep: macOS's monotonic clock pauses during sleep. Cloud Run doesn't sleep, but the failover fix covers the real failure mode.
- **Live runs (`python -m src.run_turn`) that worked:**
  - Research over the corpus: 1 search, 3 page-level citations, $0.00025, both providers used in one turn.
  - Data → Summary chain: the Summary agent was re-routed mid-run from `openai:fast` to `nvidia:strong` when `fetch_document` grew its context to 42k tokens.
  - A follow-up turn resolved "instead" from conversation memory.
  - Live web question: `tavily_search` via MCP, citing PyPI and GitHub releases.
  - Code agent after the fix: 2 searches, a syntax check, 44 s.

## 2026-10-26: Phase 5 (FastAPI + SSE + UI)

- **Token streaming needs no streaming code in the agents.** Agents call `ainvoke`. When the graph is streamed with `stream_mode="messages"`, LangGraph attaches a streaming callback, so LangChain switches those calls to the streaming API. The API layer forwards the tokens as SSE `token` frames tagged with the producing node.
- **Bug: the final answer appeared twice in the UI.** `messages` mode also re-emits whole `AIMessage`s that nodes write to state (the synthesizer's final answer). Fix: forward only `AIMessageChunk`s.
- **Bug: OpenAI-backed agents never streamed (0 token frames), while NVIDIA-backed agents did.**
  - Cause: with tools bound, `langchain-openai` switches to OpenAI's Responses API (`POST /v1/responses`), whose chunks carry `content` as a list of typed blocks. The supervisor's structured-output call still uses `/v1/chat/completions`, with string content. The API filtered on `isinstance(chunk.content, str)`, which silently dropped every Responses-API token.
  - Fix: use the provider-neutral `chunk.text`. Found by running the graph directly vs through the API generator, tallying chunk types by node, and spotting `/v1/responses` in the httpx log.
- **UI verified in headless Chromium (Playwright):**
  - routing chips per hop
  - 13-event trace for a single-agent turn
  - 8 citation markers resolving to 3 sources
  - history restored from the checkpointer after a page reload
  - no horizontal overflow at 390 px
  - zero console errors
  - multi-agent turn: data (openai) → summary (re-routed to nemotron-ultra) → synthesizer (nemotron-ultra), 26 citations

## 2026-10-27: Phase 6 (Cloud Run deployment)

- **Deployed** to project `csci599-a2-617204` (us-east4) with `scripts/deploy.sh`. Live URL: https://multi-agent-rag-157472143408.us-east4.run.app. The Cloud Build image build took 1 min 32 s.
- **First run stopped at `artifacts repositories create` with `PERMISSION_DENIED`,** although the account is project owner. The project was a few minutes old and the owner role hadn't propagated yet. Re-running the idempotent script worked.
- **`/healthz` returned Google's HTML 404** (not FastAPI's JSON). Cloud Run's front end reserves paths ending in `z`. Renamed to `/health`.
- **First live request: `_DeadlockError("deadlock detected by _ModuleLock('openai.resources.embeddings')")` in the supervisor.**
  - Cause: the startup warm-up ran as a background task, importing the openai SDK's lazily loaded modules in a worker thread, while the first request imported them on the event loop.
  - The guard degraded gracefully: supervisor error → Research, and the answer was still correct with 3 ReAct citations.
  - Fix: await the warm-up in `lifespan`. Uvicorn opens the port only after startup, so no request can race it. Startup log: warm-up 2.7 s, then "Application startup complete".
- **Wrong live-web answer.** "Latest LangGraph on PyPI" came back as 1.0.7 (Jan 2026), but 1.2.18 was released Oct 26, 2026, and it's the version this project pins.
  - Cause: Tavily search returns relevance-chosen *excerpts*, and for that query the PyPI excerpt listed old 2025 pre-releases. A differently worded query got the correct top rows.
  - Fix: Research gets `tavily_extract` (capped at 20k characters) and a prompt rule that excerpts are partial pages, so for exact current facts it reads the authoritative page.
  - Result: correct answer (1.2.18, Oct 26, 2026) in 20.5 s, versus 52 s for the wrong one.
- **Found while reviewing the fix: Firestore caps documents at 1 MiB, and a checkpoint is one document.** The source registry kept full text in state, and one extracted PyPI page is 103k characters. State now keeps at most 4,000 characters per source (test added; 48 tests pass).
- **Durable state verified across instance replacement.**
  - Session `durable-check-001` ran 2 turns on revision `00001`: 14 checkpoint documents in Firestore.
  - After deploying revision `00002` (100% of traffic), `GET /api/sessions/durable-check-001` returned both turns.
  - The follow-up "what was my first question about, and which version did your second answer report?" was answered correctly from memory: direct mode, 7.5 s, no worker.
- **Browser test (Playwright) against the live URL.**
  - One run never finished in the browser, although Cloud Run logged the turn complete (200, 24.9 s).
  - The next 4 runs passed. Restoring that session rendered fine, so it wasn't a rendering bug; it looks like a transport-level stream drop.
  - Fix: the UI isolates per-frame errors, aborts after 45 s of silence (3 missed heartbeats), and recovers the answer from the saved conversation. A simulated truncated stream recovered in 19.5 s.
- **Observed live timings (warm):**

  | Turn | Time to first token | Total |
  |---|---|---|
  | Corpus RAG turn | 3.2–4.0 s | 25–26 s |
  | Live-web turn | 4.3 s | 20.5 s |
  | Memory-only follow-up | — | 7.5 s |

  Each costs $0.0002–0.0003.

## 2026-10-27: Phase 7 (evaluation)

- **Test set: `src/eval/questions.json`, 30 questions across 10 topics** (agents, reasoning, tool use, RAG, retrieval, routing, multi-agent, evaluation, LangGraph, MCP/A2A).
  - Includes 3 cross-document comparisons.
  - I drafted each reference answer, then **checked every key fact against the chunk text** with a regex fact-check script.
  - Two drafts were wrong and got fixed:
    - MAST's first category is "system design issues", not "specification issues".
    - The MCP docs list tools, resources and prompts as the server's three building blocks.
- **The RAGAS environment is separate** (`requirements-eval.txt`). ragas 0.4.3 (latest) imports `langchain_community.chat_models.vertexai`, which langchain-community 0.4.2 removed. Pinning 0.4.1 fixes the import. ragas also needs openai 1.x, while the app pins 3.x.
- **Judge `gpt-6.1-sol` rejected every RAGAS call:** "temperature does not support 0.01 with this model". It's a reasoning model that only accepts its default temperature. Fix: `LangchainLLMWrapper(..., bypass_temperature=True)`. Also `bypass_n=True`, because answer relevancy asked for 3 generations with `n=3` and got 1.
- **`ragas.evaluate()` hung for ~15 min after its last API call.** The debug log ended at `ragas._analytics Flushing`: ragas sends usage telemetry by default, and the flush blocked the process. With `RAGAS_DO_NOT_TRACK=true` the same run finishes in 43 s.
- **Judge cost is about $0.07 per question:** 15k input + 4k output tokens (mostly reasoning), so about $6.50 for 3 configurations × 30 questions.
- **The smoke test found a routing-boundary issue.** "In the Tree of Thoughts paper, how did GPT-4 … compare … on the Game of 24?" went to the **Summary** agent, which loads the whole paper ($0.002, no retrieval step). Rule 7 now sends specific questions about a named document to Research; Summary is only for "summarize / overview" requests. Checked on 5 prompts (2 fact questions → research, 2 summary requests → summary, 1 counting question → data).
- **First collection run: 7 of 23 requests failed with `ReadError` / one HTTP 500.**
  - The Cloud Run logs showed requests arriving in bursts 15 minutes apart, the instance shutting down while idle, and one request hitting "no available instance" during that scale-down.
  - `pmset -g log` explained it: **the Mac went to sleep** at 12:34 and only briefly woke at 12:50 and 13:04 (the bursts). Python's `perf_counter` pauses during sleep, which is why the failed requests showed only ~10 s elapsed.
  - Fix: rerun everything under `caffeinate -dimsu`. Lesson: check the client before blaming the server.
- **Cold start measured by accident in the smoke test.** The service had scaled to zero; per the logs, 5.8 s went to container start and imports, then 9.8 s to app startup (Firestore client, graph, 2.2 s warm-up). First byte arrived at 16 s, TTFT was 20.4 s, against ~3.7 s server-side TTFT when warm.
- **v1 RAGAS (auto mode, revision 00004, 29/30 scored):** faithfulness 0.757, answer relevancy 0.853, context precision 0.959, context recall 0.776. Judge cost $1.84. `moa` was skipped: Research answered it from web search only, with no corpus retrieval to score.
- **Diagnosis of the weak questions** (recall 0.33 on reflexion, raptor, mast, selfrag_vs_crag, rewoo_vs_llmcompiler):
  - All 8 hits came from the same paper, so the hard per-document cap (3) left only 3 chunks. For Reflexion those were two appendix prompt examples and the method section; the abstract with "91% pass@1 … GPT-4 80%" ranked 4th–8th and was dropped.
  - Nemotron then filled the gap from memory with "GPT-4 baseline 67%" (the GPT-4 tech report figure) and **cited [S3] for it**. That's exactly what faithfulness catches.
- **Fixes:**
  1. The cap is now a preference, with backfill: if fewer than 5 chunks survive it, the best capped-out hits fill the free slots (3 tests).
  2. The citation rules say every number, result and name must come from the sources, never from memory, and never cite a source for something it doesn't say.
  3. The Research prompt says to search again with more specific terms when a requested detail is missing.

  Local check on Reflexion: 5 chunks on the first search, then a second, targeted search found §4.3. Answer: "91.0% pass@1, surpassing the previous SOTA of 80.1% (GPT-4)", correct and cited.
- **Re-collection:** the partial OpenAI-only run was stopped, and all three modes are being re-collected on the new revision, so the provider comparison runs on identical code. v1 is kept as the "before" baseline.
- **v2 collection (revision 00005) exposed three more bugs, all visible only in the trace's `llm_error` events.** No HTTP errors occurred, because the guards absorbed them:
  1. **NVIDIA supervisor: 60 of 60 calls failed** with `NotImplementedError: include_raw=True is not implemented`.
     - The router used `with_structured_output(schema, include_raw=True)` to keep token usage; `ChatNVIDIA` doesn't support that option. The Phase 0 benchmark called it without `include_raw`, so it didn't catch this.
     - **This also broke the Economy policy in the live UI** (Lightning supervisor): every turn silently fell back to the guard (research → finish).
     - Fix: NVIDIA structured calls take the parsed object alone, with usage estimated (the model is free). Regression test added; live check on 5 prompts with Lightning: 0 errors, 4/5 routed correctly ("Summarize the RAPTOR paper" → research, which fits Lightning's weaker routing score).
  2. **OpenAI supervisor `LengthFinishReasonError` on 7 of ~120 hops.** `max_tokens=600` includes gpt-6-luna's hidden reasoning (successful decisions: median 109, p90 327, max 510 tokens). Raised to 2000. On those hops the guard made the same decision the LLM made everywhere else (research, then finish), so those answers are unaffected.
  3. **NVIDIA "research produced no answer" (2 of 30, forced mode):** NIM returned **empty completions**, 200 OK with 0 tokens in/out in ~0.3 s, twice in a row. With no exception raised, the router couldn't fail over. Fix: an empty completion (no text, no tool calls, 0 output tokens) raises a retryable `EmptyCompletionError`, so auto mode fails over to OpenAI (test added). **Deployed after the measurements**, so the evaluation numbers are pre-fix.
- **Availability routing showed up in the v2 auto run.** At 20:23:51 one Nemotron call hit the 120 s timeout. The circuit breaker opened, and the next 7 Research calls within the 60 s cooldown went to gpt-6.1-sol with `dimension: availability`. Because of this, the v1 → v2 comparison is restricted to the 21 questions Nemotron answered in both runs: faithfulness 0.724 → 0.810, context recall 0.722 → 0.810.
- **Run-to-run variance matters.** `rewoo_vs_llmcompiler` dropped to recall 0.0 in v2 because Nemotron ran one broad search instead of three targeted ones (the agent decides how many searches to run). Per-question deltas are noisy; aggregates over 30 questions are the signal.
- **Cold starts (3 probes on revision 00006, each confirmed by a "Starting new instance" log line):**

  | Probe | First byte | Container + imports | App startup | TTFT | Total |
  |---|---|---|---|---|---|
  | 1 | 20.6 s | 7.3 s | 12.4 s | 29.3 s | 39 s |
  | 2 | 8.3 s | 3.4 s | 4.3 s | 24.4 s | 83 s |
  | 3 | 7.8 s | 3.2 s | 4.2 s | 20.8 s | 224 s |

  The first probe (after the longest idle) was about 2.5× slower to start; the likely cause is the image having to be pulled onto a fresh host.
- **Probes 2 and 3 revealed supervisor over-orchestration.** A one-part question ("Lost in the Middle") ran research → **code ("write Python to chart the U-curve")** → research again ("the preview cuts off") → summary: 224 s instead of ~15 s.
  - A replay harness re-ran the supervisor's decision after Research had answered, using 8 real answers × 3–5 tries. Results:
    - original prompt: only 9/24 FINISH (13 → summary "to summarize the findings");
    - with a new rule 8 ("answers reach the user as written; never post-process, never add unrequested work"): 23/40;
    - with a schema field `remaining` (parts still uncovered) plus a guard: 20/40, worse, and the model now said "the preview cuts off before the pass@1 value".
  - **Root cause: the supervisor saw only an 800-character preview of each result.** When the key fact sat past the cut, it rationally concluded the answer was incomplete. The "COMPLETE" label from Phase 4 treated the symptom, not the cause.
  - **Fix:** show each result in full up to 6,000 characters (answers: median 1.7k, p90 3k, max 7.5k). Replay: **40/40 FINISH**. Multi-part requests still chain correctly (data → summary; research → code). Cost: ~$0.0001 more per hop on gpt-6-luna.
  - The same cause probably explains the Lightning supervisor's 13/30 extra-agent turns in NVIDIA-only mode.
  - The `remaining` experiment was reverted (it made things worse); rule 8 stays.
- **TTFT measurement bug:** Nemotron streams a bare `"\n"` alongside its tool call, about 2 s before the real answer, and the API counted it as the first token. TTFT now starts at the first non-whitespace text (API and collector). The collector also records phase times (routed, agent start, first retrieval, first answer token). Latency was re-measured in a dedicated pass on revision 00007.
- **Correction (found later the same day): the OpenAI credit ran out at 21:09 UTC.** From then on every OpenAI call failed with `429 "You have no credits remaining"`, and the router failed over to NVIDIA.
  - So **the "over-orchestration" in cold probes 2–3 and in all the replay experiments was the NVIDIA Lightning supervisor (after failover), not gpt-6-luna.** That matches the NVIDIA-only run (13/30 extra-agent turns) and the luna-supervised v2 runs (30/30 finished after research).
  - My replay script used a no-op `emit`, so it never showed the `llm_error`s. Lesson: a test harness must surface the trace, or it can silently measure a different model.
  - The visibility fix is still real: it took Lightning from 9/24 to 40/40 correct finishes, which matters for the Economy policy and for failover.
  - The latency pass run after 22:10 UTC is invalid: corpus search needs OpenAI embeddings, so retrieval failed. All RAGAS scoring finished before 20:40 UTC and is unaffected.
- **Security: the Tavily API key appeared in Cloud Run logs.** httpx logs each request URL at INFO, and the MCP URL carries the key as a query parameter. Fix: httpx logs at WARNING. The existing log entries stay readable only to the project owner; rotating the Tavily key after grading invalidates them.
- **$5 OpenAI top-up (~16:10 PDT): calls kept failing with "no credits remaining" until 16:27,** even though one test call at 16:18 succeeded. The top-up took about 15 minutes to propagate through OpenAI's systems. One supervisor call in the latency pass hit it and failed over correctly.
- **Routing changed to gpt-6-luna-first (revision 00009).** In the balanced policy, high-complexity research and synthesis use Nemotron Ultra with thinking mode, and every fallback to OpenAI uses gpt-6-luna. gpt-6.1-sol runs only under the Quality policy. A test enforces that the balanced policy never selects it, for any role, complexity or provider outage.
- **Final warm latency (revision 00009, corrected TTFT), p50:** first byte 0.1 s → supervisor decided 1.7 s → first retrieval done 3.6 s → first answer token 7.3 s → done 16.7 s (auto). NVIDIA-only: 6.1 s / 15.4 s, and **28/30 single-agent turns** (the other 2 were guard fallbacks after a Lightning `guided_json` 400). The supervisor-visibility fix holds in production.
- **MCP-sharing evidence** (live, revision 00009): one turn had research → `tavily_search`×3 + `tavily_extract`, then code → `tavily_search`. The same tool, two agents, different purposes.
- **Packaging:** `scripts/package.sh` builds `dist/csci599-a2-<date>.zip` (no `.env`, no venvs, no raw corpus or handout files, no discarded runs; it refuses to ship anything that looks like an API key) and the `submission/` folder (live URL + the 5 files Brightspace asks for) as `submission.zip`.
- **Hands-on testing by me (the user) found problems the automated runs missed**, so I added a rubric-driven self-test: 13 question types + memory ×3 against the live URL, about 100 automatic checks. Fixed (revisions 00010–00012):
  1. **Citation brackets:** Nemotron sometimes writes `【W2】` instead of `[W2]` (2 of 30 answers in the latency pass, which nobody had looked at), so markers weren't clickable and the citations panel stayed empty. Agent outputs and syntheses now normalize to `[W2]`, and the UI accepts both while streaming.
  2. **Wrong "latest version":** the agent read pypistats.org, which lags PyPI by a day. The Research prompt now says to web_extract `pypi.org/project/<package>/` itself.
  3. **Preamble as answer:** with Nemotron's thinking mode (the new high-complexity route), Research replied "Let me search for more…" with no tool call, and the loop accepted it. A reply that cites nothing after searching now triggers the forced, tool-less answer round (tests added).
  4. **Empty FINISH:** for "what was my first question?", the supervisor sometimes chose FINISH without filling `direct_answer` (2 of 5 tries), so the user got "Sorry…". The schema now requires it; one retry happens with the reason, and otherwise the guard routes to research (tests added).
  - Final self-test on revision 00012: **0 failed checks**; browser test passed. 59 offline tests.
  - Lesson: aggregate metrics (RAGAS, latency) don't look at what a user actually sees. A short, assertion-based end-to-end test over every question type catches presentation and edge-case bugs the metrics can't.
- **Final rubric audit (revisions 00013–00015):**
  - **Provider in every chip:** chips now show `agent · model (Provider) · dimension`, so §2.6's "which LLM provider was routed to" is visible without opening a tab.
  - **README deploy section:** rewritten as a fresh-project sequence (fill `.env` → build the Pinecone index first → `gcloud auth` + billing ID → deploy with your own project ID → verify), because a TA with a fresh Pinecone account would otherwise deploy an empty index.
  - **NVIDIA stalled twice** for the full 120 s timeout before failover (economy turn: 228 s). Successful LLM calls max out at 51 s (NVIDIA) and 13 s (OpenAI) over 1,004 calls, so a 60 s per-call timeout (`llm_call_timeout_s`) halves the stall without cutting off real answers.
  - **One Self-RAG answer was uncited** even after the forced round (2 reproductions cited fine). Now it gets a "Sources consulted: [S…]" line listing what the agent read.
  - **Final self-test: 0 failed checks; browser test passed.** It caught a live availability failover (`research · gpt-6-luna (OpenAI) · availability`) while NVIDIA's Lightning endpoint was congested (26–34 s for a one-word reply).
