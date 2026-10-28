# Learning Guide: Multi-Agent RAG System

A study companion for Assignment 2. Read it top to bottom once the system is built.

- **Part 1: Concepts.** The ideas behind multi-agent orchestration, RAG, routing, MCP and streaming. It does not depend on our implementation.
- **Part 2: How this system works.** How we applied each concept. It is updated at the end of every build phase.
- **Part 3: Decision log.** What we chose, what we rejected, and why.
- **Part 4: Check yourself.** Questions to answer without looking.

> **Status:** Part 1 complete · Part 2 is the *planned* design and gets replaced with real file paths, model IDs and measured numbers as each phase lands · Part 3 decisions marked *proposed* await confirmation.

---

## Part 1: Concepts

### 1.1 What makes something an "agent"

An LLM call is a function: text in, text out. An **agent** puts an LLM inside a loop with **tools**. The model decides which tool to call, your code runs the tool, the result goes back to the model, and this repeats until the model gives a final answer. This is the **ReAct** loop (Reason + Act) from Assignment 1:

```
question → LLM → "call search_corpus('chunking')" → tool result → LLM → … → final answer
```

The assignment's definition is strict. An agent needs a **distinct system prompt and distinct tool access**. Four "agents" that make the same LLM call under different names count as one agent, not four.

### 1.2 Why split work across agents?

| You gain | You pay |
|---|---|
| Short, focused prompts, so the model follows instructions better | More LLM calls, so more latency and cost |
| Scoped tool access (the Data agent can't browse; Research can't run SQL) | Coordination can fail (wrong route, loops) |
| A different model per role (cheap where possible, strong where needed) | One agent's error can spread unless it is isolated |
| Pieces can be tested and evaluated independently | Shared state has to be designed deliberately |

Rule of thumb: split when roles need **different tools or different models**. Different wording alone is not a reason.

### 1.3 Orchestration patterns

**Supervisor.** One coordinator picks the next worker, and every worker reports back to it.

```
                 ┌──────── supervisor ◄───────┐
        ┌────────┼─────────┬──────────┐       │
        ▼        ▼         ▼          ▼       │
    research   code      data     summary ────┘ (all return to supervisor)
```

Every decision passes through one node, so the pattern is simple and easy to inspect. *Failure mode:* the supervisor becomes a bottleneck. Every hop costs one more LLM call, and long plans become fragile.

**Hierarchical.** Supervisors manage other supervisors. The top level sees *teams*; team leads see workers. Teams are usually built as **subgraphs**.

```
                top supervisor
               /              \
      knowledge lead      engineering lead
        /      \             /      \
   research  summary      code      data
```

This scales to many agents and keeps each choice small. *Failure mode:* a task that needs two teams has to escalate up and come back down, which adds hops.

**Network / swarm (handoffs).** There is no central boss. Agents pass control directly to each other ("transfer_to_code_agent"). There are fewer hops, but the flow is harder to reason about and termination is harder to guarantee.

**Codex worktree model.** Parallel coding agents each work in an isolated git worktree on a delegated task, and the results are merged explicitly. This fits code-change tasks, not a live chat service.

A pattern can route in two ways:
- **LLM-decided.** The supervisor model reads the state and outputs `{"next": "code", "reason": "…"}`. This is flexible, but it needs structured output and guards such as a hop limit.
- **Rule-decided.** Plain Python looks at state fields. This is deterministic, free and testable, but it can't handle nuance.

### 1.4 LangGraph in one page

LangGraph models a workflow as a **state machine over one shared state object**.

| Concept | What it is |
|---|---|
| **State** | A `TypedDict` that every node reads. Nodes return *partial updates*, not the whole state. |
| **Reducer** | Decides how an update merges into the state. The default overwrites. `operator.add` appends lists. `add_messages` merges chat messages by ID. |
| **Node** | A function `state → update`. An agent is a node, or a whole subgraph. |
| **Edge** | Decides what runs next. It can be static (`a → b`) or conditional (`a → f(state)`). A node can also return `Command(goto="code", update={…})`, which updates state *and* routes in one step. |
| **Send** | Fan-out. It runs nodes in parallel, each with its own input (map-reduce). |
| **Compile** | `graph.compile(checkpointer=…)` turns the graph into a runnable. `recursion_limit` caps the total number of steps and acts as a loop guard. |
| **Checkpointer** | Saves the state after every step under a `thread_id`. If you call again with the same `thread_id`, the conversation continues. Locally this is in memory or SQLite; in production it is an external store. |
| **Streaming** | `graph.astream(…, stream_mode=[…])`. `updates` gives each node's state change (our trace). `messages` gives LLM tokens as they are generated (the chat text). `custom` carries any event a node emits (e.g. "retrieved 8 chunks"). |

### 1.5 RAG (retrieval-augmented generation)

An LLM has never seen your documents, so it will make up answers about them. RAG fetches relevant passages first and makes the model answer **from them**.

1. **Ingest.** Load the documents and split them into **chunks**. Turn each chunk into an **embedding**: a vector (e.g. 1536 numbers) in which similar meanings end up close together.
2. **Store.** Write the vectors, text and metadata (doc id, page, section) to a **vector database**.
3. **Retrieve.** Embed the question and return the **top-k** most similar chunks by cosine similarity.
4. **Augment.** Paste those chunks into the prompt as numbered sources.
5. **Generate.** The model answers and cites sources as `[S2]`. Each citation maps back to a chunk ID.

The settings that matter:
- **Chunk size and overlap.** Chunks that are too small lose context. Chunks that are too big dilute the match and waste tokens. Splitting on structure (headings, sections) works better than blind fixed-size cuts.
- **k.** A larger k improves recall but lowers precision and costs more tokens. *Lost in the middle:* models pay less attention to material buried mid-prompt.
- **MMR (maximal marginal relevance).** Picks chunks that are relevant *and* not near-duplicates of each other.
- **Reranking.** Retrieve wide with fast vector search (k≈20), score each (query, chunk) pair with a slower, more accurate model, and keep the best 5.
- **Metadata filters.** Restrict the search, e.g. to chunks that contain code.

Pinecone vocabulary: an **index** has a fixed *dimension* and *metric*. A **namespace** partitions an index. **Upsert** with the same ID overwrites the existing record, so re-ingestion is *idempotent* when chunk IDs are deterministic.

### 1.6 LLM routing

This system makes two separate routing decisions:
- **Agent routing** decides *which agent* handles a task. This is the supervisor's job.
- **Model routing** decides *which provider and model* runs a given LLM call. This is the router's job.

Model routing can be driven by several dimensions:
- **Cost:** a cheap model for classification and short lookups.
- **Capability:** a long context window, or stronger reasoning or code.
- **Latency:** a fast model on the hot path.
- **Availability:** fall back to another provider when one rate-limits or errors. A *circuit breaker* stops sending traffic to a failing provider for a while.

**Hybrid** routing combines these dimensions. Every decision has to be logged with its reason. A router that always picks the same model is not a router.

### 1.7 MCP vs A2A

**MCP (Model Context Protocol)** connects an agent to **tools and data**.
- A *server* exposes tools (and resources and prompts). A *client* discovers the tools at runtime and calls them.
- There are two transports: **stdio**, for a local subprocess, and **Streamable HTTP**, for a remote server such as Tavily's hosted one.
- Because tools are discovered at runtime, several agents can share one server. That is **MCP tool sharing**.

**A2A (Agent2Agent)** connects **agents to other agents** that run as separate services.
- Each agent publishes an **Agent Card** (name, skills, endpoint) at a well-known URL.
- A client sends it messages that create **tasks**. A task has a lifecycle (submitted → working → completed / failed) and can stream progress over SSE.
- The wire format is JSON-RPC over HTTP.

Mnemonic: MCP is *agent → tool*. A2A is *agent → agent*, when the other agent is someone else's service.

### 1.8 Streaming with Server-Sent Events (SSE)

SSE is a one-way HTTP stream from the server to the browser. Each message is plain text and ends with a blank line:

```
event: token
data: {"text": "Retrieval"}

```

The browser's native `EventSource` only supports GET. For a POST chat endpoint, the page calls `fetch()`, reads `response.body` chunk by chunk, and splits the stream on blank lines.

- **TTFT (time to first token)** is the latency the user feels.
- **Total time** is what you pay compute for.

### 1.9 State on Cloud Run

Cloud Run starts container instances when traffic arrives (a **cold start**), adds instances under load, and kills idle ones. Two messages in the same conversation can land on different instances, and an instance can disappear between them. Process memory is therefore **not** durable. Any state that must persist lives in an external store, keyed by session or thread ID. If the vector index is external too, the container holds nothing that matters and can be replaced at any time.

### 1.10 Evaluating RAG with RAGAS

RAGAS uses an LLM as a judge. It scores each sample of (question, retrieved contexts, answer, reference answer):

| Metric | Question it answers | What it diagnoses |
|---|---|---|
| **Faithfulness** | Are the answer's claims supported by the retrieved context? | Hallucination in the generator |
| **Answer relevancy** | Does the answer address the question that was asked? | Drift or padding in the generator |
| **Context precision** | Are the relevant chunks ranked near the top? | Ranking in the retriever |
| **Context recall** | Did retrieval find everything the reference answer needs? | Coverage or chunking in the retriever |

Read the scores together:
- **Low recall with high faithfulness:** the model is honest, but retrieval missed things. Fix chunking or k.
- **High recall with low faithfulness:** the context was good, but the model wandered. Fix the prompt or the model.

LLM judges have biases. Keep the judge model fixed across runs and state which one you used.

### 1.11 Reasoning ("thinking") models and reasoning budgets

Many current models can **think before answering**: they generate hidden reasoning tokens, then the visible answer. Thinking improves hard, multi-step problems, but every hidden token costs time (and money on paid APIs), and the user sees nothing until it ends.

Our measurement on NVIDIA Lightning: with thinking on, 559 of 639 output tokens were hidden reasoning, and the answer took 6.3 s. With thinking off, it took 0.9 s, with the first token after 0.23 s. Thinking is therefore a **routing decision**, not a global setting: turn it on only when the task is hard enough to pay for it.

Thinking can also change *behavior*, not just speed. The same Lightning model picked the right tool 100% of the time with thinking on and 0% with it off. It answered from memory instead of calling the tool. Always benchmark the exact configuration you will deploy.

---

## Part 2: How this system works

> **Status:** sections 2.1–2.7 describe the built, tested system (Phases 0–5). Deployment details are added after Phase 6.

### 2.1 One chat turn, end to end (as built)

Watch this live with `python -m src.run_turn "your question"`, which prints every trace event.

1. **`start_turn`** resets the turn-level fields: `routes`, `results`, `sources`, `hops`. Their reducers treat `None` as "reset". `messages` keeps accumulating, so the conversation persists.
2. **`supervisor`** gets the conversation, the request and the results so far (full text up to 6,000 characters). It returns a `SupervisorDecision`: `next`, `task`, `complexity`, `reason`. Deterministic guards can override it (hop limit, repeated task, an agent that failed twice, a failed supervisor call). The resulting `RouteRecord` is appended to `state["routes"]`, the node returns `Command(goto=next)`, and an `agent_route` event is emitted.
3. **The worker node** (`research` / `code` / `data` / `summary`) runs `run_agent`, a bounded ReAct loop:
   - **Before each LLM call**, the model router re-checks its choice against the current context size. The Summary agent typically starts on `openai:fast` and switches to `nvidia:strong` once a 40k-token paper is in its context.
   - **Tools write their sources into the turn's shared `SourceRegistry`,** so `[S#]` (corpus) and `[W#]` (web) labels are unique across all agents in the turn.
   - **The last round withholds tools,** which forces a text answer.
4. The worker returns `Command(goto="supervisor")` with an `AgentResult`. If anything inside raises, the node catches it and returns `status="error"`. The supervisor sees the failure and routes around it.
5. **`synthesize`** produces the final answer. It makes **no extra LLM call** when exactly one agent succeeded (passthrough). With several results it merges them, keeping citation labels, on `nvidia:strong` (or `openai:strong` when complexity is high). If every agent failed, it writes an explicit error answer.
6. The checkpointer saves the state under `thread_id`. `turn_stats` records the agents, hops, providers, tokens and USD cost.

### 2.2 The agents (built)

What makes each one an agent rather than a renamed LLM call is its own prompt and its own tools:

| Agent | Tools (only this agent has this set) | Prompt focus | Typical model |
|---|---|---|---|
| Supervisor | none; structured output (`SupervisorDecision`) | Pick one next step; self-contained task; when to finish | `openai:fast` |
| Research | `search_corpus`, `web_search` + `web_extract` (Tavily MCP) | Corpus first, web for current info; read the authoritative page for exact current facts; cite every claim | `nvidia:strong`; `openai:strong` if complexity is high |
| Code | `search_code_examples` (code-only filter), `web_search` (Tavily MCP), `check_python_syntax` | At most 2 lookups; syntax-check; answer in text | `nvidia:strong` |
| Data | `list_tables`, `describe_table`, `run_sql` (read-only) | Describe, then one SELECT; show the SQL and the result | `openai:fast` |
| Summary | `find_document`, `fetch_document` (whole paper by ID prefix), `web_extract` (Tavily MCP) | Fixed TL;DR, key ideas, method, results, limits structure, with section citations | `openai:fast` to locate, then `nvidia:strong` once input > 30k tokens |

**MCP tool sharing.** Two Tavily MCP tools are each used by two agents in different contexts:
- **`web_search`:** Research (current events) and Code (current library versions).
- **`web_extract`:** Research (reading the authoritative page for an exact current fact) and Summary (summarizing a URL).

Every call emits `mcp_tool_call` with the calling agent's name.

**Why Research needs `web_extract` (found in production).** Search results are *excerpts* of pages, chosen for relevance to the query. Asked for "the latest LangGraph release", the PyPI excerpt happened to list old 2025 pre-releases, and the model confidently answered "1.0.7". The real answer, 1.2.18 from the day before, was at the top of the same page. A snippet is evidence about the part of a page you were shown, not about the whole page. For exact current facts, the agent now opens the authoritative page.

### 2.3 Model routing (built, from the Phase 0 benchmark)

Benchmark medians (3 runs; NVIDIA thinking off unless noted; files in `data/bench/`):

| Slot | Model | Routing accuracy / time | Tool accuracy / time | RAG TTFT / total |
|---|---|---|---|---|
| `openai:fast` | gpt-6-luna | 100% / 1.7 s | 100% / 2.0 s | 1.4 / 2.5 s |
| `openai:strong` | gpt-6.1-sol | 100% / 2.2 s | 100% / 2.1 s | 1.7 / 6.1 s |
| `nvidia:strong` | nemotron-3-ultra-550b-a55b | 100% / 1.3 s | 100% / 1.6 s | 1.5 / 3.5 s |
| `nvidia:fast` | nemotron-3.5-lightning-30b-a3b | 92% / 0.7 s | **0%** / 6.7 s | 0.3 / 1.3 s |
| (thinking on) `nvidia:strong` | same | 100% / 3.5 s | 100% / 8.9 s | 8.8 / 9.6 s |

The **balanced** policy, which is the default. First matching rule wins (`src/llm/router.py`):

| Call | Slot | Dimension | Why |
|---|---|---|---|
| Supervisor | openai:fast | latency | On every hop; 100% routing accuracy |
| Summary, input > 30k tokens | nvidia:strong | capability | 1M context; free where OpenAI cost scales with input |
| Summary, short input | openai:fast | cost | Cheapest reliable tool-caller |
| Research, high complexity | nvidia:strong + thinking | capability | Multi-hop questions get Nemotron's reasoning mode (free) |
| Research | nvidia:strong | cost | Free, 100% tool accuracy, 1.5 s TTFT |
| Code | nvidia:strong | cost | Strong and free |
| Data | openai:fast | capability | Precise text-to-SQL, cheapest reliable model |
| Synthesizer, high complexity | nvidia:strong + thinking | capability | Hard merges get the reasoning mode (free) |
| Synthesizer | nvidia:strong | cost | Routine merge, free |

The other policies:
- **Economy** sends everything to NVIDIA, with Lightning as the supervisor.
- **Quality** sends all agent work to `openai:strong` (`gpt-6.1-sol`), the only way that model runs. Falling back to OpenAI on an NVIDIA failure always uses `gpt-6-luna`.

Three layers adjust every decision:
1. **Availability.** A circuit breaker opens for 60 s after a 429, 5xx, timeout or dropped connection. Calls fail over to the other provider, and the trace says why. Tool-using roles never fall back to Lightning.
2. **Context window.** Slots whose window would be over 90% full are skipped.
3. **Thinking.** It turns on only for high-complexity research, code or synthesis on `nvidia:strong`.

Every decision is emitted as `llm_route` with its slot, dimension, rule and reason. Every call is emitted as `llm_call` with latency, tokens and cost.

### 2.4 RAG ingestion pipeline (built)

```
fetch_corpus ──► build_chunks ──► ingest ──► Pinecone index "agent-engineering-corpus" (namespace "corpus")
 arXiv PDFs        PDF/MDX → md        OpenAI text-embedding-3-small (1536-d)
 GitHub docs       sections → chunks   upsert, delete stale IDs, skip unchanged docs
```

- **Corpus** (`data/corpus_manifest.json`):
  - 58 arXiv papers across reasoning, agents, tool use, multi-agent, RAG, retrieval, routing and evaluation.
  - 99 docs pages: LangGraph and LangChain multi-agent docs, the MCP 2026-07-28 spec and guides, and the A2A spec and topics.
  - Each paper's title is checked against the manifest, so a typo in an ID fails loudly.
  - **Result:** 157 documents → 4,105 chunks, 1.61M tokens (43–864 tokens per chunk, median 324), 964 of them containing code. Embedding everything costs about $0.03.
- **Cleaning that mattered.** Each rule was found by inspecting outlier chunks, not guessed:
  - running headers and page numbers stripped
  - "headings" that were really code listings demoted to body text
  - figure text and reference lists dropped
  - leftovers under 20 tokens discarded

  Lesson: always look at your smallest and largest chunks before embedding anything.
- **Chunking** (`src/rag/chunking.py`): structure-aware, in two passes.
  1. Split on markdown headings into sections, keeping the heading path and page numbers.
  2. Split long sections by tokens: 800 tokens per chunk with 100 overlap. Sections under 80 tokens merge into the next one.
  - Reference lists are dropped because they attract false matches. Figure-text noise is stripped.
- **Context header**: each chunk starts with `Title — Section path`, so a generic sentence like "we set k=5" still embeds with its document's context.
- **Deterministic IDs**: `arxiv:2210.03629#0007`. This enables three things:
  - idempotent upserts
  - prefix listing (`fetch_document` rebuilds a whole paper with `index.list(prefix=…)`)
  - deleting chunks left over when a document shrinks
- **Citations carry page numbers**: chunk URLs look like `https://arxiv.org/pdf/2210.03629#page=4`, which opens the PDF at the cited page. Doc URLs are GitHub links pinned to the exact commit that was indexed.
- **Retrieval at question time** (`src/rag/retriever.py`, an LCEL chain `pinecone_search | diversify`):
  1. Embed the query.
  2. Take the top 8 chunks by cosine similarity, optionally filtered (e.g. code-only).
  3. Keep 5, with at most 3 per document so one paper can't crowd out the others.
  4. Label each kept chunk `[S#]` for the turn and inject it with the augmentation template `[S1] Title — section, p.X (relevance 0.82)` followed by the chunk text.
  - **The cap is a preference, not a filter.** It started as a hard filter. RAGAS then showed context recall of 0.33 on single-paper questions: all 8 hits came from one paper, the cap kept 3, and the abstract with the key numbers was thrown away. Now, if the cap leaves fewer than 5 chunks, the best capped-out hits fill the free slots.
- **Structured side** (`src/data/build_db.py`): SQLite tables `papers`, `authors`, `paper_authors`, `doc_pages` and `llm_models` for the Data agent. Column meanings are SQL comments inside `CREATE TABLE`, so the agent can read the schema documentation straight from `sqlite_master`.

### 2.5 File map (so far)

| Path | Role |
|---|---|
| `src/config.py` | All settings and model IDs, from env/.env |
| `src/verify_setup.py` | Phase 0 key and model-ID checks |
| `src/llm/providers.py` | Slot → LangChain chat model (OpenAI or NVIDIA) |
| `src/llm/pricing.py` | Price table, the single source of truth for cost |
| `src/llm/bench.py` | Phase 0 model benchmark |
| `src/rag/fetch_corpus.py` | Download papers and docs |
| `src/rag/chunking.py` | PDF/MDX cleaning, sectioning, chunking |
| `src/rag/build_chunks.py` | Stage 1: raw files → `chunks.jsonl` |
| `src/rag/ingest.py` | Stage 2: embed → Pinecone, with re-indexing logic |
| `src/data/build_db.py` | SQLite DB for the Data agent: 58 papers, 451 authors, 99 doc pages, 7 models |
| `src/tools/sql_tools.py` | Data agent tools. Three guards make LLM-written SQL safe: read-only connection, SQLite authorizer, single-SELECT check. |
| `src/llm/router.py` | Policy tables, `choose()`, circuit breaker, NVIDIA rate limiter, fallback, cost accounting |
| `src/trace.py` | `emit()`: trace events into LangGraph's `custom` stream (later forwarded to the browser) |
| `src/mcp_client.py` | Tavily Remote MCP client (Streamable HTTP); traced calls |
| `src/rag/retriever.py` | LCEL retrieval chain, `SourceRegistry` ([S#]/[W#] labels), augmentation template, whole-document fetch |
| `src/agents/tools.py` | Tool factories (corpus, web, documents, SQL, syntax), each traced |
| `src/agents/base.py` | `AgentSpec` and the bounded ReAct loop with per-call re-routing |
| `src/agents/specialists.py` | The four agents: prompts and tool sets |
| `src/graph/state.py` | `GraphState` schema and reset-able reducers |
| `src/graph/supervisor.py` | Supervisor decision schema, prompt, guards |
| `src/graph/build.py` | Graph assembly, worker nodes with failure isolation, synthesizer |
| `src/run_turn.py` | Terminal runner that prints the live trace |
| `src/api/app.py` | FastAPI: SSE chat endpoint, session history, startup warm-up |
| `src/persistence.py` | Checkpointer factory: memory / sqlite (local) / firestore (Cloud Run) |
| `static/` | `index.html`, `styles.css`, `app.js` (vanilla) + vendored `marked` and DOMPurify |
| `Dockerfile`, `scripts/deploy.sh` | Container image; idempotent Cloud Run deploy (Secret Manager, Firestore, least-privilege service account) |
| `src/eval/questions.json` | The fixed 30-question test set with verified reference answers |
| `src/eval/collect.py` | Runs the questions through the deployed API; records answers, contexts and timings |
| `src/eval/ragas_eval.py` | RAGAS scoring (separate venv, `requirements-eval.txt`) |
| `src/eval/cold_start.py`, `src/eval/report.py` | Cold-start probes confirmed via Cloud Run logs; the tables in `evaluation.md` |
| `tests/` | 60 tests: chunking, SQL safety, routing table (incl. "balanced never picks gpt-6.1-sol"), failover (incl. empty completions), NVIDIA structured output, agent loop (incl. uncited preambles and the sources-consulted fallback), graph failure isolation, guards (incl. empty FINISH), turn reset, retrieval diversity, citation normalization, checkpoint size bound |

### 2.6 Failure handling (what happens when things break)

| Failure | Handled by | Result |
|---|---|---|
| Provider 429 / 5xx / timeout / dropped connection | Router: circuit breaker + fallback | Same call retried once on the other provider; `llm_error` + `llm_route(availability)` in trace |
| NVIDIA nearing its ~40 requests/min | Local sliding-window limiter (35/min) | Waits up to 5 s, otherwise routes to OpenAI |
| A tool raises | Agent loop | Error text goes back to the model as the tool result; the agent continues |
| Model never stops calling tools / answers empty | Agent loop | Final round withholds tools, forcing a text answer |
| A whole agent fails or times out (120 s) | Worker node `try/except` | `AgentResult(status="error")`; supervisor routes around it |
| Supervisor LLM call fails | Supervisor guard | Research on the first hop, FINISH afterwards |
| Supervisor loops | Guards | Repeated (agent, task) → FINISH; hop limit 6 → FINISH; agent failed twice → FINISH |
| Every agent failed | Synthesizer | Explicit apology naming what failed. The turn still completes. |
| The stream to the browser drops or stalls | UI: 45 s watchdog (3 missed heartbeats) + recovery | The answer is loaded from the saved conversation (`GET /api/sessions/{id}`) and labeled as recovered |
| A request arrives while startup work is still running | Warm-up is awaited in `lifespan` | The port opens only after warm-up, so requests can't race it (this caused a real `_DeadlockError`) |

### 2.7 The web layer (built)

```
browser ──POST /api/chat {query, session_id, policy}──► FastAPI
        ◄──── text/event-stream ───────────────────────  graph.astream(stream_mode=["custom", "messages"])
              event: trace     ← every emit() in any node    (custom stream)
              event: token     ← LLM tokens, tagged by node  (messages stream, AIMessageChunk only)
              event: citation  ← sources cited in the final answer
              event: done      ← final answer + turn_stats + TTFT/total timing
              : ping           ← heartbeat every 15 s so proxies keep the stream open
```

- **The agents contain no streaming code.** They call `ainvoke`. Because the graph is streamed in `messages` mode, LangGraph attaches a streaming callback, LangChain switches the call to the provider's streaming API, and every token surfaces tagged with its node (`research`, `synthesize`, …). This is why the router could stay a plain request/response API.
- **The browser** (`static/app.js`, vanilla JS) reads the stream with `fetch()` + `ReadableStream`, because `EventSource` can only do GET. It splits frames on blank lines and dispatches on `event:`.
  - **Routing chips** are built from `agent_route` (which agent) plus `llm_route` (which model, which dimension). The full reason shows on hover.
  - **The Trace tab** lists every event; click a row to see its JSON payload. **The Routing tab** shows both routing levels as tables.
  - **Citations**: `[S#]`/`[W#]` labels become clickable markers that open the source, landing on the exact PDF page.
- **Reloading the page** calls `GET /api/sessions/{id}`, which reads the conversation back from the checkpointer. That is the visible proof of durable state.
- **Two lessons from building it:**
  - **`messages` mode re-emits whole messages.** It sends any `AIMessage` a node writes to state, not just LLM tokens, so filter on `AIMessageChunk`.
  - **Providers' chunk shapes differ.** With tools bound, `langchain-openai` uses the Responses API, whose chunk `content` is a list of typed blocks, not a string. Always read `.text`.

### 2.8 Deployment on Cloud Run (built)

```
                       ┌──────────── Cloud Run service (us-east4, 0–3 instances) ────────────┐
browser ──HTTPS/SSE──► │ container: uvicorn → FastAPI → LangGraph graph (stateless process)   │
                       └──┬──────────────┬───────────────┬──────────────┬──────────────┬──────┘
                          │ checkpoints  │ API keys      │ vectors      │ LLM calls    │ MCP
                          ▼              ▼               ▼              ▼              ▼
                      Firestore    Secret Manager    Pinecone     OpenAI, NVIDIA   Tavily
```

- **The container holds nothing that matters.** Code and a 228 KB read-only SQLite file are baked in. Vectors live in Pinecone, conversations in Firestore, and keys in Secret Manager. Cloud Run can kill or replace an instance at any moment and nothing is lost.
- **How durable conversation state works.**
  - The browser's `session_id` is LangGraph's `thread_id`.
  - After every graph step (a *super-step*), the checkpointer writes the full state as one Firestore document under that thread: about 7 documents per turn.
  - On the next turn, whichever instance receives it loads the thread's latest checkpoint, so `messages` (the conversation) come back.
  - **Verified:** two turns were run on revision 1, and revision 2 was then deployed, so new instances took all the traffic. Revision 2 restored both turns and answered a follow-up that only made sense with that memory.
- **Secrets never touch the image.**
  - `deploy.sh` pipes each key from `.env` into Secret Manager.
  - Cloud Run injects them as environment variables at startup.
  - The runtime runs as a dedicated service account allowed to do exactly two things: read/write Firestore and read those four secrets.
- **Cold start.** A new instance has to:
  1. Start the container.
  2. Import LangChain and the provider SDKs.
  3. Open the Pinecone, embeddings and MCP clients. This is the "warm-up", 2.7 s.

  Only then does it accept requests. `--cpu-boost` gives extra CPU during startup, and `min-instances 0` means you pay nothing while idle, at the price of that cold start.
- **Lessons from the first deploy:**
  - **A brand-new project's owner role takes a minute to reach every service.** The first `artifacts repositories create` was refused, and re-running the idempotent script fixed it.
  - **Cloud Run reserves URL paths ending in `z`.** `/healthz` returned Google's own 404 page and never reached the app, so the endpoint is now `/health`.
  - **Don't run startup work concurrently with requests.**
    - The warm-up ran as a background task, importing the `openai` SDK's lazily loaded modules in a worker thread.
    - The first request imported the same modules on the event loop, and Python's import lock raised `_DeadlockError`.
    - Awaiting the warm-up inside `lifespan` fixes it, because uvicorn only opens the port after startup completes.
  - **Firestore documents are capped at 1 MiB, and a checkpoint is one document.** Sources keep at most 4,000 characters in state; the model still reads the full text in the tool result.
  - **Networks drop streams.** In 1 of 5 browser tests the browser never received the end of a stream that the server had finished. The UI now detects a stalled or truncated stream and loads the answer from the saved conversation. Durable state turns out to be useful for more than multi-turn memory.

### 2.9 Evaluation (built): what measuring taught us

Full numbers are in [evaluation.md](evaluation.md). This section is about *method*: the lessons that transfer to any agent system.

- **Write the test set from the corpus, then verify it.**
  - The 30 reference answers were checked fact by fact against the chunk text. Two of my drafts were wrong.
  - An unverified reference makes context recall measure your memory, not your retriever.
- **Read the four RAGAS metrics as a diagnosis, not a grade:**

  | Pattern | Diagnosis | Where to look |
  |---|---|---|
  | Low recall, high precision | Retrieval returns good chunks but not enough of them | k, caps, query phrasing |
  | Low faithfulness, high recall | The evidence was there but the model added its own facts | The prompt, or the model |
  | Low relevancy | The answer drifted from the question | Rarely the retriever |

  Our v1 run showed the first two patterns and pointed straight at their causes: a hard per-document cap, and the model filling gaps from memory.
- **Measure, fix, re-measure, and compare like with like.**
  - After the fixes, a provider timeout happened to send 7 questions to a stronger model.
  - Comparing all 30 questions would have credited the fix with the model's quality, so the before/after uses only the 21 questions the same model answered both times.
  - Likewise, configurations are compared on the questions scored in *all* of them.
- **Agents add variance.** The same question can get one search or three, so per-question scores move a lot between runs. Trust aggregates, and look at individual traces to understand *why*, not *how much*.
- **The trace is an evaluation instrument.** Several real bugs never produced an HTTP error, because guards absorbed them; they showed up only as `llm_error` events in the trace:
  - the NVIDIA supervisor failing on every call;
  - the OpenAI supervisor's length limit;
  - empty completions from NIM.

  If you only look at final answers, a well-guarded system hides its own failures.
- **What an orchestrator can see decides what it does.**
  - The supervisor saw an 800-character preview of each result. When the key number was past the cut, it re-ran Research, or sent the answer to Summary, to "finish" it: up to 4 agents and 224 s on a one-part question.
  - This was the small NVIDIA Lightning supervisor. My OpenAI credit had run out, so the router had silently failed over to it, and my test script ignored the error events, so at first I blamed the wrong model. **Lesson: a test harness must show the trace, or it can quietly measure a different model than you think.**
  - A prompt rule ("a COMPLETE result is the full answer") and a stricter decision schema both failed. Showing the full result fixed it: 40 of 40 replayed decisions finished.
  - Before you prompt harder, ask whether the model even has the information to follow the rule.
- **Know what your timer measures.**
  - TTFT first counted a bare `"\n"` that Nemotron streams before a tool call, about 2 s earlier than the real answer.
  - The Mac's sleep paused Python's monotonic clock, so a dead connection looked like a 10-second request.
  - Every latency number needs a definition, and a sanity check against an independent clock (here, Cloud Run's request logs).
- **LLM judges have biases too.** The judge (`gpt-6.1-sol`) is also the OpenAI-only configuration's generator. *Judging LLM-as-a-Judge*, a paper in our own corpus, calls this self-enhancement bias. Report it next to the numbers instead of hiding it.
- **Evaluation costs money.** RAGAS judging cost about 15× more than generating the answers it judged ($10 vs $0.67), because every retrieved chunk is a separate judge call on a reasoning model.

---

## Part 3: Decision log

| Decision | Chosen | Rejected alternatives | Why | Status |
|---|---|---|---|---|
| Orchestration | Supervisor, hand-built `StateGraph` + `Command` | Hierarchical; prebuilt `create_supervisor`; Codex worktrees | 4 workers fit a flat supervisor. Hierarchical adds LLM hops (latency and rate-limit pressure). Building it by hand keeps routing in inspectable state fields and isolates each worker's errors. | built |
| LLM providers | NVIDIA NIM + OpenAI | — | Your choice. NVIDIA (free) does most agent work; OpenAI (`gpt-6-luna`) handles latency-critical structured output, SQL, short lookups and failover. | built |
| OpenAI model in the balanced policy | `gpt-6-luna` only; `gpt-6.1-sol` just for the Quality policy | `gpt-6.1-sol` for hard research and failover | The prepaid credit ran out during evaluation (mostly the RAGAS judge). luna matched sol on tool and routing accuracy in the benchmark at 1/20 the price, and hard tasks get Nemotron's free thinking mode instead | changed after evaluation |
| Routing style | LLM classifies the task (complexity); deterministic rules pick the model | Fully LLM-decided routing | Rules are testable and explainable. The LLM only supplies the judgment rules can't make. | built |
| NVIDIA thinking mode | On only for high-complexity research, code or synthesis | Always on (default) / always off | Benchmark: thinking on is 5–10x slower with no accuracy gain on routine work | built |
| Lightning (`nvidia:fast`) | Supervisor in economy mode and supervisor fallback only | Tool-using roles | 0% tool selection with thinking off | built |
| Single-agent answers | Passthrough, no synthesis call | Always synthesize | Saves one LLM call (seconds and cost) on most turns | built |
| Worker tasks | Self-contained task written by the supervisor | Pass the whole conversation | Workers stay stateless and cheap; follow-ups are resolved once, by the supervisor | built |
| Vector store | Pinecone serverless | Chroma embedded | Syllabus default. An external index keeps the container stateless. | built |
| Embeddings | OpenAI `text-embedding-3-small` | NVIDIA / local embeddings | Reference model; $0.03 for the whole corpus | built |
| Retrieval | k=8 → prefer ≤3 chunks per document (backfill when fewer than 5 remain) → top 5 | MMR; reranker; hard per-document cap | Simple and explainable diversity. The hard cap cost recall on single-paper questions (RAGAS). | built, tuned by RAGAS |
| Tool sharing (10%) | MCP (Tavily shared by Research, Code and Summary) | A2A | MCP is already required for live search. A2A adds protocol surface for the same credit. | built |
| Conversation state | Firestore checkpointer | Stateless; GCS FUSE; Cloud SQL | Serverless and free tier; reference canonical path | built, verified across revisions |
| Exact current facts | Search, then `web_extract` the authoritative page | Answer from search excerpts | Excerpts are partial pages; trusting one gave a wrong "latest version" in production | built |
| Startup warm-up | Awaited before the port opens | Background task | The background version raced the first request (`_DeadlockError`); a slightly later "ready" is the safer trade | built |
| Dropped streams | UI watchdog + recovery from the saved conversation | Retry the query; show an error | Retrying re-runs (and re-bills) the agents; the server usually finished and saved the answer anyway | built |
| Supervisor model (balanced) | `gpt-6-luna` | Lightning (free) | In evaluation, the Lightning supervisor ran extra agents after Research had already answered on 13/30 questions, doubling latency; luna finished correctly every time for about $0.0002 per hop | confirmed by evaluation |
| Supervisor's view of results | Full text up to 6,000 characters | 800-character preview + "COMPLETE" label | The preview made it re-run agents to finish answers that were already complete (9/24 vs 40/40 correct finishes in replay); the extra input costs about $0.0001 per hop | built (after evaluation) |
| Empty completions | Treated as a retryable provider error | Let the agent loop handle it | NIM sometimes returns 200 OK with nothing in it; without an exception the router can't fail over and the turn fails | built (after evaluation) |
| RAGAS judge | `gpt-6.1-sol`, temperature bypassed, telemetry off | Cheaper judge | Matches the reference judge family; self-enhancement bias is disclosed | built |

---

## Part 4: Check yourself

1. What is the difference between *agent routing* and *model routing* here, and which component owns each?
2. The supervisor returns `Command(goto=…)` instead of using a conditional edge. What does that buy us?
3. NVIDIA returns HTTP 429 in the middle of a turn. Trace exactly what happens next and what the user sees.
4. Context recall is 0.55 and faithfulness is 0.95. What do you change first, and why?
5. Why can't chat history live in a Python dict on Cloud Run?
6. At what point would you switch this system to the hierarchical pattern?
7. Why is MCP tool sharing a better fit than A2A for this system, and when would A2A be the right call?
8. Why do deterministic chunk IDs matter when re-running ingestion?
9. The Summary agent started on `openai:fast` and finished on `nvidia:strong` within one task. What triggered the switch, and why is that a good routing decision?
10. Lightning was 100% accurate at picking tools with thinking on and 0% with it off. What does that teach you about benchmarking models?
11. Why does the synthesizer skip its LLM call when only one agent ran? What would you lose if it always ran?
12. The supervisor kept re-running agents, or adding a summary, after a complete answer. A "COMPLETE" label in the prompt didn't fix it; showing the full result did. What does that teach about prompting versus information design for an orchestrator?
13. How would you *prove* that conversation state survives instance replacement, not just consecutive requests to one process? (Hint: how we used revision 2.)
14. Why does the UI recover a dropped answer from `GET /api/sessions/{id}` instead of simply re-sending the question?
15. A RAG system scores context precision 0.95 and context recall 0.6. Name two plausible causes and the first experiment you would run.
16. Why was the v1 → v2 improvement measured on 21 questions instead of all 30? What would the 30-question comparison have overstated?
17. Three bugs in this project raised no HTTP error at all. How were they found, and what does that imply about how you should monitor an agent system?
