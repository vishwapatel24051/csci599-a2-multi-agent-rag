# Evaluation

All numbers below were measured on **2026-10-27 against the deployed Cloud Run service** (`us-east4`), through the same `POST /api/chat` SSE endpoint the UI uses. Every figure can be reproduced with the scripts in [`src/eval/`](../src/eval/).

Revisions used:

| Measurement | Revision |
|---|---|
| Answer quality, auto and OpenAI-only (RAGAS) | `00005` |
| Answer quality, NVIDIA-only | `00006` (fixed NVIDIA supervisor) |
| Cold starts | `00006` (3 probes) and `00009` (final, 1 probe) |
| Warm latency | `00009` (final, with corrected TTFT measurement) |

Fixes found during evaluation are described where they occurred.

**Routing change after evaluation.** The OpenAI credit ran out during evaluation, mostly spent on the RAGAS judge. So the balanced policy was changed to use **only `gpt-6-luna`** on the OpenAI side:
- High-complexity research and synthesis now use Nemotron Ultra with thinking mode instead of `gpt-6.1-sol`.
- Fallbacks to OpenAI now use `gpt-6-luna`.
- `gpt-6.1-sol` remains available through the Quality policy.

In the scored auto run this would have affected only the 7 questions that failed over to `gpt-6.1-sol` (§2.2). No question was rated high-complexity.

**Fixes after evaluation** (revisions `00010`–`00012`, found by hands-on testing and a rubric-driven end-to-end self-test):
- `【W2】`-style citation brackets are normalized to `[W2]`.
- Package-version questions read PyPI itself.
- An uncited "Let me search…" preamble after searching is no longer accepted as an answer.
- FINISH with an empty `direct_answer` is retried.
- LLM calls time out after 60 s instead of 120 s, so an NVIDIA stall fails over twice as fast.
- An answer that is still uncited lists its "Sources consulted".
- Routing chips show the provider.

The final self-test on revision `00015` (13 question types + memory ×3, about 100 assertions) passes with 0 failures.

## Summary

Default configuration (balanced routing), 30-question test set:

| | Result |
|---|---|
| **RAGAS** | **faithfulness 0.842 · answer relevancy 0.839 · context precision 0.938 · context recall 0.851** |
| Warm latency, p50 (p90) | first trace event shown **0.1 s**; first answer token **7.3 s** (17.0 s); complete answer **16.7 s** (30.3 s) |
| Latency breakdown, p50 | retrieval 0.3 s · LLM calls 14.3 s · web (MCP) 0 s (1.7 s p90, when used) · server overhead 0.2 s |
| Cold start (min-instances 0) | Final revision: **6.6 s** to the first byte, **12.2 s** to the first answer token, **15.4 s** total. Across all 4 probes: 6.6–20.6 s first byte. A cold start never failed a request. |
| Cost per question | **≈ $0.0014** typical: LLM $0.0003 (only the supervisor uses OpenAI) + Cloud Run ≤ $0.0008 + Tavily, embeddings, Firestore |
| Providers compared | OpenAI-only had the best faithfulness (0.90) at about 60× the LLM cost (16× the total cost per question). NVIDIA-only costs $0 for the LLM but grounds less reliably. Auto keeps most of OpenAI's quality at near-zero cost. See §5. |

## 1. Method

### Test set
[`src/eval/questions.json`](../src/eval/questions.json) has **30 fixed questions** whose answers are in the indexed corpus:

| Topic | Count |
|---|---|
| RAG | 5 |
| Multi-agent | 7 |
| Routing | 3 |
| Retrieval | 2 |
| Agents | 3 |
| Reasoning | 2 |
| Tool use | 1 |
| Evaluation | 2 |
| LangGraph docs | 2 |
| MCP / A2A docs | 3 |

They mix three kinds of question:
- **Specific facts:** "what did ToT score on the Game of 24?"
- **Mechanisms:** "how does HyDE retrieve without labels?"
- **Cross-document comparisons:** ReWOO vs LLMCompiler, Self-RAG vs CRAG.

Each question lists the document(s) that contain its answer and a **reference answer**. Every key fact in each reference was checked against the corpus chunk text with a regex fact-check script before use, so context recall measures retrieval rather than the corpus's coverage.

### Configurations compared
| Configuration | What runs where |
|---|---|
| **auto** (balanced policy) | What users get. Supervisor on `gpt-6-luna` (OpenAI); Research on `nemotron-3-ultra` (NVIDIA); synthesis on Ultra. At evaluation time, high-complexity tasks and NVIDIA failovers went to `gpt-6.1-sol`; both now use luna or Nemotron (see the routing-change note above). |
| **OpenAI only** | `force_provider=openai`: supervisor and Data agent on `gpt-6-luna`, every other role on `gpt-6.1-sol`. |
| **NVIDIA NIM only** | `force_provider=nvidia`: supervisor and Data agent on `nemotron-3.5-lightning`, every other role on `nemotron-3-ultra` (thinking off). |

Forced runs disable cross-provider fallback, so a provider's failures show up as failures instead of being hidden by the other provider.

### Collection ([`collect.py`](../src/eval/collect.py))
Each question runs in a **fresh session** (no conversation history), 3 at a time (2 for NVIDIA-only, to stay under the free tier's ~40 requests/min). For each turn it records:
- the final answer;
- the **retrieved contexts**: every chunk returned by `search_corpus` in the turn, in retrieval order, deduplicated across searches. This is what the generating agent actually saw.
- client-side time to first byte, time to first answer token (TTFT) and total time;
- server-side timings;
- a latency breakdown summed from the trace events: `rag_retrieval` (query embedding + Pinecone + diversification), `llm_call` and `mcp_tool_call`;
- token counts and LLM cost (tokens × the price table in [`src/llm/pricing.py`](../src/llm/pricing.py)).

### RAGAS ([`ragas_eval.py`](../src/eval/ragas_eval.py), ragas 0.4.3)
| Metric | What it measures | How |
|---|---|---|
| **Faithfulness** | Are the answer's claims supported by the retrieved contexts? | The judge splits the answer into statements and checks each against the contexts; score = supported / total. |
| **Answer relevancy** | Does the answer address the question? | The judge generates 3 questions from the answer; score = mean cosine similarity to the real question (`text-embedding-3-small`). |
| **Context precision** (with reference) | Are the useful chunks ranked first? | The judge marks each retrieved chunk useful or not for the reference answer; score = mean precision@k over the useful positions. |
| **Context recall** | Did retrieval find everything the reference needs? | The judge checks each reference sentence for support in the contexts; score = supported / total. |

- **Judge:** `gpt-6.1-sol`, the reference configuration's judge family. It is a reasoning model that only runs at its default temperature, so repeated runs vary slightly.
- **Bias caveat:** the judge is also the OpenAI-only configuration's generator, which can give that configuration a small self-preference advantage ("self-enhancement bias", from the corpus's own *Judging LLM-as-a-Judge* paper).

## 2. Answer quality (RAGAS)

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall | Scored | Judge cost |
|---|---|---|---|---|---|---|
| **auto (balanced routing)** | **0.842** | **0.839** | **0.938** | **0.851** | 29/30 | $2.04 |
| OpenAI only | 0.896 | 0.827 | 0.921 | 0.885 | 29/30 | $2.10 |
| NVIDIA NIM only | 0.783 | 0.840 | 0.907 | 0.872 | 26/30 | $2.40 |
| *auto, v1 baseline (before the fixes in §2.1)* | *0.757* | *0.853* | *0.959* | *0.776* | 29/30 | $1.84 |

A question is scored only when its answer used corpus retrieval, so that RAGAS has contexts to judge. The skipped questions differ by configuration (see §2.2), so here are the same metrics over the **25 questions scored in every configuration**:

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| auto (balanced routing) | 0.839 | 0.843 | 0.942 | 0.827 |
| OpenAI only | 0.900 | 0.835 | 0.915 | 0.893 |
| NVIDIA NIM only | 0.782 | 0.836 | 0.912 | 0.867 |
| auto, v1 baseline | 0.747 | 0.852 | 0.976 | 0.740 |

What the numbers say:
- **Retrieval is precise:** context precision is 0.91–0.98 everywhere, so the chunks that come back are almost always the useful ones, ranked first.
- **Recall is good:** 0.83–0.89. Misses come from how agents search, not from the index; see §2.1 and the per-question table in the appendix.
- **Faithfulness separates the generators.** `gpt-6.1-sol` (0.90) stays closest to its sources. Nemotron Ultra (0.78–0.84) more often adds a detail from its own memory, for example a baseline number, which RAGAS correctly counts as unsupported.
- **Answer relevancy is flat** (0.83–0.85) across all configurations. Every generator answers the question that was asked.

### 2.1 What RAGAS changed in the system (v1 → v2)
The first full run (v1) exposed two faults; the fixes shipped in revision `00005`:

1. **The per-document cap threw away the answer.**
   - Retrieval keeps at most 3 chunks per document out of the top 8, so one paper can't crowd out the others.
   - For a question about a single paper, all 8 hits come from that paper, so only 3 chunks survived. For *Reflexion*, those were two appendix prompt examples and the method section; the abstract with "91% pass@1 … GPT-4 80%" was discarded. Five questions had context recall 0.33.
   - **Fix:** the cap became a preference. When it leaves fewer than 5 chunks, the best capped-out hits fill the free slots (unit-tested).
2. **The model filled gaps from memory and cited a source for it.** With the abstract missing, Nemotron answered "GPT-4 baseline 67%" (the GPT-4 technical report's figure) and attached `[S3]`, which says no such thing.
   - **Fix:** the citation rules now require every number, result and name to come from the sources, and never to cite a source for something it doesn't say.
   - The Research prompt now also says to search again, with more specific terms, for a requested detail that is missing.
   - After the fix, the same question ran a second, targeted search, found §4.3, and answered "91.0% pass@1 vs 80.1%", correctly cited.

To isolate these fixes from a model change, compare only the **21 questions Nemotron answered in both v1 and v2**. In v2, a provider timeout sent 7 questions to `gpt-6.1-sol`; see §2.2.

| Metric | v1 | v2 |
|---|---|---|
| Faithfulness | 0.724 | **0.810** |
| Context recall | 0.722 | **0.810** |

Context precision dipped slightly (0.976 → 0.942 on the common subset), which is expected: backfilled chunks rank lower by construction.

Single questions are noisy because the agent decides how many searches to run. `rewoo_vs_llmcompiler` dropped from recall 0.33 to 0.00 in v2 because Nemotron ran one broad search instead of three targeted ones, and the key numbers sit in the two abstracts. The aggregates over 30 questions are the signal.

### 2.2 How answers were grounded
| Configuration | Corpus retrieval | Web only | No sources | Extra agents after Research | Supervisor guard fallbacks |
|---|---|---|---|---|---|
| auto (balanced routing) | 29 | 1 | 0 | 0 | 4 |
| OpenAI only | 29 | 1 | 0 | 0 | 3 |
| NVIDIA NIM only | 26 | 3 | 1 | 13 | 1 |

- **Web only:** the agent answered from Tavily instead of the corpus, which the prompt says to search first. The answers were correct but cited the web: Mixtral (auto), ReAct (OpenAI), and Mixtral, More-Agents and MoA (NVIDIA).
- **No sources:** in NVIDIA-only mode, `mcp_server_features` was answered from memory with no tool call. The content was right, but it's ungrounded, the one real grounding failure in 90 answers.
- **Extra agents:** with Lightning as supervisor, 13 of 30 turns ran more agents after Research had already answered (usually a redundant Summary). The OpenAI supervisor finished after Research every time.
  - **Root cause:** the supervisor saw only an 800-character preview of each result. When the key fact was past the cut, the weaker model kept going to "complete" an answer that was already complete.
  - **Fix:** it now sees each result in full, up to 6,000 characters. In a replay of real post-research decisions, correct finishes went from 9/24 to 40/40.
  - In the final latency pass (revision `00009`), NVIDIA-only turns were single-agent 28/30 times (the other 2 were guard fallbacks).
- **Guard fallbacks:**
  - In auto and OpenAI-only modes these were `LengthFinishReasonError`s: the supervisor's 600-token cap includes `gpt-6-luna`'s hidden reasoning. On those hops the guard made the same decision the model made everywhere else (Research, then finish). The cap is now 2,000.
  - In NVIDIA-only mode, one Lightning structured-output call was rejected by the API (`400 unknown field guided_json`).
- **Availability failover, observed live:** during the auto run, one Nemotron call hit the 120 s timeout. The circuit breaker opened, and the next 7 Research calls within the 60 s cooldown went to `gpt-6.1-sol`, each traced as `llm_route` with `dimension: availability`.

## 3. Latency

**How it was measured:**
- Client-side wall-clock time from sending the request: first byte, first answer token (TTFT) and the complete answer.
- A server-side breakdown summed from the trace events of each turn:
  - **retrieval:** query embedding + Pinecone + diversification;
  - **LLM calls:** every supervisor, agent and synthesis call;
  - **MCP:** Tavily calls;
  - **server overhead:** everything else on the server;
  - **network + queue:** client time minus server time.
- **Warm and cold are separated.** A request is warm if its first byte arrived within 3 s; all 90 warm-pass requests qualified. Cold starts were measured separately (§3.2).
- **TTFT counts only the first non-whitespace answer text.** Nemotron streams a bare `"\n"` next to its tool call about 2 s before the real answer. Counting it understated TTFT, so the warm pass was re-run on revision `00009` with the corrected definition. The OpenAI-only figures come from the scored run; OpenAI emits no such preamble, so its TTFT was always correct.

### 3.1 Warm (steady-state) latency, seconds, p50 / p90

| Configuration | TTFT | Total | Retrieval | LLM calls | MCP (web) | Server overhead | Network + queue |
|---|---|---|---|---|---|---|---|
| **auto (balanced routing)** | **7.3** / 17.0 | **16.7** / 30.3 | 0.3 / 0.9 | 14.3 / 23.3 | 0.0 / 1.7 | 0.2 / 7.7 | 0.1 / 0.1 |
| OpenAI only | 9.0 / 17.2 | 18.0 / 28.6 | 0.5 / 0.9 | 16.5 / 25.6 | 0.0 / 2.1 | 0.2 / 1.6 | 0.1 / 0.1 |
| NVIDIA NIM only | 6.1 / 20.1 | 15.4 / 46.7 | 0.4 / 0.8 | 13.4 / 31.7 | 0.0 / 0.1 | 0.2 / 2.6 | 0.1 / 0.2 |

**What happens before the first token** (p50, seconds from sending the request):

| Configuration | First byte (trace visible) | Supervisor decided | Agent started | First retrieval done | First answer token |
|---|---|---|---|---|---|
| auto (balanced routing) | 0.1 | 1.7 | 1.7 | 3.6 | 7.3 |
| NVIDIA NIM only | 0.2 | 1.3 | 1.4 | 3.4 | 6.1 |

**Why our warm TTFT (≈ 7 s) is higher than the reference's ~1.5–2 s.** Two LLM calls, one tool call and their round trips all run *before* any answer text exists:
1. The supervisor decides which agent should run.
2. The agent decides what to search for.
3. Retrieval runs.
4. Only then does the agent start writing.

That is the price of a supervisor plus a grounded ReAct agent, as opposed to a single retrieve-then-generate call. The UI isn't idle meanwhile: the trace and routing chips appear within 0.1–1.7 s, and the status line shows each step as it happens.

**Where the time goes:** LLM calls are about 85% of the total. Retrieval is only 0.3–0.9 s, and server overhead is 0.2 s. The p90 overhead outliers are turns where a provider error triggered a failover.

### 3.2 Cold starts

Each probe waited 17–18 minutes with no traffic, so that Cloud Run scaled to zero, then sent one RAG question. Every probe was confirmed by a `Starting new instance` entry in Cloud Run's logs. The startup split comes from the same logs: container start → `Waiting for application startup` → `Application startup complete`.

| Probe (UTC) | Revision / conditions | First byte | TTFT | Total | Container + imports | App startup |
|---|---|---|---|---|---|---|
| 21:08 | rev 00006; the OpenAI credit ran out during this probe (supervisor failover) | 20.6 s | 29.3 s | 39 s | 7.3 s | 12.4 s |
| 21:28 | rev 00006, during the credit outage; total inflated by extra agents (preview bug, since fixed) | 8.3 s | 24.4 s | 83 s | 3.4 s | 4.3 s |
| 21:49 | rev 00006: same as above | 7.8 s | 20.8 s | 224 s | 3.2 s | 4.2 s |
| 23:53 | **rev 00009 (final), clean** | 6.6 s | 12.2 s | 15 s | 2.6 s | 3.5 s |

- **A cold start adds 6.6–20.6 s before the first byte.** That's 2.6–7.3 s to start the container and import Python, then 3.5–12.4 s of app startup: the Firestore client, the graph, and the warm-up of the Pinecone and embeddings clients plus the MCP handshake.
- **The first probe after a long idle was slowest.** The likely cause is the image being pulled onto a new host.
- **TTFT on a cold start:** 12.2 s on the final revision; 20.8–29.3 s for the earlier probes, which also paid for failover during the credit outage. That's at or below the reference deployment's ~20–25 s. **No cold-start request failed.** The warm-up runs before the port opens, so the first request never races startup work.
- `min-instances 1` would remove cold starts at the cost of an always-on instance; this deployment deliberately scales to zero.

## 4. Cost per query

| Configuration | LLM tokens in / out | LLM | Embeddings | Tavily | Cloud Run | Firestore | **Total** |
|---|---|---|---|---|---|---|---|
| auto (balanced routing) | 8,035 / 822 | $0.00336 | $0.0000004 | $0.0011 | $0.00067 | $0.000030 | **$0.0051** |
| OpenAI only | 12,607 / 746 | $0.01863 | $0.0000007 | $0.0040 | $0.00053 | $0.000030 | **$0.0232** |
| NVIDIA NIM only | 17,855 / 1,164 | $0 | $0.0000003 | $0.0021 | $0.00137 | $0.000030 | **$0.0035** |
| auto, v1 baseline (no failover) | 7,320 / 886 | $0.00032 | $0.0000004 | $0.0003 | $0.00080 | $0.000030 | **$0.0014** |

How each component was estimated (prices checked on 2026-10-26/27):
- **LLM:** measured tokens × list price.
  - `gpt-6-luna` $0.10 / $0.50 per 1M input/output tokens; `gpt-6.1-sol` $2 / $10.
  - NVIDIA NIM's developer tier is free for evaluation use.
  - **Auto mode's typical LLM cost is the v1 figure, $0.0003:** the supervisor on `gpt-6-luna`, answers on free Nemotron. The v2 mean is 10× higher only because 7 questions failed over to `gpt-6.1-sol` (§2.2). Failover now goes to `gpt-6-luna`, so an NVIDIA outage adds only about $0.001 per question.
- **Embeddings:** query tokens × $0.02 per 1M (`text-embedding-3-small`). Embedding the whole corpus once cost about $0.03.
- **Tavily:** credits × $0.008, pay-as-you-go: 1 credit per basic search, 1 per extraction. The free plan's 1,000 credits a month cover this project, so the actual charge is $0.
- **Cloud Run:** request duration × (1 vCPU × $0.000024 + 1 GiB × $0.0000025 per second) + $0.40 per million requests, Tier 1 request-based billing. This is an **upper bound**: up to 10 concurrent requests share one instance's CPU. Usage this small also falls inside the monthly free tier (180k vCPU-seconds).
- **Firestore:** measured 31 document writes + about 8 reads per single-agent turn, at $0.09 / $0.03 per 100k. That is inside the free daily quota (20k writes ≈ 645 turns a day).
- **Pinecone:** Starter plan, free.

**Cost of the evaluation itself:**

| Item | Cost |
|---|---|
| Generating the 152 evaluated answers (all runs) | $0.67 |
| RAGAS judging (`gpt-6.1-sol`: about 15k input + 4k output tokens, $0.07, per question) | $10.00 |

The judge cost includes re-scoring after each round of fixes. **Lesson:** with the same token counts, a `gpt-6-luna` judge would have cost about $0.09 per run, roughly 5% of what `gpt-6.1-sol` cost. The provider comparison only needs one consistent judge across configurations, so the cheaper judge would have been the better choice.

## 5. Provider comparison

The same 30 questions, run with each provider doing all the work (`force_provider`) and with the default routing:

| | **OpenAI only** (luna supervisor, sol agents) | **NVIDIA NIM only** (Lightning supervisor, Ultra agents) | **auto** (luna supervisor, Ultra agents) |
|---|---|---|---|
| Faithfulness (common 25) | **0.900** | 0.782 | 0.839 |
| Context recall (common 25) | **0.893** | 0.867 | 0.827 |
| Context precision (common 25) | 0.915 | 0.912 | **0.942** |
| Answer relevancy (common 25) | 0.835 | 0.836 | **0.843** |
| Warm TTFT p50 | 9.0 s | **6.1 s** | 7.3 s |
| Warm total p50 | 18.0 s | **15.4 s** | 16.7 s |
| LLM cost per question | $0.0186 | **$0** | $0.0003* |
| Answers not grounded in the corpus | 1 web-only | 3 web-only, 1 no sources | 1 web-only |
| Supervisor behavior | finished after Research every time | ran extra agents on 13/30 (fixed: §2.2) | finished after Research every time |
| Reliability issues seen | `LengthFinishReasonError` on the supervisor (cap raised); a hard stop when the prepaid credit ran out | one 120 s timeout; empty completions; intermittent `400 unknown field guided_json` | absorbed by failover |

\* Typical (the v1 run). The v2 mean, $0.0034, includes 7 questions that failed over to `gpt-6.1-sol` during an NVIDIA timeout.

**Caveat:** the judge is `gpt-6.1-sol`, the OpenAI-only generator, so its faithfulness lead may be inflated by self-enhancement bias.

**What this says, and how the router uses it:**
- **`gpt-6.1-sol` is the most faithful writer** (+0.06 over auto), but at about 60× auto's LLM cost and with the slowest first token, because it reasons before streaming. It is kept for the **Quality** policy, where a user explicitly trades cost for faithfulness.
- **Nemotron Ultra (free) is a sound default for agent work:** recall within 0.03 of OpenAI's, equal relevancy, the fastest TTFT. Its weaker faithfulness is why the citation rules forbid unsupported facts (§2.1), and why it runs behind failover.
- **The supervisor should not be Lightning.**
  - Lightning is fast and free, but it over-routes and occasionally fails structured output. `gpt-6-luna` routes correctly for about $0.0002 a hop, so the balanced policy puts the supervisor (latency dimension) and the Data agent (capability dimension: precise SQL) on luna.
  - Lightning remains the Economy supervisor, now that its view of results is fixed.
- **The balanced policy captures most of the gap at near-zero cost:**
  - faithfulness 0.839 vs 0.900 for OpenAI-only;
  - the best context precision and relevancy of the three configurations;
  - about $0.0014 per question in total, versus about $0.023 for OpenAI-only.

## Appendix A: per question (auto, balanced routing)

Quality (RAGAS) and cost are from the scored run (revision `00005`). The latency columns are from the warm latency pass (revision `00009`), whose answers are independent of the scored run. Ctx is the number of retrieved chunks. A dash means the answer used no corpus retrieval, so it wasn't scored.

| Question | Agent | Ctx | TTFT | Total | Retrieval | LLM | Cost | Faith. | Relev. | Prec. | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| react | research | 6 | 3.6 | 10.0 | 0.30 | 9.3 | 0.00039 | 0.85 | 0.92 | 0.65 | 1.00 |
| reflexion | research | 9 | 11.2 | 18.1 | 0.00 | 14.0 | 0.00028 | 0.80 | 0.87 | 0.99 | 0.67 |
| rewoo_vs_llmcompiler | research | 5 | 7.5 | 14.9 | 0.54 | 13.9 | 0.00030 | 0.35 | 0.89 | 1.00 | 0.00 |
| self_consistency | research | 5 | 13.0 | 25.5 | 0.21 | 17.4 | 0.00026 | 0.67 | 0.85 | 1.00 | 1.00 |
| tot | research | 5 | 3.3 | 16.3 | 0.48 | 14.7 | 0.00027 | 1.00 | 0.75 | 1.00 | 1.00 |
| toolformer | research | 5 | 8.0 | 14.0 | 0.85 | 5.5 | 0.00048 | 1.00 | 0.79 | 1.00 | 0.50 |
| selfrag_vs_crag | research | 10 | 21.8 | 30.3 | 0.24 | 15.7 | 0.00033 | 0.72 | 0.91 | 0.88 | 1.00 |
| adaptive_rag | research | 7 | 18.1 | 46.2 | 0.55 | 37.0 | 0.00034 | 0.83 | 0.79 | 0.92 | 1.00 |
| raptor | research | 7 | 16.9 | 28.7 | 0.85 | 18.8 | 0.00013 | 0.76 | 0.86 | 0.82 | 1.00 |
| graphrag | research | 8 | 6.6 | 12.0 | 0.94 | 10.8 | 0.00028 | 0.90 | 0.88 | 0.75 | 1.00 |
| lost_middle | research | 5 | 2.9 | 8.3 | 0.24 | 7.8 | 0.00028 | 0.92 | 0.78 | 0.83 | 1.00 |
| dpr | research | 10 | 6.2 | 41.0 | 0.30 | 40.5 | 0.00044 | 0.85 | 0.91 | 0.77 | 0.33 |
| hyde | research | 5 | 11.7 | 25.7 | 0.00 | 22.7 | 0.00029 | 1.00 | 0.89 | 1.00 | 1.00 |
| frugalgpt | research | 5 | 4.0 | 9.9 | 0.51 | 9.0 | 0.00042 | 0.82 | 0.72 | 1.00 | 1.00 |
| routellm | research | 5 | 17.9 | 30.2 | 1.29 | 28.6 | 0.00013 | 0.71 | 0.77 | 1.00 | 0.67 |
| mixtral | research | 0 | 12.3 | 15.8 | 0.24 | 8.1 | 0.00030 | — | — | — | — |
| metagpt | research | 5 | 5.7 | 14.0 | 0.23 | 12.9 | 0.00025 | 0.81 | 0.85 | 1.00 | 1.00 |
| debate | research | 7 | 7.2 | 17.6 | 0.50 | 14.8 | 0.00026 | 0.82 | 0.89 | 1.00 | 1.00 |
| more_agents | research | 5 | 7.7 | 13.6 | 0.45 | 11.2 | 0.00040 | 0.65 | 0.72 | 1.00 | 1.00 |
| moa | research | 5 | 4.5 | 18.7 | 0.23 | 18.1 | 0.00044 | 0.94 | 0.80 | 1.00 | 1.00 |
| magentic_one | research | 5 | 14.3 | 22.3 | 0.29 | 21.8 | 0.00899 | 1.00 | 0.93 | 1.00 | 0.67 |
| mast | research | 7 | 7.9 | 18.7 | 0.50 | 17.9 | 0.00012 | 0.95 | 0.81 | 0.95 | 0.33 |
| llm_judge | research | 5 | 6.9 | 17.1 | 0.62 | 16.2 | 0.00013 | 0.59 | 0.85 | 0.64 | 0.50 |
| ragas | research | 5 | 5.4 | 9.5 | 0.32 | 9.0 | 0.00029 | 1.00 | 0.82 | 1.00 | 1.00 |
| lg_checkpointer | research | 13 | 4.1 | 11.1 | 0.24 | 10.5 | 0.02181 | 0.87 | 0.82 | 1.00 | 1.00 |
| lg_command | research | 7 | 7.3 | 14.0 | 0.52 | 13.2 | 0.01397 | 0.96 | 0.88 | 1.00 | 1.00 |
| mcp_transports | research | 5 | 4.2 | 14.5 | 0.22 | 14.0 | 0.02547 | 0.65 | 0.81 | 1.00 | 1.00 |
| mcp_server_features | research | 5 | 3.7 | 10.0 | 0.20 | 9.5 | 0.00737 | 1.00 | 0.84 | 1.00 | 1.00 |
| a2a_agent_card | research | 5 | 9.1 | 19.3 | 0.00 | 16.9 | 0.00709 | 1.00 | 0.87 | 1.00 | 1.00 |
| a2a_vs_mcp | research | 5 | 5.9 | 20.2 | 0.28 | 19.7 | 0.00930 | 1.00 | 0.87 | 1.00 | 1.00 |

## Appendix B: files and reproduction

| Path | Contents |
|---|---|
| [`src/eval/questions.json`](../src/eval/questions.json) | The 30 questions, their expected documents and the verified reference answers |
| `src/eval/runs/{auto,openai,nvidia,auto_v1}.jsonl` | Every answer, retrieved contexts, trace-derived timings and costs (in the submission zip) |
| `src/eval/runs/{auto,nvidia}_lat.jsonl` | The warm latency pass |
| [`src/eval/results/`](../src/eval/results/) | RAGAS scores per question (`ragas_*.json`), cold-start probes, the generated `report.md` |

Commands to reproduce are in the README ("Reproduce the evaluation"). The tables above come from:

```
python -m src.eval.report --runs auto openai nvidia auto_v1 --latency-runs auto_lat openai nvidia_lat --firestore-writes 31 --firestore-reads 8
```
