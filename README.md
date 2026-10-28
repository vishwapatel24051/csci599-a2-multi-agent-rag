# Multi-Agent RAG System (CSCI 599, Assignment 2)

A LangGraph **supervisor** orchestrates four specialized agents:
- **Research:** corpus RAG plus live web search.
- **Code:** grounded code generation.
- **Data:** SQL over corpus metadata.
- **Summary:** long-document summarization.

The supervisor answers questions about LLM agents, multi-agent systems, RAG, and the LangGraph, MCP and A2A protocols.

Generation is routed across **two providers, NVIDIA NIM and OpenAI**, by a hybrid cost, capability and availability policy. Retrieval runs over a **Pinecone** index of ~58 arXiv papers plus the LangGraph, MCP and A2A docs. Live knowledge comes from **Tavily Remote MCP**. The UI is a vanilla HTML/JS page fed by **FastAPI + SSE**, deployed on **Google Cloud Run**.

**Live:** https://multi-agent-rag-157472143408.us-east4.run.app (public; scales to zero, so the first request after idle includes a cold start)

- **Architecture:** [docs/architecture.md](docs/architecture.md). Diagram, state schema, agents and tools, routing table, RAG pipeline, durable state.
- **Evaluation:** [docs/evaluation.md](docs/evaluation.md). RAGAS, latency (warm and cold), cost per query, provider comparison.
- **How it works, for learning:** [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md).
- **Build log:** [docs/BUILD_NOTES.md](docs/BUILD_NOTES.md). Dated notes of what broke and how it was fixed.

---

## Provider requirements and API keys

| Service | Used for | Plan / cost | Key looks like |
|---|---|---|---|
| OpenAI | `gpt-6-luna`: supervisor routing, SQL, short-document summaries, failover. `gpt-6.1-sol`: Quality policy and RAGAS judge only. Embeddings. | Pay-as-you-go, prepaid credits (≈ $15 including RAGAS evaluation; the app alone costs cents) | `sk-proj-…` |
| NVIDIA NIM | Research, code, synthesis and long-document summaries | Free developer tier (~40 requests/min, evaluation use) | `nvapi-…` |
| Pinecone | Vector store | Starter plan, free (2 GB, AWS us-east-1 only) | `pcsk_…` |
| Tavily | Live web search via Remote MCP | Free Researcher plan (1,000 credits/month) | `tvly-…` |
| Google Cloud | Cloud Run, Firestore, Secret Manager, Cloud Build | USC education credits | (gcloud login) |

### 1. OpenAI
1. Sign in at <https://platform.openai.com>.
2. Add credit, because API calls fail with `insufficient_quota` until the account has a balance:
   - Go to **Settings → Billing**, add a payment method and buy prepaid credits. $5 covers the app; budget about $15 to also run the RAGAS evaluation, whose judge is the main cost. Credits can take ~15 minutes to become usable.
   - Optional: set a monthly budget under **Settings → Limits**.
3. Go to **API keys** (<https://platform.openai.com/api-keys>) and click **Create new secret key**.
   - Owner: *You*. Project: *Default project*, or create one named `csci599-a2`. Permissions: *All*.
4. Copy the key right away; it is shown only once.

### 2. NVIDIA NIM
1. Go to <https://build.nvidia.com> and **Log in**. Joining the NVIDIA Developer Program is free.
2. Open any model page, e.g. <https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b>, and click **Get API Key**. (You can also go to your profile menu → **API Keys** → **Generate API Key**.)
3. Copy the key.
4. Free-tier limit: about 40 requests per minute per account. The app throttles itself and falls back to OpenAI when NVIDIA rate-limits.

### 3. Pinecone
1. Sign up at <https://app.pinecone.io>. The Starter plan is free and needs no card.
2. In the left sidebar, open **API keys**, click **Create API key**, name it `csci599-a2`, and copy it.
3. Do **not** create an index by hand. `python -m src.rag.ingest` creates `agent-engineering-corpus` (1536 dimensions, cosine, serverless, aws/us-east-1).

### 4. Tavily
1. Sign up at <https://app.tavily.com>. The free Researcher plan gives 1,000 credits a month with no card. Claim the CSCI 599 student plan if it's offered, or reuse your Assignment 1 key.
2. Copy the API key from the dashboard.
3. The app passes the key as a URL query parameter to `https://mcp.tavily.com/mcp/`, as the assignment specifies.

### 5. Put the keys in `.env`
```bash
cp .env.example .env     # .env is git-ignored; never commit it
# edit .env and fill in OPENAI_API_KEY, NVIDIA_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY
```

---

## Local development

Requires Python 3.13.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # runtime deps + ingestion/eval/test tools

.venv/bin/python -m src.verify_setup            # checks every key and model ID
```

### Build the corpus
```bash
.venv/bin/python -m src.rag.fetch_corpus    # papers from arXiv + docs pages from GitHub (≈ 6 min, polite rate limits)
.venv/bin/python -m src.rag.build_chunks    # PDF/MDX → structure-aware chunks (no keys needed)
.venv/bin/python -m src.data.build_db       # SQLite DB for the Data agent
.venv/bin/python -m src.rag.ingest          # embed + upsert to Pinecone (incremental; --force to redo)
```

### Run the app
```bash
.venv/bin/uvicorn main:app --port 8080 --reload     # then open http://localhost:8080
.venv/bin/python -m src.run_turn "How does ReAct differ from chain-of-thought?"   # same graph, trace in the terminal
.venv/bin/python -m pytest -q                         # 60 offline tests (no API calls)
```
Locally, conversation state goes to `data/checkpoints.sqlite` (`STATE_BACKEND=sqlite`). On Cloud Run it goes to Firestore.

The UI is plain HTML/CSS/JS served by FastAPI from `static/`. There's no framework and no build step. Two small libraries are vendored in `static/vendor/`: `marked` 12.0.2 for Markdown and DOMPurify 3.1.6 to sanitize model output.

### Reproduce the evaluation
```bash
.venv/bin/python -m src.eval.collect --mode auto     # also: --mode openai, --mode nvidia --concurrency 2
python3.13 -m venv .venv-eval && .venv-eval/bin/pip install -r requirements-eval.txt   # RAGAS needs its own venv
.venv-eval/bin/python src/eval/ragas_eval.py src/eval/runs/auto.jsonl
.venv/bin/python -m src.eval.cold_start --probes 3 --idle-min 18
.venv/bin/python -m src.eval.report --runs auto openai nvidia --firestore-writes 31 --firestore-reads 8
```
`collect` and `cold_start` call the deployed URL by default (`--url` to change it); `collect` needs `data/processed/chunks.jsonl` to attach retrieved chunk texts. Keep the machine awake (`caffeinate` on macOS): a sleeping laptop breaks the streams.

## Deploy to Cloud Run

**Deploying to a fresh GCP project, step by step:**
1. **Keys:** `cp .env.example .env` and fill in the four API keys (see [Provider requirements](#provider-requirements-and-api-keys)).
2. **Vector index, once per Pinecone account:** the container only *connects* to Pinecone, so the index must be populated first. Run the four commands in [Build the corpus](#build-the-corpus): about 10 minutes and $0.03 of embeddings. The Data agent's SQLite file (`data/agentlab.db`) ships with the repo.
3. **gcloud:** install the CLI, run `gcloud auth login`, and find your billing account ID with `gcloud billing accounts list`.
4. **Deploy.** Pick a new, globally unique project ID; this is the one command that does everything, and it's safe to re-run:

   ```bash
   PROJECT_ID=<your-new-project-id> BILLING_ACCOUNT=<XXXXXX-XXXXXX-XXXXXX> ./scripts/deploy.sh
   ```

   For an existing project that already has billing, omit `BILLING_ACCOUNT`.
5. **Verify:** the script ends by printing the service URL and calling `/health`. Open the URL and ask a question.

(This deployment: `PROJECT_ID=csci599-a2-617204`, region `us-east4`.)

[`scripts/deploy.sh`](scripts/deploy.sh):
1. Creates the project and links billing.
2. Enables Cloud Run, Cloud Build, Artifact Registry, Firestore, Secret Manager and IAM.
3. Creates a Firestore (Native) database in `us-east4` for conversation checkpoints.
4. Copies the four API keys from `.env` into **Secret Manager**. They are never baked into the image or printed. Set `UPDATE_SECRETS=1` to push rotated keys.
5. Creates a dedicated runtime service account with only `roles/datastore.user` plus `secretAccessor` on those four secrets.
6. Builds the image with **Cloud Build** (no local Docker needed) and deploys it, with these settings:
   - `--min-instances 0 --max-instances 3`: scales to zero and caps cost.
   - `--cpu-boost`: faster cold starts.
   - `--timeout 600`: long multi-agent turns over SSE.
   - `--concurrency 10`.
   - Env: `STATE_BACKEND=firestore`.

The container is a multi-stage `python:3.13-slim` image that runs as a non-root user (uid 10001). `.gcloudignore` limits the upload to code, UI and the 228 KB SQLite file, so `.env` and the raw corpus never leave your machine. The tokenizer vocabulary is baked into the image, so cold starts don't download it.

Re-deploying after a code change uses the same command. It skips the setup steps that are already done.

**Troubleshooting a first deploy to a brand-new project:**
- **`PERMISSION_DENIED` on `artifacts repositories create` right after the project is created:** the owner role takes a minute or two to propagate. Wait and re-run the same command.
- **Don't use a health-check path ending in `z`:** Cloud Run's front end reserves paths like `/healthz` and returns its own 404. This app uses `/health`.
- **Laptop sleep:** `gcloud builds submit` streams the build log. If your machine sleeps, the build still finishes server-side; re-run the script to complete the deploy step.

## Cost and access notes

**Access.**
- The deployed app is public: no login, `--allow-unauthenticated`.
- `--max-instances 3` caps runaway cost. Each instance handles up to 10 concurrent chats.
- Conversations are keyed by a random session ID kept in the browser; anyone with that ID could read the conversation, so don't paste anything sensitive.
- API keys live only in Secret Manager (deployed) and the git-ignored `.env` (local). **Rotate all four after grading.**

**Cost per question** (measured 2026-10-27 over the 30-question evaluation set; details in [docs/evaluation.md](docs/evaluation.md#4-cost-per-query)):

| Configuration | Typical cost per question | What drives it |
|---|---|---|
| Balanced (default) | ≈ $0.0014 | Supervisor on `gpt-6-luna`; answers on free Nemotron; Cloud Run time. If NVIDIA fails, calls fall back to `gpt-6-luna`, adding about $0.001. |
| Economy (all NVIDIA) | ≈ $0.0035 | LLMs free; longer turns cost more Cloud Run time |
| Quality / all OpenAI | ≈ $0.023 | `gpt-6.1-sol` for all agent work |

| Service | Price used | Plan | This project |
|---|---|---|---|
| OpenAI | `gpt-6-luna` $0.10/$0.50 and `gpt-6.1-sol` $2/$10 per 1M input/output tokens; embeddings $0.02 per 1M | Prepaid credits | ≈ $0.03 to embed the corpus, under $1 of answers, ≈ $10 of RAGAS judging |
| NVIDIA NIM | $0 | Free developer tier (~40 requests/min, evaluation use) | — |
| Tavily | 1 credit per search or extraction, $0.008/credit pay-as-you-go | Free plan, 1,000 credits/month | Well within the free plan |
| Pinecone | — | Starter, free | — |
| Cloud Run | $0.000024 per vCPU-second, $0.0000025 per GiB-second, $0.40 per 1M requests (request-based billing) | Monthly free tier | ≤ $0.0014 per question (upper bound) |
| Firestore | $0.09 per 100k writes, $0.03 per 100k reads | Free daily quota: 20k writes, 50k reads | 31 writes + ~8 reads per turn |

Scaling to zero means an idle deployment costs only Artifact Registry storage (a few cents a month). The trade-off is a cold start on the first request after about 15 idle minutes; see the evaluation.
