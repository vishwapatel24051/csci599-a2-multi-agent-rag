# Process Log

## AI tools used
I used **Claude Code** (Claude Opus 5.5) as my coding assistant throughout. The system itself uses:
- **OpenAI:** `gpt-6-luna` (supervisor, SQL agent); `gpt-6.1-sol` (Quality policy, RAGAS judge); `text-embedding-3-small` (embeddings).
- **NVIDIA NIM:** `nemotron-3-ultra-550b-a55b` and `nemotron-3.5-lightning-30b-a3b`.

## Development narrative
I picked NVIDIA NIM + OpenAI and an AI-engineering corpus, and asked for a roadmap before any code.
- **What worked:** benchmarking models before routing to them. Nemotron's default "thinking" mode was 5–10× slower, so it now runs only for hard tasks.
- **Where I got stuck:**
  - **Deploy-day bugs:** an import deadlock at startup, and Cloud Run reserving `/healthz`.
  - **A supervisor that kept re-running agents** because it only saw 800-character previews.
  - **My OpenAI credit ran out mid-evaluation,** mostly on the `gpt-6.1-sol` RAGAS judge, so I had the routing switched to `gpt-6-luna`.

## Verification narrative
Beyond running the code:
- 60 offline tests of the failure paths;
- a 30-question RAGAS set with corpus-verified reference answers;
- latency and cold-start probes, confirmed against Cloud Run's logs;
- a test that Firestore memory survives instance replacement.

Testing the live app myself, I found unclickable citations and a wrong "latest version" answer. That led to an automated, rubric-based self-test (~100 checks), which also caught follow-ups sometimes returning an empty reply.

## What I learned
The AI tools could build, deploy and measure, but didn't watch my budget or see the app as a user does. I had to:
- question the expensive judge model and notice when my credit ran out;
- test the UI by hand to find bugs that the aggregate metrics hid.

I also learned that what an orchestrator can see decides what it does, and that a trace is an evaluation tool, not just a log.
