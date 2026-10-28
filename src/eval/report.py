"""Turn collected runs + RAGAS scores + cold-start probes into the tables in docs/evaluation.md.

    python -m src.eval.report --runs auto_2026-10-27 openai_2026-10-27 nvidia_2026-10-27

Writes src/eval/results/report.md (and prints it). Latency is reported for warm requests only; a request
counts as cold when its first byte took > 3 s (the server emits its first trace event within ~0.3 s, so a
slower first byte means the request waited for an instance to start). Cold starts are reported separately
from the cold-start probes.
"""

import argparse
import json
import statistics

from src.config import ROOT
from src.rag.chunking import n_tokens

RUNS = ROOT / "src/eval/runs"
RESULTS = ROOT / "src/eval/results"
COLD_FIRST_BYTE_S = 3.0

# Unit prices used for the per-query cost estimate (USD), checked 2026-10-27.
EMBED_PER_TOKEN = 0.02 / 1e6           # OpenAI text-embedding-3-small
TAVILY_PER_CREDIT = 0.008              # pay-as-you-go; basic search = 1 credit, extract = 1 credit per 5 URLs
RUN_VCPU_S = 0.000024                  # Cloud Run request-based billing, Tier 1 (us-east4), 1 vCPU
RUN_GIB_S = 0.0000025                  # ... 1 GiB memory
RUN_PER_REQUEST = 0.40 / 1e6
FIRESTORE_PER_WRITE = 0.09 / 1e5       # Standard edition; below the free daily quota at this scale
FIRESTORE_PER_READ = 0.03 / 1e5

METRICS = [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer relevancy"),
           ("llm_context_precision_with_reference", "Context precision"), ("context_recall", "Context recall")]
MODE_LABEL = {"auto": "auto (balanced routing)", "openai": "OpenAI only", "nvidia": "NVIDIA NIM only"}


def label(name: str, recs: list[dict]) -> str:
    base = MODE_LABEL[recs[0]["mode"]]
    return f"{base}, v1 baseline" if name.endswith("_v1") else base


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def fmt(x, digits=1) -> str:
    return "—" if x is None or x != x else f"{x:.{digits}f}"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (RUNS / f"{name}.jsonl").read_text().splitlines()]


def infra_cost(r: dict, firestore_writes: int, firestore_reads: int) -> dict:
    embed_tokens = sum(n_tokens(q) for q in r["retrieval_queries"])
    credits = r["mcp_tools"].get("tavily_search", 0) + r["mcp_tools"].get("tavily_extract", 0)
    seconds = r["timing"]["server_total_s"] or r["timing"]["total_s"]
    return {
        "embedding": embed_tokens * EMBED_PER_TOKEN,
        "tavily": credits * TAVILY_PER_CREDIT,
        "cloud_run": seconds * (RUN_VCPU_S + RUN_GIB_S) + RUN_PER_REQUEST,
        "firestore": firestore_writes * FIRESTORE_PER_WRITE + firestore_reads * FIRESTORE_PER_READ,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="scored runs: quality, cost, grounding")
    ap.add_argument("--latency-runs", nargs="*", help="dedicated latency runs (default: --runs)")
    ap.add_argument("--firestore-writes", type=int, default=0, help="measured Firestore writes per turn")
    ap.add_argument("--firestore-reads", type=int, default=0, help="measured Firestore reads per turn")
    args = ap.parse_args()

    runs = {name: load(name) for name in args.runs}
    lat_runs = {name: load(name) for name in args.latency_runs} if args.latency_runs else runs
    ragas = {name: json.loads((RESULTS / f"ragas_{name}.json").read_text())
             for name in args.runs if (RESULTS / f"ragas_{name}.json").exists()}
    out: list[str] = []
    w = out.append

    # ---------------------------------------------------------------- RAGAS
    w("### RAGAS scores (mean over the 30-question set)\n")
    w("| Configuration | " + " | ".join(label for _, label in METRICS) + " | Scored | Judge cost |")
    w("|---|" + "---|" * (len(METRICS) + 2))
    for name, recs in runs.items():
        rg = ragas.get(name)
        if not rg:
            continue
        cells = [fmt(rg["means"][m]["mean"], 3) for m, _ in METRICS]
        w(f"| {label(name, recs)} | " + " | ".join(cells)
          + f" | {rg['n_scored']}/{rg['n_questions']} | ${rg['judge']['cost_usd']:.2f} |")
    w("")

    # Same questions for every configuration: a question is skipped when its answer used no corpus retrieval,
    # and which questions those are differs by configuration.
    scored = {name: {row["id"]: row for row in rg["rows"]} for name, rg in ragas.items()}
    common = set.intersection(*(set(rows) for rows in scored.values())) if scored else set()
    w(f"**Common subset** (the {len(common)} questions scored in every configuration above):\n")
    w("| Configuration | " + " | ".join(lbl for _, lbl in METRICS) + " |")
    w("|---|" + "---|" * len(METRICS))
    for name, rows in scored.items():
        cells = [fmt(statistics.fmean(rows[i][m] for i in common if rows[i][m] is not None), 3) for m, _ in METRICS]
        w(f"| {label(name, runs[name])} | " + " | ".join(cells) + " |")
    w("")

    # ---------------------------------------------------------------- grounding behaviour
    w("### How answers were grounded\n")
    w("| Configuration | Corpus retrieval | Web only | No sources | Extra agents after research | Supervisor guard fallbacks |")
    w("|---|---|---|---|---|---|")
    for name, recs in runs.items():
        corpus = sum(1 for r in recs if r["n_retrieved"] or any(c.startswith("S") for c in r["cited"]))
        web_only = sum(1 for r in recs if not r["contexts"] and r["web_used"])
        none = sum(1 for r in recs if not r["contexts"] and not r["web_used"])
        multi = sum(1 for r in recs if len(r["agents"]) > 1)
        guards = sum(1 for r in recs for x in r["routes"] if "supervisor LLM failed" in x["reason"])
        w(f"| {label(name, recs)} | {corpus} | {web_only} | {none} | {multi} | {guards} |")
    w("")

    # ---------------------------------------------------------------- latency (warm)
    def warm_of(recs):
        return [r for r in recs if not r["error"] and r["timing"]["first_byte_s"] <= COLD_FIRST_BYTE_S]

    def col(recs, get, both=True):
        vals = [v for v in map(get, recs) if v is not None]
        return f"{fmt(pct(vals, .5))} / {fmt(pct(vals, .9))}" if both else fmt(pct(vals, .5))

    w("### Warm latency per query (seconds; p50 / p90)\n")
    w("| Configuration | n warm | TTFT (client) | Total (client) | Retrieval | LLM calls | MCP (web) | "
      "Server overhead | Network + queue |")
    w("|---|---|---|---|---|---|---|---|---|")
    for name, recs in lat_runs.items():
        warm = warm_of(recs)
        w(f"| {label(name, recs)} | {len(warm)}/{len(recs)} "
          f"| {col(warm, lambda r: r['timing']['ttft_s'])} | {col(warm, lambda r: r['timing']['total_s'])} "
          f"| {col(warm, lambda r: r['latency']['retrieval_s'])} | {col(warm, lambda r: r['latency']['llm_s'])} "
          f"| {col(warm, lambda r: r['latency']['mcp_s'])} "
          f"| {col(warm, lambda r: (r['timing']['server_total_s'] or 0) - r['latency']['retrieval_s'] - r['latency']['llm_s'] - r['latency']['mcp_s'])} "
          f"| {col(warm, lambda r: r['timing']['total_s'] - (r['timing']['server_total_s'] or r['timing']['total_s']))} |")
    w("")
    if any("routed_s" in r["timing"] for recs in lat_runs.values() for r in recs):
        w("**What happens before the first token** (client clock, p50 seconds from sending the request):\n")
        w("| Configuration | First byte | Supervisor decided | Agent started | First retrieval done | First answer token |")
        w("|---|---|---|---|---|---|")
        for name, recs in lat_runs.items():
            warm = warm_of(recs)
            w(f"| {label(name, recs)} | " + " | ".join(
                col(warm, lambda r, k=k: r["timing"].get(k), both=False)
                for k in ("first_byte_s", "routed_s", "agent_start_s", "first_retrieval_s", "ttft_s")) + " |")
        w("")

    # ---------------------------------------------------------------- cost
    w("### Cost per query (USD, mean over the set)\n")
    w("| Configuration | LLM tokens in / out | LLM | Embeddings | Tavily | Cloud Run | Firestore | **Total** |")
    w("|---|---|---|---|---|---|---|---|")
    for name, recs in runs.items():
        ok = [r for r in recs if not r["error"]]
        infra = [infra_cost(r, args.firestore_writes, args.firestore_reads) for r in ok]
        llm = statistics.fmean(r["llm_cost_usd"] for r in ok)
        parts = {k: statistics.fmean(i[k] for i in infra) for k in infra[0]}
        total = llm + sum(parts.values())
        w(f"| {label(name, recs)} | {statistics.fmean(r['tokens']['in'] for r in ok):,.0f} / "
          f"{statistics.fmean(r['tokens']['out'] for r in ok):,.0f} | {llm:.5f} | {parts['embedding']:.7f} "
          f"| {parts['tavily']:.4f} | {parts['cloud_run']:.5f} | {parts['firestore']:.6f} | **{total:.4f}** |")
    w("")

    # ---------------------------------------------------------------- routing actually used
    w("### Agents and models used\n")
    w("| Configuration | Agents (questions) | LLM calls by node:model | Errors |")
    w("|---|---|---|---|")
    for name, recs in runs.items():
        agents, models = {}, {}
        for r in recs:
            key = "+".join(r["agents"]) or "(direct)"
            agents[key] = agents.get(key, 0) + 1
            for k, v in r["models"].items():
                models[k] = models.get(k, 0) + v
        errs = sum(1 for r in recs if r["error"] or r["llm_errors"])
        w(f"| {label(name, recs)} | " + ", ".join(f"{k} ({v})" for k, v in sorted(agents.items()))
          + " | " + ", ".join(f"{k.replace('nvidia/', '')} ×{v}" for k, v in sorted(models.items())) + f" | {errs} |")
    w("")

    # ---------------------------------------------------------------- per question (first run)
    # Quality and cost from the scored run; latency from the latency run of the same mode when there is one.
    first = args.runs[0]
    scores = {row["id"]: row for row in ragas.get(first, {}).get("rows", [])}
    mode = runs[first][0]["mode"]
    lat = next(({r["id"]: r for r in recs} for recs in lat_runs.values() if recs[0]["mode"] == mode),
               {r["id"]: r for r in runs[first]})
    w(f"### Per question: {label(first, runs[first])}\n")
    w("| Question | Agent | Ctx | TTFT | Total | Retrieval | LLM | Cost | Faith. | Relev. | Prec. | Recall |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs[first]:
        s, t = scores.get(r["id"], {}), lat.get(r["id"], r)
        cold = " (cold)" if t["timing"]["first_byte_s"] > COLD_FIRST_BYTE_S else ""
        w(f"| {r['id']} | {'+'.join(r['agents'])} | {len(r['contexts'])} | {fmt(t['timing']['ttft_s'])}{cold} "
          f"| {fmt(t['timing']['total_s'])} | {fmt(t['latency']['retrieval_s'], 2)} | {fmt(t['latency']['llm_s'])} "
          f"| {r['llm_cost_usd']:.5f} | " + " | ".join(fmt(s.get(m), 2) for m, _ in METRICS) + " |")
    w("")

    # ---------------------------------------------------------------- cold starts
    cold_path = RESULTS / "cold_starts.jsonl"
    if cold_path.exists():
        probes = [json.loads(line) for line in cold_path.read_text().splitlines()]
        w("### Cold starts (min-instances = 0)\n")
        w("| Probe (UTC) | Confirmed new instance | First byte | TTFT | Total | Container + imports | App startup |")
        w("|---|---|---|---|---|---|---|")
        for p in probes:
            t, s = p["timing"], p["startup"]
            w(f"| {p['at']} | {'yes' if p['cold_confirmed'] else 'no'} | {fmt(t['first_byte_s'])} | {fmt(t['ttft_s'])} "
              f"| {fmt(t['total_s'])} | {fmt(s.get('container_and_imports_s'))} | {fmt(s.get('app_startup_s'))} |")
        w("")

    text = "\n".join(out)
    (RESULTS / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
