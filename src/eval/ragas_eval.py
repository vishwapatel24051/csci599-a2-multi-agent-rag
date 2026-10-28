"""Score collected runs with RAGAS: faithfulness, answer relevancy, context precision, context recall.

Runs in its own virtualenv because ragas 0.4.3 imports langchain-community modules that LangChain 1.x
removed (see requirements-eval.txt):

    python3.13 -m venv .venv-eval && .venv-eval/bin/pip install -r requirements-eval.txt
    .venv-eval/bin/python src/eval/ragas_eval.py src/eval/runs/auto_2026-10-27.jsonl [more runs ...]

Writes src/eval/results/ragas_<run name>.json: per-question scores, means, and the judge's token cost.
The judge is OpenAI gpt-6.1-sol (the reference configuration's judge family); embeddings for answer
relevancy are text-embedding-3-small, the same model the index uses.
"""

import argparse
import json
import math
import os
import statistics
import time
import warnings
from pathlib import Path

# ragas sends usage telemetry by default; its exit-time flush hung the process for ~15 min here. Opt out
# before ragas is imported.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from dotenv import load_dotenv  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "src/eval/questions.json"
RESULTS = ROOT / "src/eval/results"
JUDGE_MODEL = "gpt-6.1-sol"
JUDGE_PRICE = (2.00, 10.00)  # USD per 1M input / output tokens (src/llm/pricing.py, checked 2026-10-26)
EMBEDDING_MODEL = "text-embedding-3-small"
METRICS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]


def score_run(path: Path, llm, embeddings, references: dict[str, str]) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    scored = [r for r in records if r["answer"] and not r["error"] and r["contexts"]]
    skipped = [{"id": r["id"], "why": r["error"] or ("no contexts" if r["answer"] else "no answer")}
               for r in records if r not in scored]
    dataset = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"],
         "reference": references[r["id"]]} for r in scored])
    t0 = time.perf_counter()
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm, embeddings=embeddings,
        run_config=RunConfig(max_workers=8, timeout=240, max_retries=6),
        token_usage_parser=get_token_usage_for_openai,
        show_progress=False,
    )
    df = result.to_pandas()
    rows = []
    for r, (_, s) in zip(scored, df.iterrows()):
        rows.append({"id": r["id"], "topic": r["topic"], "agents": r["agents"], "n_contexts": len(r["contexts"]),
                     **{m: (None if s[m] is None or math.isnan(s[m]) else round(float(s[m]), 4)) for m in METRICS}})
    try:
        usage = result.total_tokens()
        tokens_in, tokens_out = usage.input_tokens, usage.output_tokens
    except IndexError:  # no judge call reported usage (all failed)
        tokens_in = tokens_out = 0
    means = {}
    for m in METRICS:
        vals = [row[m] for row in rows if row[m] is not None]
        means[m] = {"mean": round(statistics.fmean(vals), 4) if vals else None, "n": len(vals),
                    "nan": len(rows) - len(vals)}
    return {
        "run": path.stem, "mode": records[0]["mode"], "n_questions": len(records), "n_scored": len(rows),
        "skipped": skipped, "means": means, "rows": rows,
        "judge": {"model": JUDGE_MODEL, "embeddings": EMBEDDING_MODEL, "input_tokens": tokens_in,
                  "output_tokens": tokens_out,
                  "cost_usd": round((tokens_in * JUDGE_PRICE[0] + tokens_out * JUDGE_PRICE[1]) / 1e6, 4),
                  "wall_s": round(time.perf_counter() - t0, 1)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    # gpt-6.1-sol is a reasoning model that only accepts its default temperature (1); ragas would send 0.01.
    # bypass_n: request answer relevancy's 3 generated questions one by one (with n=3 only 1 came back).
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, api_key=os.environ["OPENAI_API_KEY"], timeout=180),
                              bypass_temperature=True, bypass_n=True)
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL,
                                                             api_key=os.environ["OPENAI_API_KEY"]))
    references = {q["id"]: q["reference"] for q in json.loads(QUESTIONS.read_text())["questions"]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for path in args.runs:
        out = score_run(path, llm, embeddings, references)
        dest = RESULTS / f"ragas_{path.stem}.json"
        dest.write_text(json.dumps(out, indent=2))
        means = "  ".join(f"{m.split('_')[0] if m != 'answer_relevancy' else 'relevancy'}={v['mean']}"
                          for m, v in out["means"].items())
        print(f"{path.stem}: scored {out['n_scored']}/{out['n_questions']}  {means}  "
              f"judge ${out['judge']['cost_usd']} in {out['judge']['wall_s']}s -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
