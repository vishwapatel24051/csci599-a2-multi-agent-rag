#!/usr/bin/env bash
# Build the submission zip: source, docs, evaluation results, small data files. No secrets, no virtualenvs,
# no downloaded corpus (rebuild with `python -m src.rag.fetch_corpus`), no generated chunk files.
#
#   ./scripts/package.sh            → dist/csci599-a2-<date>.zip
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="dist/csci599-a2-$(date +%Y-%m-%d).zip"
mkdir -p dist
rm -f "$OUT"

zip -qr "$OUT" . \
  -x '.env' '.env.*' \
  -x '.git/*' '.venv/*' '.venv-eval/*' 'dist/*' 'submission/*' 'submission.zip' \
  -x '*__pycache__/*' '*.pyc' '.pytest_cache/*' '.DS_Store' '*/.DS_Store' \
  -x 'data/raw/*' 'data/processed/*' 'data/checkpoints.sqlite*' \
  -x 'src/eval/runs/smoke*' 'src/eval/runs/local*' 'src/eval/runs/*INVALID*' 'src/eval/runs/nvidia_bad_supervisor*' \
  -x 'src/eval/results/ragas_smoke.json' 'src/eval/results/ragas_nvidia_bad_supervisor.json' \
  -x 'Assignment_2_Description*.pdf' 'docs.zip' 'docs 2/*' '.vscode/*' '.idea/*'
# .env.example is wanted; the '.env.*' pattern above excludes it, so add it back.
zip -q "$OUT" .env.example

# Guard: refuse to ship anything that looks like a key.
if unzip -Z1 "$OUT" | grep -qE '(^|/)\.env$'; then echo "ERROR: .env in zip" >&2; rm -f "$OUT"; exit 1; fi
if unzip -p "$OUT" 2>/dev/null | grep -aqE 'sk-proj-[A-Za-z0-9_-]{20}|nvapi-[A-Za-z0-9_-]{20}|pcsk_[A-Za-z0-9_]{20}|tvly-[A-Za-z0-9-]{20}'; then
  echo "ERROR: something that looks like an API key is in the zip" >&2; rm -f "$OUT"; exit 1
fi
[[ -s PROCESS_LOG.md ]] && grep -q "Write this yourself" PROCESS_LOG.md && echo "WARNING: PROCESS_LOG.md is still the template"
if grep -q "^<!-- DRAFT" PROCESS_LOG.md; then
  echo "STOP: PROCESS_LOG.md is still marked DRAFT. Review it, rewrite it in your own words, delete the DRAFT line, then re-run." >&2
  exit 1
fi

echo "$OUT: $(unzip -Z1 "$OUT" | wc -l | tr -d ' ') files, $(du -h "$OUT" | cut -f1)"

# Brightspace wants 1 URL (pasted in the comments) + 5 files. submission/ holds exactly those, and
# submission.zip bundles them.
URL="https://multi-agent-rag-157472143408.us-east4.run.app"
rm -rf submission submission.zip
mkdir submission
cp "$OUT" submission/source-code.zip
cp docs/architecture.md docs/evaluation.md PROCESS_LOG.md README.md submission/
printf 'Live deployment URL (paste into the Brightspace comments):\n%s\n' "$URL" > submission/LIVE_URL.txt
zip -qr submission.zip submission
echo "submission/: $(ls submission | tr '\n' ' ')"
echo "submission.zip: $(du -h submission.zip | cut -f1)"
