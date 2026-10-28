"""Measure cold-start latency on the deployed service (min-instances=0).

    python -m src.eval.cold_start --probes 3 --idle-min 20

Cloud Run shuts idle instances down after roughly 15 minutes without traffic. Each probe waits --idle-min
minutes since the previous request, sends one RAG question in a fresh session, and records client-side
first byte / TTFT / total. It then reads Cloud Run's logs to confirm the request really started a new
instance ("Starting new instance") and to split the cold start into container start → app startup.
Results are appended to src/eval/results/cold_starts.jsonl.
"""

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone

from src.config import ROOT
from src.eval.collect import DEFAULT_URL, load_chunk_texts, run_question

PROJECT = "csci599-a2-617204"
SERVICE = "multi-agent-rag"
OUT = ROOT / "src/eval/results/cold_starts.jsonl"
PROBE = {"id": "cold_probe", "topic": "rag", "docs": ["arxiv:2307.03172"],
         "question": "According to 'Lost in the Middle', how does the position of relevant information in a long "
                     "context affect language model performance?"}


def startup_events(since: datetime, until: datetime) -> list[tuple[str, str]]:
    """(timestamp, message) of instance-start and app-startup log lines in the window."""
    flt = (f'resource.type="cloud_run_revision" AND resource.labels.service_name="{SERVICE}" AND '
           f'timestamp>="{since.isoformat()}" AND timestamp<="{until.isoformat()}" AND '
           '(textPayload:"Starting new instance" OR textPayload:"Waiting for application startup" OR '
           'textPayload:"Application startup complete" OR textPayload:"warm-up done")')
    out = subprocess.run(["gcloud", "logging", "read", flt, "--project", PROJECT, "--order", "asc",
                          "--format", "value(timestamp,textPayload)"], capture_output=True, text=True, check=True)
    return [tuple(line.split("\t", 1)) for line in out.stdout.splitlines() if "\t" in line]


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


async def probe(chunk_texts: dict[str, str], url: str) -> dict:
    import httpx

    started = datetime.now(timezone.utc)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=120)) as client:
        rec = await run_question(client, url, PROBE, "auto", "balanced", chunk_texts)
    await asyncio.sleep(90)  # Cloud Logging ingestion delay
    events = startup_events(started - timedelta(seconds=5), started + timedelta(seconds=120))
    phases = {}
    for ts, msg in events:
        for key, needle in (("instance_start", "Starting new instance"), ("app_start", "Waiting for application"),
                            ("ready", "Application startup complete")):
            if needle in msg and key not in phases:
                phases[key] = _ts(ts)
    breakdown = {}
    if {"instance_start", "app_start", "ready"} <= phases.keys():
        breakdown = {
            "container_and_imports_s": round((phases["app_start"] - phases["instance_start"]).total_seconds(), 2),
            "app_startup_s": round((phases["ready"] - phases["app_start"]).total_seconds(), 2),
            "request_to_instance_start_s": round((phases["instance_start"] - started).total_seconds(), 2),
        }
    return {"at": started.isoformat(timespec="seconds"), "cold_confirmed": "instance_start" in phases,
            "timing": rec["timing"], "latency": rec["latency"], "startup": breakdown,
            "agents": rec["agents"], "error": rec["error"]}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=3)
    ap.add_argument("--idle-min", type=float, default=20)
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()
    chunk_texts = load_chunk_texts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.probes):
        print(f"probe {i + 1}/{args.probes}: idling {args.idle_min} min (until "
              f"{(datetime.now() + timedelta(minutes=args.idle_min)):%H:%M})", flush=True)
        await asyncio.sleep(args.idle_min * 60)
        result = await probe(chunk_texts, args.url)
        with open(OUT, "a") as f:
            f.write(json.dumps(result) + "\n")
        t = result["timing"]
        print(f"  cold={result['cold_confirmed']} first_byte={t['first_byte_s']}s ttft={t['ttft_s']}s "
              f"total={t['total_s']}s startup={result['startup']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
