#!/usr/bin/env python3
"""Aggregate LLM/VLM call JSONLs into a per-episode efficiency table.

Reads ``$SKILLNAV_LOG_DIR/{llm,vlm}_calls_<RUN_ID>.jsonl`` (or paths
passed on the command line) and emits per-episode metrics suitable for
EMNLP Table 2 (Efficiency).

Per-episode metrics produced:

  - llm_calls            : count of LLM invocations
  - llm_prompt_tokens    : sum of input tokens (exact when provider gave
                           usage; approximate ~chars/4 otherwise)
  - llm_response_tokens  : sum of output tokens
  - llm_total_tokens     : prompt + response
  - llm_latency_ms_p50   : median LLM latency
  - llm_latency_ms_p95   : 95th percentile LLM latency
  - vlm_calls            : count of VLM invocations (e.g. BLIP-2 ITM)
  - vlm_latency_ms_p50/p95

Outputs both a per-episode CSV and a summary line for the table.

Usage:
    python scripts/aggregate_calls.py --run-id skillnav_v1
    python scripts/aggregate_calls.py --llm-jsonl path1.jsonl path2.jsonl \\
                                      --vlm-jsonl path3.jsonl \\
                                      --out out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from statistics import median
from typing import Dict, Iterable, List, Optional


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _read_jsonl(paths: Iterable[str]) -> List[dict]:
    records: List[dict] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[aggregate] warning: missing {p}", file=sys.stderr)
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[aggregate] skipping bad line in {p}: {e}", file=sys.stderr)
    return records


def aggregate(llm_records: List[dict], vlm_records: List[dict]) -> Dict:
    """Return (per_episode_dict, summary_dict)."""
    # Per-episode buckets ----------------------------------------------------
    by_ep: Dict = defaultdict(lambda: {
        "scene": None,
        "target": None,
        "llm_calls": 0,
        "llm_prompt_tokens": 0,
        "llm_response_tokens": 0,
        "llm_latencies": [],
        "vlm_calls": 0,
        "vlm_latencies": [],
        "vlm_by_endpoint": defaultdict(int),
    })

    for r in llm_records:
        ep = r.get("episode_id")
        bucket = by_ep[ep]
        bucket["scene"] = bucket["scene"] or r.get("scene")
        bucket["target"] = bucket["target"] or r.get("target")
        bucket["llm_calls"] += 1
        bucket["llm_prompt_tokens"] += int(r.get("prompt_tokens") or 0)
        bucket["llm_response_tokens"] += int(r.get("response_tokens") or 0)
        bucket["llm_latencies"].append(float(r.get("latency_ms") or 0.0))

    for r in vlm_records:
        ep = r.get("episode_id")
        bucket = by_ep[ep]
        bucket["scene"] = bucket["scene"] or r.get("scene")
        bucket["target"] = bucket["target"] or r.get("target")
        bucket["vlm_calls"] += 1
        bucket["vlm_latencies"].append(float(r.get("latency_ms") or 0.0))
        bucket["vlm_by_endpoint"][r.get("endpoint") or "?"] += 1

    # Finalise per-episode rows ---------------------------------------------
    rows = []
    for ep, b in sorted(by_ep.items(), key=lambda kv: (kv[0] is None, kv[0])):
        rows.append({
            "episode_id": ep,
            "scene": b["scene"],
            "target": b["target"],
            "llm_calls": b["llm_calls"],
            "llm_prompt_tokens": b["llm_prompt_tokens"],
            "llm_response_tokens": b["llm_response_tokens"],
            "llm_total_tokens": b["llm_prompt_tokens"] + b["llm_response_tokens"],
            "llm_latency_ms_p50": _percentile(b["llm_latencies"], 50),
            "llm_latency_ms_p95": _percentile(b["llm_latencies"], 95),
            "vlm_calls": b["vlm_calls"],
            "vlm_latency_ms_p50": _percentile(b["vlm_latencies"], 50),
            "vlm_latency_ms_p95": _percentile(b["vlm_latencies"], 95),
        })

    # Summary across episodes ----------------------------------------------
    if rows:
        n = len(rows)
        summary = {
            "n_episodes": n,
            "llm_calls_per_ep": sum(r["llm_calls"] for r in rows) / n,
            "llm_total_tokens_per_ep": sum(r["llm_total_tokens"] for r in rows) / n,
            "llm_prompt_tokens_per_ep": sum(r["llm_prompt_tokens"] for r in rows) / n,
            "llm_response_tokens_per_ep": sum(r["llm_response_tokens"] for r in rows) / n,
            "vlm_calls_per_ep": sum(r["vlm_calls"] for r in rows) / n,
            "llm_latency_ms_p50": _percentile(
                [r["llm_latency_ms_p50"] for r in rows if r["llm_calls"] > 0], 50
            ),
            "llm_latency_ms_p95": _percentile(
                [r["llm_latency_ms_p95"] for r in rows if r["llm_calls"] > 0], 95
            ),
            "vlm_latency_ms_p50": _percentile(
                [r["vlm_latency_ms_p50"] for r in rows if r["vlm_calls"] > 0], 50
            ),
            "vlm_latency_ms_p95": _percentile(
                [r["vlm_latency_ms_p95"] for r in rows if r["vlm_calls"] > 0], 95
            ),
        }
    else:
        summary = {"n_episodes": 0}

    return {"rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=os.environ.get("SKILLNAV_RUN_ID", "default"),
                    help="resolve $SKILLNAV_LOG_DIR/{llm,vlm}_calls_<RUN_ID>.jsonl")
    ap.add_argument("--log-dir", default=os.environ.get("SKILLNAV_LOG_DIR", "/tmp/skillnav_logs"))
    ap.add_argument("--llm-jsonl", nargs="*", default=None)
    ap.add_argument("--vlm-jsonl", nargs="*", default=None)
    ap.add_argument("--out", default=None,
                    help="per-episode CSV path (default: stdout summary only)")
    args = ap.parse_args()

    llm_paths = args.llm_jsonl or [os.path.join(args.log_dir, f"llm_calls_{args.run_id}.jsonl")]
    vlm_paths = args.vlm_jsonl or [os.path.join(args.log_dir, f"vlm_calls_{args.run_id}.jsonl")]

    llm_records = _read_jsonl(llm_paths)
    vlm_records = _read_jsonl(vlm_paths)

    result = aggregate(llm_records, vlm_records)

    if args.out:
        if result["rows"]:
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()))
                writer.writeheader()
                writer.writerows(result["rows"])
            print(f"[aggregate] wrote {len(result['rows'])} rows to {args.out}")
        else:
            print("[aggregate] no rows to write")

    summary = result["summary"]
    print("\n=== Summary (per-episode averages) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:10.2f}")
        else:
            print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
