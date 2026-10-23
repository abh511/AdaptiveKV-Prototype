"""
benchmarks/summarise.py
------------------------
Read results/raw_results.jsonl, compute mean ± std per (method, context_length),
write results/summary.csv.

All statistics are computed from the raw records — no values are hardcoded.
"""

import json
import os
import sys
import csv
import math
from collections import defaultdict
from typing import List, Optional

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RAW_PATH    = os.path.join(RESULTS_DIR, "raw_results.jsonl")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")


def mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else None


def std(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m)**2 for v in vals) / (len(vals) - 1))


def load_records(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarise():
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: {RAW_PATH} not found. Run run_baselines.py first.")
        sys.exit(1)

    records = load_records(RAW_PATH)
    print(f"Loaded {len(records)} records from {RAW_PATH}")

    # Group by (method, context_length)
    groups: dict = defaultdict(list)
    for r in records:
        if r.get("oom") or r.get("error") or r.get("metric") == "perplexity":
            continue
        key = (r.get("method", "?"), r.get("context_length", 0))
        groups[key].append(r)

    # Metrics to aggregate
    metrics = [
        "kv_bytes_real", "kv_bytes_fp16", "compression_ratio", "effective_bits",
        "token_agreement", "tier_dist_int8", "tier_dist_int4",
        "peak_vram_bytes", "gen_latency_s", "tokens_per_sec",
        "prefill_s", "decode_ms_per_tok",
    ]

    rows = []
    for (method, ctx_len), recs in sorted(groups.items()):
        row = {
            "method": method,
            "context_length": ctx_len,
            "n_runs": len(recs),
            "n_oom": sum(1 for r in records
                         if r.get("method") == method
                         and r.get("context_length") == ctx_len
                         and r.get("oom")),
            "n_error": sum(1 for r in records
                           if r.get("method") == method
                           and r.get("context_length") == ctx_len
                           and r.get("error")),
        }
        for m in metrics:
            vals = [r.get(m) for r in recs]
            row[f"{m}_mean"] = mean(vals)
            row[f"{m}_std"]  = std(vals)
        rows.append(row)

    if not rows:
        print("No non-OOM records found. Check raw_results.jsonl.")
        return

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written: {SUMMARY_CSV}")
    print(f"\n{'Method':<18} {'Ctx':>6} {'Runs':>5} {'eff_bits':>9} "
          f"{'kv_KB':>9} {'ratio':>7} {'tok_agree':>10} {'lat_s':>8}")
    print("-" * 80)
    for r in rows:
        kv_kb  = (r["kv_bytes_real_mean"] or 0) / 1024
        ratio  = r["compression_ratio_mean"] or 0
        ebits  = r["effective_bits_mean"] or 0
        agree  = r["token_agreement_mean"]
        lat    = r["gen_latency_s_mean"] or 0
        agree_s = f"{agree*100:.1f}%" if agree is not None else "N/A"
        print(f"{r['method']:<18} {r['context_length']:>6} {r['n_runs']:>5} "
              f"{ebits:>9.2f} {kv_kb:>9.1f} {ratio:>7.2f} {agree_s:>10} {lat:>8.2f}")

    # Print OOM summary
    oom_records = [r for r in records if r.get("oom") or r.get("error")]
    if oom_records:
        print(f"\n--- OOM / Error records ({len(oom_records)}) ---")
        for r in oom_records[:10]:  # first 10
            print(f"  method={r.get('method')} ctx={r.get('context_length')} "
                  f"reason={(r.get('oom_reason') or r.get('error') or 'unknown')[:80]}")
        if len(oom_records) > 10:
            print(f"  ... and {len(oom_records)-10} more")


if __name__ == "__main__":
    summarise()
