"""
benchmarks/extrapolate.py
--------------------------
EXTRAPOLATED KV-cache sizes for longer contexts and larger models.
Every row is labelled EXTRAPOLATED and written to its own file, separate
from measured results.

Method (two inputs, both traceable):
  1. measured ratio  = kv_bytes_real / kv_bytes_fp16   per method, from
     results/raw_results.jsonl (real tensor .nbytes of real runs; the ratio
     already includes scales, zero-points and index overhead)
  2. FP16 bytes/token = 2 (K,V) x layers x kv_heads x head_dim x 2 bytes,
     read from each model's config.json (AutoConfig; no weights downloaded)
  extrapolated bytes = ratio x FP16 bytes/token x context length

Assumption (stated in every row): the measured ratio carries over to other
context lengths and models. That is plausible for uniform methods (overhead
per value is constant) but is NOT verified for adaptive/recency.

Models: `extrapolate_models` in config.yaml.
Outputs: results/extrapolated.csv
"""

import csv
import json
import os
import sys

import yaml
from transformers import AutoConfig

ROOT        = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_PATH    = os.path.join(RESULTS_DIR, "raw_results.jsonl")
EXT_PATH    = os.path.join(RESULTS_DIR, "extrapolated.csv")
EXTRAP_CONTEXTS = [4096, 8192, 16384, 32768]


def fp16_bytes_per_token(model_id: str) -> tuple:
    c = AutoConfig.from_pretrained(model_id)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    shape = (c.num_hidden_layers, c.num_key_value_heads, head_dim)
    return 2 * shape[0] * shape[1] * shape[2] * 2, shape


def load_records(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extrapolate():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    models = cfg.get("extrapolate_models", [cfg["model_id"]])
    if not os.path.exists(RAW_PATH):
        sys.exit(f"ERROR: {RAW_PATH} not found. Run benchmarks/run_baselines.py first.")

    ratios: dict = {}
    for r in load_records(RAW_PATH):
        if r.get("oom") or r.get("error") or r.get("metric"):
            continue
        if r.get("kv_bytes_real") and r.get("kv_bytes_fp16"):
            ratios.setdefault(r["method"], []).append(r["kv_bytes_real"] / r["kv_bytes_fp16"])
    if not ratios:
        sys.exit("ERROR: no usable measured records.")
    mean_ratio = {m: sum(v) / len(v) for m, v in ratios.items()}

    print("Measured stored/FP16 byte ratio (mean over runs):")
    for m, q in sorted(mean_ratio.items()):
        print(f"  {m:18s} {q:.4f}  (n={len(ratios[m])})")

    rows = []
    for model_id in models:
        bpt, (L, H, D) = fp16_bytes_per_token(model_id)
        print(f"{model_id}: {L} layers x {H} KV heads x {D} head_dim -> {bpt} FP16 bytes/token (from config)")
        for method, q in sorted(mean_ratio.items()):
            for ctx in EXTRAP_CONTEXTS:
                fp16 = bpt * ctx
                rows.append({
                    "LABEL": "EXTRAPOLATED",
                    "model": model_id, "method": method, "context_length": ctx,
                    "fp16_bytes_per_token_from_config": bpt,
                    "measured_ratio": q, "n_measured_runs": len(ratios[method]),
                    "kv_MB_extrap": q * fp16 / 1024 ** 2,
                    "fp16_MB_extrap": fp16 / 1024 ** 2,
                    "note": "EXTRAPOLATED, NOT a real run: measured ratio x FP16 size from config; "
                            "assumes the ratio carries over (unverified for adaptive/recency).",
                })

    with open(EXT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten: {EXT_PATH} ({len(rows)} EXTRAPOLATED rows)")


if __name__ == "__main__":
    extrapolate()
