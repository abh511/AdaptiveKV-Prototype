"""
sweep.py - which precision mix keeps quality? (measured, not assumed)

Computes perplexity THROUGH each method's KV cache (teacher forcing) on real
text, for FP16, uniform INT8/INT4 and several adaptive tier mixes.
Writes results/sweep.csv and prints a table.

Run:  python sweep.py                          # tries WikiText-2 via `datasets`
      python sweep.py --text-file path.txt     # or any real text file
      python sweep.py --windows 3              # faster
"""
import argparse
import csv
import datetime
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptivekv.generate import load_config
from adaptivekv.metrics import compute_perplexity_kv

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--text-file", default=None)
ap.add_argument("--prefill", type=int, default=192)
ap.add_argument("--eval", type=int, default=64)
ap.add_argument("--windows", type=int, default=3)
ap.add_argument("--chars", type=int, default=60000)
args = ap.parse_args()

cfg = load_config(args.config)

LOCAL_DEFAULT = "data/wikitext2_test.txt"
path = args.text_file or (LOCAL_DEFAULT if os.path.exists(LOCAL_DEFAULT) else None)
if path:
    text = open(path, encoding="utf-8").read()[:args.chars]
    source = path
else:
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])[:args.chars]
        source = "Salesforce/wikitext wikitext-2-raw-v1/test"
    except Exception as e:                                   # noqa: BLE001
        raise SystemExit(f"No local text and could not download WikiText-2 ({e}). "
                         f"Pass --text-file <real text file>.")

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_id"], dtype=dtype, device_map="cuda" if torch.cuda.is_available() else "cpu",
    attn_implementation=cfg.get("attn_impl", "eager"), trust_remote_code=True).eval()

# (label, method, config overrides)
variants = [("fp16", "fp16", {}),
            ("int8 uniform", "int8_uniform", {}),
            (f"int4 uniform gs={cfg['quant_group_size']}", "int4_uniform", {}),
            ("int4 uniform gs=32", "int4_uniform", {"quant_group_size": 32}),
            ]
# same tier fractions for adaptive and for the recency heuristic (both keep sink tokens at INT8)
for p8, p4 in [(0.5, 0.5), (0.3, 0.7), (0.6, 0.4), (0.4, 0.6)]:
    tag = f"p8={p8} p4={p4}"
    over = {"p8": p8, "p4": p4, "sink_tokens": 4}
    variants.append((f"adaptive {tag}", "adaptive", over))
    variants.append((f"recency  {tag}", "recency", over))

rows, base = [], None
print(f"text: {source} | window {args.prefill}+{args.eval} tokens x {args.windows} | dtype {dtype} "
      f"| group_size {cfg['quant_group_size']}\n")
print(f"{'variant':46s} {'ppl':>8s} {'vs fp16':>9s} {'eff bits':>9s} {'compress':>9s}  tiers(8/4)")
for label, method, over in variants:
    r = compute_perplexity_kv(model, tok, text, method, {**cfg, **over},
                              prefill_len=args.prefill, eval_len=args.eval, n_windows=args.windows)
    if "reason" in r:
        raise SystemExit(r["reason"])
    if base is None:
        base = r["ppl"]
    comp = r["kv_fp16_equiv_last_window"] / r["kv_total_bytes_last_window"]
    d = r["tier_dist_last_window"]
    print(f"{label:46s} {r['ppl']:8.3f} {(r['ppl'] / base - 1) * 100:+8.1f}% "
          f"{r['effective_bits_last_window']:9.2f} {comp:8.2f}x  {d[8]}/{d[4]}")
    rows.append({"variant": label, "ppl": r["ppl"], "ppl_vs_fp16_pct": (r["ppl"] / base - 1) * 100,
                 "effective_bits": r["effective_bits_last_window"], "compression_x": comp,
                 "tiers_8_4": f"{d[8]}/{d[4]}", "n_windows": r["n_windows_used"],
                 "n_scored": r["n_scored"], "text_source": source, "dtype": str(dtype),
                 "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})

os.makedirs("results", exist_ok=True)
with open("results/sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nwritten: results/sweep.csv")