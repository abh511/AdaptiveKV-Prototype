"""
passkey.py
----------
Needle-in-a-haystack retrieval through each KV-cache method.

A random 5-digit pass key is hidden at a chosen depth inside real WikiText-2
text; the model is then asked for it (greedy decoding). A trial is correct if
the generated answer contains the exact key. This tests what perplexity cannot:
whether a single old but important token survives quantization. The recency
heuristic demotes old tokens by construction, so this is where
attention-based importance should matter.

Metrics: accuracy per (method, context length, depth) and overall, with Wilson
95% CIs; adaptive vs recency paired on identical prompts with an exact McNemar
test (binomial test on discordant pairs).

Outputs (generated, never edit by hand):
  results/passkey_<tag>.csv          one row per (method, trial)
  results/PASSKEY_REPORT_<tag>.md    test cases, tables, verdicts

Settings: `passkey` block in config.yaml.
Usage:
  python passkey.py
  python passkey.py --model Qwen/Qwen2.5-3B-Instruct --gpu-mem-gb 2.8
"""

import argparse
import csv
import datetime
import math
import os
import platform
import re
import sys
import time

import numpy as np
import torch
import transformers
from scipy.stats import binomtest
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptivekv.generate import load_config, run_method  # noqa: E402
from decision_test import load_model  # noqa: E402
from adaptivekv.safety import Guard  # noqa: E402

INTRO = "There is a pass key hidden inside the following text. Find it and remember it.\n\n"
NEEDLE = " The pass key is {key}. Remember it. {key} is the pass key. "
QUESTION = "\n\nWhat is the pass key? Answer with the number only."


def build_variants(pk: dict) -> list:
    """(label, method, cfg overrides): two-tier and (optional) three-tier mixes."""
    v = [("fp16", "fp16", {}),
         ("int8_uniform", "int8_uniform", {}),
         ("int4_uniform", "int4_uniform", {})]
    for p8, p4 in pk["fractions"]:
        over = {"p8": p8, "p4": p4, "int2_enabled": False}
        for m in ("adaptive", "recency"):
            v.append((f"{m} 8/4 {p8:.2f}/{p4:.2f}", m, over))
    for p8, p4, p2 in pk.get("three_tier", []):
        over = {"p8": p8, "p4": p4, "p2": p2, "int2_enabled": True}
        for m in ("adaptive", "recency"):
            v.append((f"{m} 8/4/2 {p8:.2f}/{p4:.2f}/{p2:.2f}", m, over))
    return v


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def make_prompt(tok, filler_ids: list, ctx: int, depth: float, key: int) -> list:
    """Chat prompt of ~ctx tokens with the needle at relative depth `depth` of the filler."""
    def wrap(body: str) -> list:
        msgs = [{"role": "user", "content": INTRO + body + QUESTION}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tok(text, add_special_tokens=False).input_ids

    needle = NEEDLE.format(key=key)
    overhead = len(wrap(needle))
    n_fill = max(16, ctx - overhead)
    fill = filler_ids[:n_fill]
    cut = int(round(depth * len(fill)))
    body = tok.decode(fill[:cut]) + needle + tok.decode(fill[cut:])
    return wrap(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpu-mem-gb", type=float, default=None)
    ap.add_argument("--cpu-mem-gb", type=float, default=6.0)
    ap.add_argument("--trials", type=int, default=None, help="override passkey.trials_per_depth")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pk = cfg["passkey"]
    model_id = args.model or cfg["model_id"]
    trials = args.trials or pk["trials_per_depth"]
    tag = model_id.split("/")[-1]
    rdir = args.out_dir or cfg.get("results_dir", "results")
    os.makedirs(rdir, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id, cfg["attn_impl"], args.gpu_mem_gb, args.cpu_mem_gb)
    device = torch.device("cuda")
    filler_all = tok(open(cfg["wikitext_file"], encoding="utf-8").read()).input_ids
    variants = build_variants(pk)

    # test cases: identical prompts for every method
    cases = []
    for ctx in pk["context_lengths"]:
        for depth in pk["depths"]:
            for t in range(trials):
                key = int(rng.integers(10000, 100000))
                start = int(rng.integers(0, len(filler_all) - ctx - 1))
                ids = make_prompt(tok, filler_all[start:start + ctx], ctx, depth, key)
                cases.append({"ctx": ctx, "depth": depth, "trial": t, "key": key,
                              "filler_start": start, "ids": ids})
    print(f"model {model_id} | {len(cases)} prompts x {len(variants)} methods")

    run_method(model, tok, torch.tensor([cases[0]["ids"]], device=device), "fp16", cfg, 2)   # warm-up

    guard = Guard(cfg)
    rows = []
    t0 = time.time()
    for ci, c in enumerate(cases):
        x = torch.tensor([c["ids"]], device=device)
        for label, method, over in variants:
            row = {"variant": label, "method": method, "case": ci, "ctx_target": c["ctx"],
                   "prompt_tokens": len(c["ids"]), "depth": c["depth"], "trial": c["trial"],
                   "key": c["key"], "filler_start": c["filler_start"]}
            guard()
            try:
                r = run_method(model, tok, x, method, {**cfg, **over}, pk["max_new_tokens"])
                ans = tok.decode(r["gen"], skip_special_tokens=True)
                row.update({"answer": ans.strip()[:40],
                            "correct": int(str(c["key"]) in re.sub(r"\D", "", ans)),
                            "eff_bits": r["eff_bits"], "error": ""})
            except torch.cuda.OutOfMemoryError as e:
                row.update({"answer": "", "correct": None, "eff_bits": None,
                            "error": f"CUDA OOM: {str(e)[:80]}"})
                torch.cuda.empty_cache()
            rows.append(row)
        if (ci + 1) % 10 == 0 or ci + 1 == len(cases):
            print(f"  {ci + 1}/{len(cases)} prompts ({(time.time() - t0) / 60:.1f} min)", flush=True)

    path = os.path.join(rdir, f"passkey_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    report = os.path.join(rdir, f"PASSKEY_REPORT_{tag}.md")
    write_report(report, cfg, pk, model_id, model, variants, cases, rows, trials, args, guard)
    print(f"safety: {guard.summary()}")
    print(f"\nwritten: {path}\n         {report}")


def write_report(path, cfg, pk, model_id, model, variants, cases, rows, trials, args, guard):
    ok = [r for r in rows if not r["error"]]
    out = []
    w = out.append
    w(f"# Passkey retrieval: {model_id}\n")
    w("Generated by `passkey.py`. Every number below was measured in this run. Do not edit by hand.\n")
    w("## Setup\n")
    w("| Item | Value |\n|---|---|")
    w(f"| Date | {datetime.datetime.now().isoformat(timespec='seconds')} |")
    w(f"| Model | {model_id} (FP16 weights, {cfg['attn_impl']} attention, "
      f"{'partly CPU-offloaded' if args.gpu_mem_gb else 'fully on GPU'}) |")
    w(f"| GPU | {torch.cuda.get_device_name(0)} |")
    w(f"| Software | Python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__} |")
    w(f"| Haystack | {cfg['wikitext_file']} (real text), random start per prompt |")
    w(f"| Context lengths | {pk['context_lengths']} (target; actual prompt length is in the CSV) |")
    w(f"| Depths | {pk['depths']} (0 = start of text, 1 = end) |")
    w(f"| Trials per (context, depth) | {trials}, random 5-digit key each, seed {cfg['seed']} |")
    w(f"| Decoding | greedy, {pk['max_new_tokens']} new tokens; correct = exact key appears in the answer digits |")
    w(f"| Sink / recent pins, EMA alpha, importance window | {cfg.get('sink_tokens')} / {cfg.get('recent_tokens')}, "
      f"{cfg['ema_alpha']}, {cfg.get('importance_window')} |")
    w(f"| Key quantization | {cfg.get('key_quant', 'per_channel')}, key_group_tokens {cfg.get('key_group_tokens', 32)} |")
    w(f"| Hardware safety | {guard.summary()} |\n")

    w("## Test cases\n")
    w(f"{len(cases)} prompts; every method sees exactly the same prompts.\n")
    w("Needle sentence: `" + NEEDLE.strip().format(key="<key>") + "`\n")
    w("| # | Method | Fractions (INT8/INT4/INT2) |\n|---|---|---|")
    for i, (label, method, over) in enumerate(variants, 1):
        fr = "-" if "p8" not in over else (f"{over['p8']}/{over['p4']}/{over.get('p2', 0.0) if over.get('int2_enabled') else 0.0}")
        w(f"| {i} | {label} | {fr} |")
    w("")

    w("## Overall accuracy\n")
    w("| Method | Bits | Correct / n | Accuracy [Wilson 95% CI] |\n|---|---|---|---|")
    for label, _, _ in variants:
        rs = [r for r in ok if r["variant"] == label]
        k, n = sum(r["correct"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        bits = np.mean([r["eff_bits"] for r in rs]) if rs else float("nan")
        w(f"| {label} | {bits:.2f} | {k} / {n} | {k / n if n else float('nan'):.3f} [{lo:.3f}, {hi:.3f}] |")
    w("")

    w("## Accuracy by context length and depth\n")
    keys = [(c, d) for c in pk["context_lengths"] for d in pk["depths"]]
    w("| Method | " + " | ".join(f"{c} @ {d}" for c, d in keys) + " |")
    w("|---|" + "---|" * len(keys))
    for label, _, _ in variants:
        cells = []
        for c, d in keys:
            rs = [r for r in ok if r["variant"] == label and r["ctx_target"] == c and r["depth"] == d]
            cells.append(f"{sum(r['correct'] for r in rs)}/{len(rs)}")
        w(f"| {label} | " + " | ".join(cells) + " |")
    w("")

    w("## Adaptive vs recency (paired, same prompts)\n")
    w("Exact McNemar test: binomial test on prompts where exactly one of the two is correct. "
      "Significant = p < 0.05.\n")
    w("| Tier mix | Both right | Only adaptive | Only recency | Both wrong | p | Verdict |")
    w("|---|---|---|---|---|---|---|")
    mixes = [f"8/4 {p8:.2f}/{p4:.2f}" for p8, p4 in pk["fractions"]] + \
            [f"8/4/2 {p8:.2f}/{p4:.2f}/{p2:.2f}" for p8, p4, p2 in pk.get("three_tier", [])]
    for mix in mixes:
        a = {r["case"]: r["correct"] for r in ok if r["variant"] == f"adaptive {mix}"}
        b = {r["case"]: r["correct"] for r in ok if r["variant"] == f"recency {mix}"}
        common = set(a) & set(b)
        both = sum(1 for i in common if a[i] and b[i])
        oa = sum(1 for i in common if a[i] and not b[i])
        ob = sum(1 for i in common if b[i] and not a[i])
        none = len(common) - both - oa - ob
        pv = binomtest(oa, oa + ob, 0.5).pvalue if oa + ob else 1.0
        verdict = ("adaptive better" if oa > ob else "recency better") if pv < 0.05 else "no significant difference"
        w(f"| {mix} | {both} | {oa} | {ob} | {none} | {pv:.3g} | {verdict} |")
    w("")
    errs = [r for r in rows if r["error"]]
    if errs:
        w(f"Errors: {len(errs)} runs failed (see CSV `error` column).\n")
    w("## Limitations\n")
    w("- Synthetic retrieval task; it measures whether one specific old fact survives, not general quality.")
    w("- If FP16 itself fails a prompt, no KV method can be expected to pass it; compare against the FP16 row.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
