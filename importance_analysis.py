"""
importance_analysis.py
----------------------
Does EMA attention importance predict which tokens the model will attend to
NEXT? This explains (rather than just reports) the adaptive-vs-recency result.

Setup: FP16 cache (no quantization, so the attention pattern is the model's
own). For each window, prefill P tokens, then teacher-force T tokens and record
the per-token attention vector at every step (same reduction as the adaptive
method: mean over heads and the last query row, uniform over layers).

At decision times t, each policy keeps k = ceil(p8 * n_t) of the n_t cached
tokens "at high precision":
  * ema      top-k by EMA importance (exactly what adaptive uses, alpha from config)
  * recency  the newest k tokens
  * oracle   top-k by the actual attention mass of the next h steps (upper bound)
Pinned tokens (first sink_tokens, last recent_tokens) are INT8 under every
policy, so they are excluded: k = ceil(p8 * n_t) - n_pinned free slots are
chosen among the unpinned tokens only (mirrors TierAllocator). Without this,
the attention sink (token 0 takes most of the mass) makes any ranking that
keeps it look near-perfect.
Score = attention-mass recall over UNPINNED tokens: future attention mass on
the chosen tokens / future mass on all unpinned tokens. Also reported as a
fraction of the oracle.
CIs: bootstrap 95% over windows. EMA vs recency: paired Wilcoxon over windows.

Outputs: results/importance_<tag>.csv, results/IMPORTANCE_REPORT_<tag>.md
Settings: `importance_analysis` block in config.yaml.
"""

import argparse
import csv
import datetime
import math
import os
import platform
import sys

import numpy as np
import torch
import transformers
from scipy.stats import wilcoxon
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptivekv.generate import load_config  # noqa: E402
from adaptivekv.importance import AttentionHook  # noqa: E402
from decision_test import load_model, make_windows, boot_ci  # noqa: E402
from adaptivekv.safety import Guard  # noqa: E402


def attention_trace(model, prefix_ids, forced, window):
    """List of per-token attention vectors: [prefill (P,), step 1 (P+1,), ...]."""
    hook = AttentionHook(model, window=window)
    trace = []
    try:
        hook.register()
        with torch.no_grad():
            out = model(prefix_ids, use_cache=True, output_attentions=False, logits_to_keep=1)
        trace.append(hook.get_scores())
        hook.reset()
        pkv = out.past_key_values
        del out
        for tok_id in forced:
            x = torch.tensor([[tok_id]], device=prefix_ids.device)
            with torch.no_grad():
                out = model(x, past_key_values=pkv, use_cache=True, output_attentions=False)
            trace.append(hook.get_scores())
            hook.reset()
            pkv = out.past_key_values
            del out
    finally:
        hook.remove()
    return trace


def ema_series(trace, alpha):
    """EMA importance after each step, same rule as EMATracker."""
    imp, out = None, []
    for s in trace:
        imp = s.copy() if imp is None else np.concatenate(
            [alpha * s[:len(imp)] + (1 - alpha) * imp, s[len(imp):]])
        out.append(imp.copy())
    return out


def recall_at(trace, ema, t, h, p8, sink=0, recent=0):
    """Recall of future attention mass over UNPINNED tokens for each policy at decision index t."""
    imp = ema[t]
    n = len(imp)
    free = np.arange(min(sink, n), max(min(sink, n), n - recent))   # unpinned positions, ascending
    k = max(0, math.ceil(p8 * n - 1e-9) - (n - len(free)))
    future = sum(trace[s][:n] for s in range(t + 1, t + 1 + h))
    total = future[free].sum()
    if k == 0 or total <= 0:
        return {"ema": float("nan"), "recency": float("nan"), "oracle": float("nan"), "k_free": k}
    sets = {
        "ema": free[np.argsort(-imp[free], kind="stable")[:k]],
        "recency": free[-k:],
        "oracle": free[np.argsort(-future[free], kind="stable")[:k]],
    }
    out = {name: float(future[idx].sum() / total) for name, idx in sets.items()}
    out["k_free"] = k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpu-mem-gb", type=float, default=None)
    ap.add_argument("--cpu-mem-gb", type=float, default=6.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ia = cfg["importance_analysis"]
    model_id = args.model or cfg["model_id"]
    tag = model_id.split("/")[-1]
    rdir = args.out_dir or cfg.get("results_dir", "results")
    os.makedirs(rdir, exist_ok=True)
    P, T, h = ia["prefill_len"], ia["eval_len"], ia["horizon"]
    alpha, win = cfg["ema_alpha"], cfg.get("importance_window", 1)
    sink, recent = cfg.get("sink_tokens", 0), cfg.get("recent_tokens", 0)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id, cfg["attn_impl"], args.gpu_mem_gb, args.cpu_mem_gb)
    ids = tok(open(cfg["wikitext_file"], encoding="utf-8").read()).input_ids
    windows = make_windows(ids, ia["n_windows"], P + T)
    print(f"model {model_id} | {len(windows)} windows | P={P} T={T} horizon={h}")

    guard = Guard(cfg)
    rows = []
    for wi, w in enumerate(windows):
        guard()
        x = torch.tensor([w[:P]], device="cuda")
        trace = attention_trace(model, x, w[P:P + T], win)
        ema = ema_series(trace, alpha)
        for p8 in ia["p8_values"]:
            for t in range(0, len(trace) - h, ia["decision_every"]):
                rec = recall_at(trace, ema, t, h, p8, sink, recent)
                rows.append({"window": wi, "p8": p8, "t": t, **rec})
        print(f"  window {wi + 1}/{len(windows)}", flush=True)
        del trace, ema
        torch.cuda.empty_cache()

    path = os.path.join(rdir, f"importance_{tag}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    rng = np.random.default_rng(cfg["seed"])
    out = []
    W = out.append
    W(f"# Importance analysis: {model_id}\n")
    W("Generated by `importance_analysis.py`. Every number below was measured in this run. Do not edit by hand.\n")
    W("## Setup\n")
    W("| Item | Value |\n|---|---|")
    W(f"| Date | {datetime.datetime.now().isoformat(timespec='seconds')} |")
    W(f"| Model | {model_id}, FP16 KV cache (attention not perturbed by quantization) |")
    W(f"| Software | Python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__} |")
    W(f"| Text | {cfg['wikitext_file']}, {len(windows)} windows, {P} prefill + {T} teacher-forced tokens |")
    W(f"| Importance | mean over heads of last {win} query row(s), uniform over layers; EMA alpha {alpha} |")
    W(f"| Decision times | every {ia['decision_every']} steps; future horizon h = {h} steps |")
    W(f"| Pinned (excluded, INT8 under every policy) | first {sink} + last {recent} tokens |")
    W(f"| Hardware safety | {guard.summary()} |\n")
    W("## Result: share of future attention mass (unpinned tokens) on the tokens each policy keeps at INT8\n")
    W("Higher is better. 'vs oracle' = policy recall / oracle recall. Mean over windows (each window "
      "averaged over its decision times), bootstrap 95% CI over windows.\n")
    W("| p8 (kept share) | EMA (adaptive) | Recency | Oracle | EMA vs oracle | Recency vs oracle | EMA − recency, Wilcoxon p |")
    W("|---|---|---|---|---|---|---|")
    for p8 in ia["p8_values"]:
        per = {}
        for name in ("ema", "recency", "oracle"):
            per[name] = np.array([np.nanmean([r[name] for r in rows if r["window"] == wi and r["p8"] == p8])
                                  for wi in range(len(windows))])
        cell = {}
        for name, v in per.items():
            lo, hi = boot_ci(v, np.mean, cfg["decision_test"]["bootstrap"], rng)
            cell[name] = f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]"
        d = per["ema"] - per["recency"]
        pv = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
        W(f"| {p8:.2f} | {cell['ema']} | {cell['recency']} | {cell['oracle']} | "
          f"{per['ema'].mean() / per['oracle'].mean():.3f} | {per['recency'].mean() / per['oracle'].mean():.3f} | "
          f"{d.mean():+.3f}, p={pv:.3g} |")
    W("")
    W("## How to read this\n")
    W("- If EMA recall is not clearly above recency recall, attention-based ranking cannot be expected "
      "to beat the recency heuristic on quality: it is not keeping more of the attention the model will use.")
    W("- The gap between the oracle and both policies is the headroom any importance predictor could still gain.")
    W("- Attention mass is a proxy for a token's influence; it ignores the size of value vectors.")
    rep = os.path.join(rdir, f"IMPORTANCE_REPORT_{tag}.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwritten: {path}\n         {rep}")
    print(f"safety: {guard.summary()}")


if __name__ == "__main__":
    main()
