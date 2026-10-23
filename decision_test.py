"""
decision_test.py
----------------
Answers two design questions with measured numbers, on the SAME text windows:

  Decision 1  two-tier (INT8/INT4) vs three-tier (INT8/INT4/INT2)?
  Question 2  does attention-based importance (adaptive) beat the recency
              heuristic at the same tier fractions?

Every variant is scored by teacher forcing through its own KV cache:
  * perplexity (and % change vs FP16)
  * KL(FP16 || method) of the next-token distribution, same prefix
  * top-1 agreement with FP16, same prefix
  * effective bits / stored bytes (real tensor .nbytes)
  * decode time per token (after a warm-up run)
Uncertainty: bootstrap 95% CI over windows. Adaptive vs recency: paired
per-window differences, bootstrap CI and Wilcoxon signed-rank test.

Outputs (generated, never edit by hand):
  results/decision_<tag>_windows.csv   one row per (variant, window)
  results/decision_<tag>_summary.csv   one row per variant
  results/DECISION_REPORT_<tag>.md     the test cases, tables and verdicts

Settings: `decision_test` block in config.yaml.
Usage:
  python decision_test.py                                   # model from config.yaml
  python decision_test.py --model Qwen/Qwen2.5-3B-Instruct --gpu-mem-gb 3.0   # CPU offload
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import transformers
from scipy.stats import wilcoxon
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptivekv.generate import load_config, run_method  # noqa: E402
from adaptivekv.safety import Guard  # noqa: E402


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def build_variants(dt: dict) -> list:
    """(label, method, cfg overrides, group). Groups: ref, uniform, two_tier, three_tier,
    old_keys (previous per-token key scheme, ablation)."""
    v = [("fp16", "fp16", {}, "ref"),
         ("int8_uniform", "int8_uniform", {}, "uniform"),
         ("int4_uniform", "int4_uniform", {}, "uniform"),
         ("int2_uniform", "int2_uniform", {}, "uniform")]
    for p8, p4 in dt["two_tier"]:
        over = {"p8": p8, "p4": p4, "int2_enabled": False}
        for m in ("adaptive", "recency"):
            v.append((f"{m} 8/4 {p8:.2f}/{p4:.2f}", m, over, "two_tier"))
    for p8, p4, p2 in dt["three_tier"]:
        over = {"p8": p8, "p4": p4, "p2": p2, "int2_enabled": True}
        for m in ("adaptive", "recency"):
            v.append((f"{m} 8/4/2 {p8:.2f}/{p4:.2f}/{p2:.2f}", m, over, "three_tier"))
    if dt.get("old_key_ablation", True):
        old = {"key_quant": "per_token", "k_channel_norm": True}
        for m in ("int4_uniform", "int2_uniform"):
            v.append((f"{m} [per-token keys]", m, old, "old_keys"))
        for m in ("adaptive", "recency"):
            v.append((f"{m} 8/4 0.50/0.50 [per-token keys]", m,
                      {**old, "p8": 0.5, "p4": 0.5, "int2_enabled": False}, "old_keys"))
    return v


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def boot_ci(x: np.ndarray, fn, B: int, rng) -> tuple:
    x = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    stats = np.array([fn(x[i]) for i in idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def wilcoxon_p(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=np.float64)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def pareto_front(points: dict) -> set:
    """points: label -> (bits, loss). A point is on the front if no other point
    has bits <= and loss <= with at least one strictly smaller."""
    front = set()
    for a, (ba, la) in points.items():
        dominated = any((bb <= ba and lb <= la) and (bb < ba or lb < la)
                        for b, (bb, lb) in points.items() if b != a)
        if not dominated:
            front.add(a)
    return front


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_windows(ids: list, n: int, win: int) -> list:
    """n non-overlapping windows of `win` tokens, evenly spaced over the text."""
    max_n = len(ids) // win
    if max_n < n:
        print(f"WARNING: text holds only {max_n} non-overlapping windows of {win} tokens (asked {n})")
        n = max_n
    step = len(ids) // n
    return [ids[i * step:i * step + win] for i in range(n)]


def load_model(model_id: str, attn: str, gpu_mem_gb, cpu_mem_gb):
    kw = dict(dtype=torch.float16, attn_implementation=attn)
    if gpu_mem_gb is not None:
        kw.update(device_map="auto",
                  max_memory={0: f"{gpu_mem_gb}GiB", "cpu": f"{cpu_mem_gb}GiB"})
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to("cuda")
    return model.eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="override model_id from config")
    ap.add_argument("--gpu-mem-gb", type=float, default=None,
                    help="cap GPU memory for weights and offload the rest to CPU (for models > VRAM)")
    ap.add_argument("--windows", type=int, default=None, help="override decision_test.n_windows")
    ap.add_argument("--only", default=None,
                    help="comma-separated variant labels to run (fp16 is always run as the reference)")
    ap.add_argument("--cpu-mem-gb", type=float, default=6.0,
                    help="cap system RAM used for offloaded weights (keeps the laptop responsive)")
    ap.add_argument("--out-dir", default=None, help="override results_dir (e.g. for smoke tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dt = cfg["decision_test"]
    model_id = args.model or cfg["model_id"]
    n_windows = args.windows or dt["n_windows"]
    P, T, B = dt["prefill_len"], dt["eval_len"], dt["bootstrap"]
    tag = model_id.split("/")[-1]
    rdir = args.out_dir or cfg.get("results_dir", "results")
    os.makedirs(rdir, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id, cfg["attn_impl"], args.gpu_mem_gb, args.cpu_mem_gb)
    device = torch.device("cuda")
    mc = model.config
    head_dim = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
    offloaded = args.gpu_mem_gb is not None and any(
        str(d) in ("cpu", "disk") for d in getattr(model, "hf_device_map", {}).values())

    text = open(cfg["wikitext_file"], encoding="utf-8").read()
    ids = tok(text).input_ids
    windows = make_windows(ids, n_windows, P + T)
    variants = build_variants(dt)
    if args.only:
        keep = {x.strip() for x in args.only.split(",")} | {"fp16"}
        unknown = keep - {v[0] for v in variants}
        if unknown:
            raise SystemExit(f"unknown variant labels: {sorted(unknown)}")
        variants = [v for v in variants if v[0] in keep]
    print(f"model {model_id} | layers {mc.num_hidden_layers} kv_heads {mc.num_key_value_heads} "
          f"head_dim {head_dim} | offloaded {offloaded}")
    print(f"{len(windows)} windows x ({P} prefill + {T} scored) | {len(variants)} variants")

    # warm-up (CUDA context, kernels) so the first timed variant is not penalised
    w0 = torch.tensor([windows[0][:P]], device=device)
    run_method(model, tok, w0, "fp16", cfg, 0, forced=windows[0][P:P + 4])
    run_method(model, tok, w0, "int4_uniform", cfg, 0, forced=windows[0][P:P + 4])

    guard = Guard(cfg)
    rows = []
    t_start = time.time()
    for wi, w in enumerate(windows):
        guard()
        prefix = torch.tensor([w[:P]], device=device)
        forced = w[P:P + T]
        ref = run_method(model, tok, prefix, "fp16", cfg, 0, forced=forced, collect_logprobs=True)
        ref_lp = ref["logprobs"]
        for label, method, over, group in variants:
            rcfg = {**cfg, **over}
            row = {"variant": label, "method": method, "group": group, "window": wi}
            guard()
            try:
                r = ref if method == "fp16" else run_method(
                    model, tok, prefix, method, rcfg, 0, forced=forced, ref_logprobs=ref_lp)
                kl = r["kl"] if method != "fp16" else [0.0] * T
                top1 = r["top1"] if method != "fp16" else [True] * T
                td = r["tier_dist"]
                n_tok = sum(td.values()) or 1
                row.update({
                    "nll_mean": float(np.mean(r["nll"])), "kl_mean": float(np.mean(kl)),
                    "top1": float(np.mean(top1)), "eff_bits": r["eff_bits"],
                    "kv_bytes": r["kv_mem"]["total_bytes"], "fp16_bytes": r["kv_mem"]["fp16_equivalent"],
                    "decode_s_per_tok": r["decode_s"] / max(1, T - 1),
                    "peak_vram_bytes": r["peak_vram"],
                    "frac8": td.get(8, 0) / n_tok if method != "fp16" else None,
                    "frac4": td.get(4, 0) / n_tok if method != "fp16" else None,
                    "frac2": td.get(2, 0) / n_tok if method != "fp16" else None,
                    "frac16": td.get(16, 0) / n_tok if method != "fp16" else None,
                    "error": "",
                })
            except torch.cuda.OutOfMemoryError as e:
                row["error"] = f"CUDA OOM: {str(e)[:100]}"
                torch.cuda.empty_cache()
            rows.append(row)
        el = time.time() - t_start
        print(f"  window {wi + 1}/{len(windows)} done  ({el / 60:.1f} min elapsed)", flush=True)
        del ref, ref_lp
        torch.cuda.empty_cache()

    # ---- raw rows ------------------------------------------------------
    wpath = os.path.join(rdir, f"decision_{tag}_windows.csv")
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(wpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- per-variant summary ----------------------------------------------
    def vals(label, key):
        return np.array([r[key] for r in rows if r["variant"] == label and not r["error"]], dtype=np.float64)

    ppl_fn = lambda x: math.exp(x.mean())
    base_nll = vals("fp16", "nll_mean")
    base_ppl = ppl_fn(base_nll)
    summary = []
    for label, method, over, group in variants:
        nll = vals(label, "nll_mean")
        n_err = sum(1 for r in rows if r["variant"] == label and r["error"])
        if len(nll) == 0:
            summary.append({"variant": label, "group": group, "n_windows": 0, "n_errors": n_err})
            continue
        kl, top1 = vals(label, "kl_mean"), vals(label, "top1")
        ppl = ppl_fn(nll)
        lo, hi = boot_ci(nll, ppl_fn, B, rng)
        klo, khi = boot_ci(kl, np.mean, B, rng)
        tlo, thi = boot_ci(top1, np.mean, B, rng)
        summary.append({
            "variant": label, "group": group, "method": method,
            "n_windows": len(nll), "n_errors": n_err,
            "eff_bits": float(vals(label, "eff_bits").mean()),
            "compression_x": float((vals(label, "fp16_bytes") / vals(label, "kv_bytes")).mean()),
            "ppl": ppl, "ppl_ci_lo": lo, "ppl_ci_hi": hi,
            "ppl_delta_pct": (ppl / base_ppl - 1) * 100,
            "kl": float(kl.mean()), "kl_ci_lo": klo, "kl_ci_hi": khi,
            "top1": float(top1.mean()), "top1_ci_lo": tlo, "top1_ci_hi": thi,
            "decode_ms_per_tok": float(vals(label, "decode_s_per_tok").mean() * 1000),
            "peak_vram_gb": float(vals(label, "peak_vram_bytes").max() / 1024 ** 3),
            "frac8": float(vals(label, "frac8").mean()) if method != "fp16" else None,
            "frac4": float(vals(label, "frac4").mean()) if method != "fp16" else None,
            "frac2": float(vals(label, "frac2").mean()) if method != "fp16" else None,
            "frac16": float(vals(label, "frac16").mean()) if method != "fp16" else None,
        })
    spath = os.path.join(rdir, f"decision_{tag}_summary.csv")
    sfields = list(max(summary, key=len).keys())
    with open(spath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary)

    # ---- paired adaptive vs recency ---------------------------------------
    paired = []
    for label, method, over, group in variants:
        if method != "adaptive":
            continue
        rlabel = label.replace("adaptive", "recency", 1)
        a = {r["window"]: r for r in rows if r["variant"] == label and not r["error"]}
        b = {r["window"]: r for r in rows if r["variant"] == rlabel and not r["error"]}
        common = sorted(set(a) & set(b))
        if len(common) < 2:
            continue
        rec = {"fractions": label.split(" ", 1)[1], "n": len(common)}
        for key in ("nll_mean", "kl_mean", "top1"):
            d = np.array([a[i][key] - b[i][key] for i in common])
            lo, hi = boot_ci(d, np.mean, B, rng)
            rec[key] = (float(d.mean()), lo, hi, wilcoxon_p(d))
        rec["bits"] = (float(np.mean([a[i]["eff_bits"] for i in common])),
                       float(np.mean([b[i]["eff_bits"] for i in common])))
        paired.append(rec)

    # ---- bit-matched adaptive vs recency (equal-memory comparison) ----------
    sm = {s["variant"]: s for s in summary if s.get("n_windows")}
    rec_labels = [l for l in sm if sm[l]["method"] == "recency" and "[per-token keys]" not in l]
    matched = []
    for label in sm:
        if sm[label]["method"] != "adaptive" or "[per-token keys]" in label or not rec_labels:
            continue
        r = min(rec_labels, key=lambda x: abs(sm[x]["eff_bits"] - sm[label]["eff_bits"]))
        gap = sm[label]["eff_bits"] - sm[r]["eff_bits"]
        if abs(gap) > dt.get("bit_match_tolerance", 0.15):
            continue
        a = {x["window"]: x for x in rows if x["variant"] == label and not x["error"]}
        b = {x["window"]: x for x in rows if x["variant"] == r and not x["error"]}
        common = sorted(set(a) & set(b))
        if len(common) < 2:
            continue
        rec = {"adaptive": label, "recency": r, "gap": gap, "n": len(common)}
        for key in ("nll_mean", "kl_mean", "top1"):
            d = np.array([a[i][key] - b[i][key] for i in common])
            lo, hi = boot_ci(d, np.mean, B, rng)
            rec[key] = (float(d.mean()), lo, hi, wilcoxon_p(d))
        rec["bits"] = (sm[label]["eff_bits"], sm[r]["eff_bits"])
        matched.append(rec)

    # ---- Pareto fronts and decision ---------------------------------------
    ok = [s for s in summary if s.get("n_windows") and s["group"] != "ref"]
    # Usability gate: a variant only counts if it is at least as accurate as uniform
    # INT4 (the cheapest standard baseline) on BOTH PPL and KL. Without this, a
    # broken configuration sits on the frontier just because nothing is cheaper.
    i4 = next((s for s in ok if s["variant"] == "int4_uniform"), None)
    usable = [s for s in ok if i4 is None or (s["ppl"] <= i4["ppl"] and s["kl"] <= i4["kl"])]
    ufront_kl = pareto_front({s["variant"]: (s["eff_bits"], s["kl"]) for s in usable})
    ufront_ppl = pareto_front({s["variant"]: (s["eff_bits"], s["ppl"]) for s in usable})
    three_on_front = sorted(s["variant"] for s in usable if s["group"] == "three_tier"
                            and s["variant"] in ufront_kl and s["variant"] in ufront_ppl)
    usable_names = sorted(s["variant"] for s in usable)
    afront_kl = pareto_front({s["variant"]: (s["eff_bits"], s["kl"]) for s in ok})
    afront_ppl = pareto_front({s["variant"]: (s["eff_bits"], s["ppl"]) for s in ok})

    # ---- report ------------------------------------------------------------
    report = os.path.join(rdir, f"DECISION_REPORT_{tag}.md")
    write_report(report, locals())
    print(f"\nwritten: {wpath}\n         {spath}\n         {report}")
    print(f"safety: {guard.summary()}")


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_report(path: str, L: dict) -> None:
    cfg, dt, summary, paired = L["cfg"], L["dt"], L["summary"], L["paired"]
    ufront_kl, ufront_ppl, three_on_front = L["ufront_kl"], L["ufront_ppl"], L["three_on_front"]
    usable_names = set(L["usable_names"])
    afront_kl, afront_ppl = L["afront_kl"], L["afront_ppl"]
    mc = L["model"].config
    chash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    out = []
    w = out.append
    w(f"# Decision test: {L['model_id']}\n")
    w("Generated by `decision_test.py`. Every number below was measured in this run; "
      "nothing is estimated or extrapolated. Do not edit by hand.\n")
    w("## Setup\n")
    w("| Item | Value |\n|---|---|")
    w(f"| Date | {datetime.datetime.now().isoformat(timespec='seconds')} |")
    w(f"| Model | {L['model_id']} (FP16 weights, {cfg['attn_impl']} attention) |")
    w(f"| KV shape (from model.config) | {mc.num_hidden_layers} layers x {mc.num_key_value_heads} KV heads x {L['head_dim']} head_dim |")
    w(f"| Weights offloaded to CPU | {L['offloaded']} |")
    w(f"| GPU | {torch.cuda.get_device_name(0)} |")
    w(f"| Software | Python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__} |")
    w(f"| Text | {cfg['wikitext_file']}, {len(L['windows'])} non-overlapping windows |")
    w(f"| Per window | {L['P']} tokens prefilled, {L['T']} tokens teacher-forced and scored |")
    w(f"| Group size / sink / recent / EMA alpha | {cfg['quant_group_size']} / {cfg.get('sink_tokens')} / "
      f"{cfg.get('recent_tokens')} / {cfg['ema_alpha']} |")
    w(f"| Key quantization (default) | {cfg.get('key_quant', 'per_channel')}, key_group_tokens "
      f"{cfg.get('key_group_tokens', 32)}; the newest {cfg.get('recent_tokens')} tokens stay FP16, and a "
      f"key waits at FP16 until its block is complete (both counted in memory; FP16 share of the tier mix) |")
    w(f"| Config hash | {chash} |")
    w(f"| CI | bootstrap 95% over windows, {L['B']} resamples, seed {cfg['seed']} |")
    w(f"| Hardware safety | {L['guard'].summary()} |\n")

    w("## Test cases\n")
    w("Each test case is one KV-cache variant run on every window. FP16 is the reference: "
      "KL and top-1 are computed against FP16's next-token distribution given the identical prefix.\n")
    w("| # | Variant | Method | Tiers | Fractions (INT8/INT4/INT2) | Key quantization |\n|---|---|---|---|---|---|")
    for i, (label, method, over, group) in enumerate(L["variants"], 1):
        tiers = {"ref": "FP16", "uniform": "single", "two_tier": "8/4", "three_tier": "8/4/2",
                 "old_keys": "8/4" if "p8" in over else "single"}[group]
        fr = (f"{over.get('p8')}/{over.get('p4')}/{over.get('p2', 0.0) if over.get('int2_enabled') else 0.0}"
              if "p8" in over else "-")
        kq = "-" if method == "fp16" else over.get("key_quant", cfg.get("key_quant", "per_channel"))
        w(f"| {i} | {label} | {method} | {tiers} | {fr} | {kq} |")
    w("")

    w("## Results\n")
    w("PPL and KL: lower is better. Top-1: higher is better. "
      "Bits = effective bits per value including scales, zero-points and index. "
      "Pareto column: whether the variant is on the bits-vs-quality frontier among USABLE variants "
      "(at least as accurate as uniform INT4 on PPL and KL); no other usable variant has both fewer bits "
      "and better quality. 'excluded' = less accurate than uniform INT4. 'All' column: the same "
      "frontier over every variant without the usability gate.\n")
    w("| Variant | Bits | Compr. | PPL [95% CI] | ΔPPL vs FP16 | KL [95% CI] | Top-1 [95% CI] | Tier mix 8/4/2/FP16 | ms/tok | Pareto gated (KL, PPL) | Pareto all (KL, PPL) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        if not s.get("n_windows"):
            w(f"| {s['variant']} | failed on all windows ({s['n_errors']} errors) |||||||||")
            continue
        mix = "-" if s["frac8"] is None else (f"{s['frac8']:.2f}/{s['frac4']:.2f}/{s['frac2']:.2f}/"
                                              f"{(s.get('frac16') or 0.0):.2f}")
        if s["group"] == "ref":
            par = "-"
        elif s["variant"] not in usable_names:
            par = "excluded"
        else:
            par = f"{'yes' if s['variant'] in ufront_kl else 'no'}, {'yes' if s['variant'] in ufront_ppl else 'no'}"
        apar = "-" if s["group"] == "ref" else (
            f"{'yes' if s['variant'] in afront_kl else 'no'}, {'yes' if s['variant'] in afront_ppl else 'no'}")
        w(f"| {s['variant']} | {s['eff_bits']:.2f} | {s['compression_x']:.2f}x | "
          f"{s['ppl']:.3f} [{s['ppl_ci_lo']:.3f}, {s['ppl_ci_hi']:.3f}] | {s['ppl_delta_pct']:+.1f}% | "
          f"{s['kl']:.4f} [{s['kl_ci_lo']:.4f}, {s['kl_ci_hi']:.4f}] | "
          f"{s['top1']:.3f} [{s['top1_ci_lo']:.3f}, {s['top1_ci_hi']:.3f}] | {mix} | "
          f"{s['decode_ms_per_tok']:.1f} | {par} | {apar} |")
    errs = [s for s in summary if s.get("n_errors")]
    if errs:
        w("\nErrors: " + "; ".join(f"{s['variant']}: {s['n_errors']} windows failed" for s in errs))
    w("")

    w("## Decision 1: two-tier vs three-tier (INT2)\n")
    w("Rule (fixed before the full run): (1) usability gate: only variants at least as accurate as "
      "uniform INT4 on BOTH PPL and KL are considered; (2) among those, the INT2 tier is worth keeping "
      "only if at least one three-tier variant is on BOTH Pareto frontiers (bits vs KL and bits vs PPL), "
      "i.e. no other usable variant is at least as cheap and at least as accurate.\n")
    w(f"Usable variants: {', '.join(L['usable_names']) or 'none'}.\n")
    if three_on_front:
        w(f"**Result: three-tier variants on both frontiers: {', '.join(three_on_front)}. "
          "INT2 earns a place as an option; report it.**\n")
    else:
        w("**Result: no three-tier variant is on both frontiers. Every INT2 configuration is beaten "
          "by a variant that uses no INT2 at equal or fewer bits. Use two-tier INT8/INT4; report INT2 "
          "as a negative ablation.**\n")

    w("## Question 2: adaptive (attention importance) vs recency, same tier fractions\n")
    w("Paired over identical windows. Difference = adaptive minus recency. "
      "Negative ΔNLL / ΔKL or positive Δtop-1 favour adaptive. "
      "Significant = 95% CI excludes 0 AND Wilcoxon p < 0.05. With per-channel keys the two methods "
      "can use different bits at the same fractions (adaptive scatters tiers across key blocks, so more "
      "per-channel scale sets are stored); compare the Bits column, and use the Pareto columns above "
      "for the equal-bits view.\n")
    w("| Fractions | n | Bits (adaptive / recency) | ΔNLL per token [CI], p | ΔKL [CI], p | Δtop-1 [CI], p | Verdict |")
    w("|---|---|---|---|---|---|---|")
    for p in paired:
        def cell(k, nd=4):
            m, lo, hi, pv = p[k]
            return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}], p={pv:.3g}"
        m, lo, hi, pv = p["kl_mean"]
        sig = (lo > 0 or hi < 0) and pv < 0.05
        verdict = ("adaptive better" if m < 0 else "recency better") if sig else "no significant difference"
        w(f"| {p['fractions']} | {p['n']} | {p['bits'][0]:.2f} / {p['bits'][1]:.2f} | {cell('nll_mean')} | "
          f"{cell('kl_mean')} | {cell('top1', 3)} | {verdict} (by KL) |")
    w("")
    w("## Question 2b: adaptive vs recency at matched memory (bit-matched pairs)\n")
    w(f"Each adaptive variant is paired with the recency variant whose effective bits are closest "
      f"(|Δbits| ≤ {dt.get('bit_match_tolerance', 0.15)}), on identical windows. This is the equal-memory "
      "comparison. Difference = adaptive minus recency; negative ΔKL/ΔNLL favour adaptive. "
      "Significant = 95% CI excludes 0 AND Wilcoxon p < 0.05.\n")
    if not L["matched"]:
        w("No pair within the tolerance.\n")
    else:
        w("| Adaptive | Recency | Bits (a / r) | ΔNLL per token [CI], p | ΔKL [CI], p | Δtop-1 [CI], p | Verdict |")
        w("|---|---|---|---|---|---|---|")
        for p in L["matched"]:
            def cell(k, nd=4):
                m, lo, hi, pv = p[k]
                return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}], p={pv:.3g}"
            m, lo, hi, pv = p["kl_mean"]
            sig = (lo > 0 or hi < 0) and pv < 0.05
            verdict = ("adaptive better" if m < 0 else "recency better") if sig else "no significant difference"
            w(f"| {p['adaptive']} | {p['recency']} | {p['bits'][0]:.2f} / {p['bits'][1]:.2f} | "
              f"{cell('nll_mean')} | {cell('kl_mean')} | {cell('top1', 3)} | {verdict} (by KL) |")
        w("")
    w("## Limitations of this test\n")
    w("- The cache is dequantized to FP16 for each attention call, so peak VRAM is not reduced; "
      "only the bytes stored between steps are.")
    w("- Quality is measured on WikiText-2 language modelling only; long-context retrieval "
      "(passkey) is not covered by this test.")
    w(f"- {len(L['windows'])} windows of {L['P'] + L['T']} tokens; conclusions hold for this context length.")
    if L["offloaded"]:
        w("- Weights were partly offloaded to CPU, so ms/tok mostly measures PCIe transfer, not the KV cache.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
