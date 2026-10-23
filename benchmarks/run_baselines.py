"""
benchmarks/run_baselines.py
----------------------------
Run all 5 methods × all context lengths × all windows.
Writes one JSON record per run to results/raw_results.jsonl.

Every number comes from a real model run. If a run exceeds VRAM budget
it is recorded as N/A (oom=True). Any other exception is recorded with
oom=False and the message in `error`, so a bug can never pose as an OOM.
No results or byte sizes are hardcoded.

Quality per method (perplexity / KL / top-1 through each cache) is measured by
decision_test.py; this script measures memory, effective bits and speed.

Usage:
  python benchmarks/run_baselines.py [--config config.yaml] [--dry-run]

Record schema (see design.md §6):
  method, model, context_length, seed, kv_bytes_real, kv_bytes_fp16,
  compression_ratio, effective_bits, perplexity, ppl_delta,
  token_agreement, tier_dist_int8, tier_dist_int4,
  peak_vram_bytes, gen_latency_s, tokens_per_sec,
  dataset_file, sample_count, config_hash, hardware, timestamp,
  is_simulated, oom, oom_reason, error, prefill_s, decode_ms_per_tok
"""

import argparse
import hashlib
import json
import os
import sys
import time
import datetime
import traceback

import torch
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from adaptivekv.generate import generate_adaptive, generate_baseline
from adaptivekv.metrics import compute_perplexity, compute_token_agreement
from adaptivekv.safety import Guard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def hardware_string() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def vram_free() -> int:
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(0)
        return free
    return 0


def write_record(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def make_windows(text: str, tokenizer, context_length: int, n_windows: int,
                 device: torch.device) -> list:
    """
    Tokenize full text, then extract n_windows non-overlapping windows
    of exactly context_length tokens. Returns list of token-ID lists.
    """
    all_ids = tokenizer.encode(text)
    total   = len(all_ids)
    stride  = max(1, (total - context_length) // max(n_windows - 1, 1))
    windows = []
    pos = 0
    while len(windows) < n_windows and pos + context_length <= total:
        windows.append(all_ids[pos : pos + context_length])
        pos += stride
    return windows


# ---------------------------------------------------------------------------
# Per-run benchmark
# ---------------------------------------------------------------------------

def benchmark_one(model, tokenizer, method, prompt_ids, cfg, seed,
                  device, vram_budget_bytes, dataset_file, chash):
    """
    Run one (method, prompt, seed) combination.
    Returns a record dict. Never raises — catches OOM and exceptions.
    """
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    context_length = len(prompt_ids)

    record = {
        "method":           method,
        "model":            cfg["model_id"],
        "context_length":   context_length,
        "seed":             seed,
        "dataset_file":     os.path.basename(dataset_file),
        "config_hash":      chash,
        "hardware":         hardware_string(),
        "timestamp":        datetime.datetime.utcnow().isoformat(),
        "oom":              False,
        "oom_reason":       None,
        "error":            None,
        "is_simulated":     False,
        # Filled below:
        "kv_bytes_real":    None,
        "kv_bytes_fp16":    None,
        "compression_ratio": None,
        "effective_bits":   None,
        "perplexity":       None,
        "ppl_delta":        None,
        "token_agreement":  None,
        "tier_dist_int8":   None,
        "tier_dist_int4":   None,
        "peak_vram_bytes":  None,
        "gen_latency_s":    None,
        "tokens_per_sec":   None,
    }

    # ── VRAM check before run ─────────────────────────────────────────────
    free_before = vram_free()
    if torch.cuda.is_available() and free_before < 200 * 1024**2:
        record["oom"] = True
        record["oom_reason"] = f"N/A - insufficient VRAM before run: {free_before/1024**2:.0f} MB free"
        return record

    try:
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(device)

        max_new = cfg["max_new_tokens"]
        run_cfg = dict(cfg)
        run_cfg["seed"] = seed

        if method == "adaptive":
            result = generate_adaptive(model, tokenizer, prompt, run_cfg,
                                       max_new_tokens=max_new)
        else:
            result = generate_baseline(model, tokenizer, prompt, method, run_cfg,
                                       max_new_tokens=max_new)

        peak = result.peak_vram_bytes
        if torch.cuda.is_available() and peak > vram_budget_bytes:
            record["oom"] = True
            record["oom_reason"] = (
                f"N/A - exceeded VRAM budget: peak {peak/1024**3:.3f} GB "
                f"> budget {vram_budget_bytes/1024**3:.2f} GB"
            )
            record["peak_vram_bytes"] = peak
            return record

        # Fill record
        kv_real = result.kv_memory["total_bytes"]
        kv_fp16 = result.kv_memory["fp16_equivalent"]
        comp    = kv_fp16 / kv_real if kv_real > 0 else None
        n_total = result.kv_memory.get("int8_data_bytes", 0) or 1
        tier_total = sum(result.tier_dist.values())

        record.update({
            "kv_bytes_real":    kv_real,
            "kv_bytes_fp16":    kv_fp16,
            "compression_ratio": comp,
            "effective_bits":   result.effective_bits,
            "tier_dist_int8":   result.tier_dist.get(8, 0) / tier_total if tier_total else None,
            "tier_dist_int4":   result.tier_dist.get(4, 0) / tier_total if tier_total else None,
            "peak_vram_bytes":  peak,
            "gen_latency_s":    result.gen_latency_s,
            "tokens_per_sec":   result.tokens_per_sec,
            "prefill_s":        result.prefill_s,
            "decode_ms_per_tok": (1000 * result.decode_s / max(1, len(result.output_token_ids) - 1)),
            "is_simulated":     result.is_simulated,
            "generated_tokens": result.output_token_ids,
        })

    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated(device)
        record["oom"] = True
        record["oom_reason"] = f"N/A - CUDA OOM: {str(e)[:120]}"
        record["peak_vram_bytes"] = peak
        torch.cuda.empty_cache()

    except Exception as e:
        record["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        traceback.print_exc()

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmarks(cfg_path: str = "config.yaml", dry_run: bool = False):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    chash = config_hash(cfg)
    results_dir = cfg.get("results_dir", "results")
    raw_path = os.path.join(results_dir, "raw_results.jsonl")
    os.makedirs(results_dir, exist_ok=True)

    # Load model once
    print("=" * 70)
    print("BENCHMARK RUN START")
    print(f"  Config:  {cfg_path}  (hash: {chash})")
    print(f"  Model:   {cfg['model_id']}")
    print(f"  Results: {raw_path}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        free, total_v = torch.cuda.mem_get_info(0)
        print(f"  GPU: {hardware_string()}, {total_v/1024**3:.2f} GB total, {free/1024**3:.2f} GB free")
    else:
        print("  Running on CPU (no CUDA)")

    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        attn_implementation=cfg["attn_impl"],
    )
    model.eval()
    if torch.cuda.is_available():
        free_after, _ = torch.cuda.mem_get_info(0)
        print(f"  Model loaded. VRAM used: {(free - free_after)/1024**3:.3f} GB, "
              f"free: {free_after/1024**3:.3f} GB")

    vram_budget = int(cfg["vram_budget_gb"] * 1024**3)
    dataset_file = cfg["wikitext_file"]
    text = load_text(dataset_file)
    n_samples = cfg["benchmark_samples"]
    n_repeats = cfg["benchmark_repeats"]

    methods = ["fp16", "int8_uniform", "int4_uniform", "adaptive", "recency"]
    context_lengths = cfg["context_lengths"]

    # ── Check 4096 VRAM budget ─────────────────────────────────────────────
    if 4096 not in context_lengths:
        context_lengths_run = context_lengths.copy()
    else:
        context_lengths_run = [c for c in context_lengths if c != 4096]
        # Will add 4096 only if measured VRAM permits
        context_lengths_run.append(4096)

    total_runs = len(methods) * len(context_lengths_run) * n_samples * n_repeats
    print(f"\nPlan: {len(methods)} methods × {len(context_lengths_run)} ctx_lens "
          f"× {n_samples} samples × {n_repeats} repeats = {total_runs} runs")
    if dry_run:
        print("[DRY RUN] Exiting without running.")
        return

    # warm-up so the first timed run does not include CUDA/kernel start-up
    warm = tokenizer.decode(tokenizer.encode(text)[:64])
    for m in ("fp16", "int4_uniform"):
        generate_baseline(model, tokenizer, warm, m, cfg, max_new_tokens=4)

    guard = Guard(cfg)
    run_count = 0
    # FP16 results indexed by (context_length, seed, sample_idx) for token agreement
    fp16_outputs: dict = {}

    for ctx_len in context_lengths_run:
        print(f"\n{'─'*60}")
        print(f"Context length: {ctx_len}")

        # No pre-run VRAM estimate (it needed hardcoded bytes/token); a context
        # that does not fit shows up as a measured CUDA OOM record instead.
        windows = make_windows(text, tokenizer, ctx_len, n_samples, device)
        actual_samples = len(windows)
        if actual_samples < n_samples:
            print(f"  WARNING: only {actual_samples} windows available (requested {n_samples})")

        for method in methods:
            print(f"  Method: {method} ({actual_samples} samples × {n_repeats} repeats)")
            for s_idx, window in enumerate(windows):
                for rep in range(n_repeats):
                    seed = cfg["seed"] + s_idx * 100 + rep
                    run_count += 1
                    print(f"    [{run_count}/{total_runs}] ctx={ctx_len} method={method} "
                          f"sample={s_idx} rep={rep} seed={seed}", end=" ... ", flush=True)

                    guard()
                    record = benchmark_one(
                        model, tokenizer, method, window, cfg, seed,
                        device, vram_budget, dataset_file, chash
                    )

                    # Store FP16 token output for token agreement calculation
                    key = (ctx_len, seed, s_idx)
                    if method == "fp16" and not record["oom"]:
                        fp16_outputs[key] = record.get("generated_tokens", [])

                    # Token agreement vs FP16
                    if method != "fp16" and not record["oom"]:
                        fp16_toks = fp16_outputs.get(key, [])
                        if fp16_toks and record.get("generated_tokens"):
                            record["token_agreement"] = compute_token_agreement(
                                record["generated_tokens"], fp16_toks
                            )

                    # Remove large list from record before writing
                    record.pop("generated_tokens", None)
                    record["sample_count"] = actual_samples

                    write_record(raw_path, record)

                    status = "OOM" if record["oom"] else f"ERROR {record['error'][:60]}" if record.get("error") else (
                        f"eff={record['effective_bits']:.1f}b "
                        f"mem={record['kv_bytes_real']/1024:.0f}KB "
                        f"peak={record['peak_vram_bytes']/1024**3:.2f}GB"
                        if record['kv_bytes_real'] else "ok"
                    )
                    print(status)

                    torch.cuda.empty_cache()

    # ── Compute perplexity in a separate pass (not per-generation) ─────────
    print(f"\n{'─'*60}")
    print("Computing perplexity (sliding-window, all methods share same eval)...")
    _compute_and_append_ppl(model, tokenizer, text, cfg, raw_path, chash, device)

    print(f"\n{'='*60}")
    print(f"Benchmark complete. {run_count} runs written to {raw_path}")
    print(f"safety: {guard.summary()}")


def _compute_and_append_ppl(model, tokenizer, text, cfg, raw_path, chash, device):
    """
    FP16 reference perplexity only (unquantized forward pass, first 50K chars).
    It is NOT a per-method quality number: the KV cache is never quantized
    here. Per-method perplexity / KL / top-1 through each cache come from
    decision_test.py.
    """
    max_length = min(512, cfg["context_lengths"][0])
    stride = max_length // 2

    print(f"  PPL eval: max_length={max_length}, stride={stride}")
    try:
        ppl = compute_perplexity(model, tokenizer, text[:50000],
                                 stride=stride, max_length=max_length,
                                 device=device)
        print(f"  FP16 perplexity (first 50K chars, ctx={max_length}): {ppl:.3f}")

        # Write one PPL record
        import datetime
        record = {
            "method": "fp16",
            "model": cfg["model_id"],
            "metric": "perplexity",
            "perplexity": ppl,
            "ppl_context_length": max_length,
            "ppl_stride": stride,
            "dataset_file": os.path.basename(cfg["wikitext_file"]),
            "config_hash": chash,
            "hardware": hardware_string(),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "note": "FP16 model, no KV quantization during scoring",
        }
        ppl_path = os.path.join(cfg.get("results_dir", "results"), "ppl_results.jsonl")
        write_record(ppl_path, record)
    except Exception as e:
        print(f"  PPL computation failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_benchmarks(args.config, args.dry_run)
