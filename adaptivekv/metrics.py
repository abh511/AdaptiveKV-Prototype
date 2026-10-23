"""
adaptivekv/metrics.py
---------------------
Metric functions used by benchmarks. All values come from real model outputs.

IMPORTANT FIX
  The old compute_perplexity() ran one normal forward pass, so the KV cache was
  never quantized and every method got the SAME perplexity. It is kept only as
  the full-forward FP16 reference (with corrected token weighting).
  compute_perplexity_kv() is the one to use for comparing methods: it feeds the
  text token by token (teacher forcing) THROUGH each method's KV cache, so
  quantization error actually affects the result.
"""

import math
import torch
import numpy as np
from typing import List, Optional


def compute_perplexity(model, tokenizer, text: str, stride: int,
                       max_length: int, device: torch.device) -> float:
    """
    Sliding-window perplexity with an UNQUANTIZED forward pass (FP16 reference only).
    Token weighting follows the number of tokens that actually carry loss
    (HF shifts labels internally, so a window scores n_valid - 1 tokens).
    """
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    n = ids.shape[1]
    if n < 2:
        return float("nan")

    nll_sum, n_scored, prev_end = 0.0, 0, 0
    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        target_len = end - prev_end
        chunk = ids[:, begin:end]
        labels = chunk.clone()
        labels[:, : (end - begin) - target_len] = -100
        n_loss = int((labels != -100).sum().item()) - 1       # label shift
        if n_loss > 0:
            with torch.no_grad():
                loss = model(chunk, labels=labels).loss.item()
            nll_sum += loss * n_loss
            n_scored += n_loss
        prev_end = end
        if end == n:
            break
    return math.exp(nll_sum / n_scored) if n_scored else float("nan")


def compute_perplexity_kv(model, tokenizer, text: str, method: str, cfg: dict,
                          prefill_len: int = 128, eval_len: int = 128,
                          n_windows: int = 20, stride: Optional[int] = None) -> dict:
    """
    Perplexity under a given KV-cache method (fp16, int8_uniform, int4_uniform,
    recency, adaptive).

    For each window of prefill_len + eval_len tokens: prefill the first
    prefill_len tokens, then teacher-force the next eval_len tokens one at a
    time through the method's cache and sum their NLL.

    Returns dict with ppl, nll_sum, n_scored, n_windows_used, window params,
    and kv stats of the LAST window (bytes, effective bits, tier_dist, peak VRAM).
    Use identical text/params for every method so results are comparable.
    """
    from .generate import run_method          # local import: avoid cycles

    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
    win = prefill_len + eval_len
    stride = stride or win
    starts = list(range(0, max(0, len(ids) - win) + 1, stride))[:n_windows]
    if not starts or len(ids) < win:
        return {"ppl": float("nan"), "reason": f"text has {len(ids)} tokens < window {win}"}

    nll_sum, n_scored, last = 0.0, 0, None
    for s in starts:
        prefix = torch.tensor([ids[s:s + prefill_len]], device=device)
        forced = ids[s + prefill_len:s + win]
        r = run_method(model, tokenizer, prefix, method, cfg, len(forced), forced=forced)
        nll_sum += float(sum(r["nll"]))
        n_scored += len(r["nll"])
        last = r

    return {
        "method": method,
        "ppl": math.exp(nll_sum / n_scored),
        "nll_sum": nll_sum, "n_scored": n_scored,
        "n_windows_used": len(starts),
        "prefill_len": prefill_len, "eval_len": eval_len, "stride": stride,
        "kv_total_bytes_last_window": last["kv_mem"]["total_bytes"],
        "kv_fp16_equiv_last_window": last["kv_mem"]["fp16_equivalent"],
        "effective_bits_last_window": last["eff_bits"],
        "tier_dist_last_window": last["tier_dist"],
        "peak_vram_bytes_last_window": last["peak_vram"],
    }


def compute_token_agreement(tokens_method: List[int], tokens_fp16: List[int]) -> float:
    """Fraction of positions (up to the shorter length) identical to FP16 greedy output."""
    n = min(len(tokens_method), len(tokens_fp16))
    if n == 0:
        return float("nan")
    return sum(a == b for a, b in zip(tokens_method[:n], tokens_fp16[:n])) / n


def first_divergence(tokens_method: List[int], tokens_fp16: List[int]) -> Optional[int]:
    """Index of first differing token, or None if identical over the shared length."""
    for i, (a, b) in enumerate(zip(tokens_method, tokens_fp16)):
        if a != b:
            return i
    return None


def mean_std(values: List[float]):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def measure_peak_vram(device: torch.device) -> int:
    """Peak VRAM bytes since last torch.cuda.reset_peak_memory_stats()."""
    return torch.cuda.max_memory_allocated(device)