"""
adaptivekv/generate.py
----------------------
One decode engine shared by generation AND perplexity evaluation.

Methods: fp16 | int8_uniform | int4_uniform | int2_uniform | recency | adaptive

INT2 ablation: with int2_enabled: true, adaptive and recency use three tiers
(fractions p8 / p4 / p2). With it off (default) they are two-tier and p2 is ignored.

Adaptive policy
  * every token enters the cache at INT8; the newest token is never low precision
  * every `reassign_every` steps, tokens outside the protected zone
    (first `sink_tokens` + last `recent_tokens`) are re-ranked by EMA attention
    importance and may be DEMOTED (INT8 -> INT4); never promoted
  * prefill runs once (attentions + KV in the same forward pass)

Teacher forcing (forced_ids): instead of sampling, the given tokens are fed and
their log-probabilities recorded. This is how perplexity UNDER A QUANTIZED KV
CACHE is measured (see metrics.compute_perplexity_kv).
With ref_logprobs (the FP16 run's per-step log-probs, from collect_logprobs=True)
each step also records KL(FP16 || method) and whether the top-1 prediction
matches FP16, both given the SAME prefix.

Key quantization (config key_quant):
  per_channel (default)  ChannelKVCache: keys quantized per channel over blocks
                         of key_group_tokens tokens, values per token. New tokens
                         stay FP16 (tier code 16) while in the recent window; on
                         leaving it they get their tier (value quantized at once,
                         key quantized when its block is complete). Same for every method.
  per_token              AdaptiveKVCache (previous design; kept as an ablation).

Config keys used (config.yaml): seed, max_new_tokens, quant_group_size,
ema_alpha, p8, p4, p2, int2_enabled, sink_tokens, recent_tokens, reassign_every,
importance_window (default 16), key_quant, key_group_tokens (default 32),
k_channel_norm (per_token only, default true),
kv_storage_device (default null = same device as the model).
"""

import time
import math
import torch
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from transformers import DynamicCache

from .cache import AdaptiveKVCache, measure_kv_bytes, compute_effective_bits
from .cache_channel import (ChannelKVCache, FP16_TIER, measure_kv_bytes_channel,
                            compute_effective_bits_channel)
from .importance import AttentionHook, EMATracker
from .allocator import TierAllocator

METHODS = ("fp16", "int8_uniform", "int4_uniform", "int2_uniform", "recency", "adaptive")
_BITS = {"int8_uniform": 8, "int4_uniform": 4, "int2_uniform": 2}


@dataclass
class GenerationResult:
    method:           str
    prompt:           str
    generated_text:   str
    input_token_ids:  list
    output_token_ids: list
    kv_memory:        dict
    effective_bits:   float
    peak_vram_bytes:  int
    gen_latency_s:    float
    tokens_per_sec:   float
    tier_dist:        Dict[int, int]
    config_snapshot:  dict
    is_simulated:     bool = False
    tier_history:     List[dict] = field(default_factory=list)
    nll:              List[float] = field(default_factory=list)   # only when teacher-forced
    final_tiers:      List[int] = field(default_factory=list)     # tier of each cached position
    prefill_s:        float = 0.0                                 # prefill wall time (synced)
    decode_s:         float = 0.0                                 # decode wall time (synced)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _check_logits(logits: torch.Tensor) -> None:
    """Fail loudly instead of silently emitting garbage ('!!!' = argmax of NaN = token id 0)."""
    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Model produced NaN/Inf logits. This is an fp16 overflow in the model itself "
            "(not the KV cache). Load the model with dtype=torch.bfloat16.")


def _greedy(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak(device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def _layer_kv(pkv, li):
    """Version-tolerant access to (keys, values) of layer li."""
    if hasattr(pkv, "layers"):
        lay = pkv.layers[li]
        return lay.keys, lay.values
    if hasattr(pkv, "key_cache"):
        return pkv.key_cache[li], pkv.value_cache[li]
    return pkv[li][0], pkv[li][1]


def _build_dynamic_cache(kv_cache: AdaptiveKVCache, n_layers: int, device) -> DynamicCache:
    dyn = DynamicCache()
    for li in range(n_layers):
        K, V = kv_cache.get_kv_fp16(li, device)
        dyn.update(K, V, li)
    return dyn


def _append_last(kv_cache, pkv, n_layers, tier_arr):
    for li in range(n_layers):
        K, V = _layer_kv(pkv, li)
        kv_cache.append_tokens(li, K[:, :, -1:, :], V[:, :, -1:, :], tier_arr)


def _snapshot(step: int, tiers: np.ndarray) -> dict:
    return {"step": int(step), "n_tokens": int(len(tiers)),
            **{b: int((tiers == b).sum()) for b in (8, 4, 2, 16)}}


def _tiers_recency(n: int, p8: float, p4: float, sink: int = 0, p2: float = 0.0) -> np.ndarray:
    """Newest p8 fraction -> INT8, next p4 -> INT4, oldest p2 -> INT2
    (p2 = 0: every remaining token is INT4).
    The first `sink` tokens stay INT8 (same protection the adaptive method gets)."""
    n8 = math.ceil(p8 * n - 1e-9)
    n84 = min(n, math.ceil((p8 + p4) * n - 1e-9)) if p2 > 0 else n
    tiers = np.full(n, 2, dtype=np.int32)
    tiers[max(0, n - n84):] = 4
    tiers[max(0, n - n8):] = 8
    tiers[:sink] = 8
    return tiers


def _tier_fractions(cfg: dict):
    """(p8, p4, p2) from config. p2 is used only when int2_enabled is true."""
    p8, p4 = cfg["p8"], cfg["p4"]
    p2 = float(cfg.get("p2", 0.0)) if cfg.get("int2_enabled", False) else 0.0
    if abs(p8 + p4 + p2 - 1.0) > 1e-9:
        raise ValueError(f"p8 + p4 + p2 must sum to 1 (int2_enabled="
                         f"{cfg.get('int2_enabled', False)}), got {p8} + {p4} + {p2}")
    return p8, p4, p2


class _StepScorer:
    """Teacher-forced per-step scoring: NLL, and vs FP16 reference: KL + top-1 match."""

    def __init__(self, ref_logprobs: Optional[torch.Tensor], collect: bool):
        self.ref = ref_logprobs          # (T, V) float32 on CPU, or None
        self.collect = collect
        self.nll, self.kl, self.top1, self.logprobs = [], [], [], []

    def __call__(self, step: int, logits: torch.Tensor, tok_id: int) -> None:
        lp = torch.log_softmax(logits.float(), -1)
        self.nll.append(-lp[tok_id].item())
        if self.collect:
            self.logprobs.append(lp.cpu())
        if self.ref is not None:
            ref = self.ref[step].to(lp.device)
            self.kl.append(float((ref.exp() * (ref - lp)).sum().item()))
            self.top1.append(bool(int(lp.argmax()) == int(ref.argmax())))

    def result(self) -> dict:
        out = {"nll": self.nll, "kl": self.kl, "top1": self.top1}
        if self.collect:
            out["logprobs"] = torch.stack(self.logprobs) if self.logprobs else None
        return out


def _kv_values_per_token(model) -> int:
    mc = model.config
    head_dim = getattr(mc, "head_dim", None) or (mc.hidden_size // mc.num_attention_heads)
    return 2 * mc.num_hidden_layers * mc.num_key_value_heads * head_dim


# ---------------------------------------------------------------------------
# Engine: FP16 (native cache)
# ---------------------------------------------------------------------------

def _run_fp16(model, tokenizer, input_ids, cfg, max_new, forced, scorer):
    device = input_ids.device
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_attentions=False, logits_to_keep=1)
    pkv = out.past_key_values
    logits = out.logits[0, -1, :].clone()   # clone: do not keep the full (seq x vocab) tensor alive
    _check_logits(logits)
    del out
    _sync(device)
    prefill_s = time.perf_counter() - t0
    gen = []

    for step in range(max_new):
        if forced is not None:
            tok_id = int(forced[step])
            scorer(step, logits, tok_id)
        else:
            tok_id = _greedy(logits)
        gen.append(tok_id)
        if forced is None and tok_id == tokenizer.eos_token_id:
            break
        if step == max_new - 1:
            break
        tok = torch.tensor([[tok_id]], device=device)
        with torch.no_grad():
            out = model(tok, past_key_values=pkv, use_cache=True, output_attentions=False)
        pkv = out.past_key_values
        logits = out.logits[0, -1, :].clone()
        _check_logits(logits)
        del out

    kv_bytes = 0
    for li in range(model.config.num_hidden_layers):
        K, V = _layer_kv(pkv, li)
        kv_bytes += K.nbytes + V.nbytes
    n_tok = _layer_kv(pkv, 0)[0].shape[2]
    vpt = _kv_values_per_token(model)
    kv_mem = {
        "int8_data_bytes": 0, "int4_data_bytes": 0,
        "int8_scale_bytes": 0, "int4_scale_bytes": 0,
        "aux_bytes": 0, "index_bytes": 0,
        "total_bytes": kv_bytes,
        "n_original_values": n_tok * vpt,
        "fp16_equivalent": n_tok * vpt * 2,
        "fp16_tokens": n_tok,
        "is_simulated": False,
    }
    return {"gen": gen, "kv_mem": kv_mem, "prefill_s": prefill_s,
            "eff_bits": 8.0 * kv_bytes / (n_tok * vpt),
            "tier_dist": {8: 0, 4: 0, 2: 0}, "tier_history": []}


# ---------------------------------------------------------------------------
# Engine: quantized cache (uniform / recency / adaptive)
# ---------------------------------------------------------------------------

def _run_quant(model, tokenizer, input_ids, method, cfg, max_new, forced, scorer):
    device   = input_ids.device
    n_layers = len(model.model.layers)
    p8, p4, p2 = _tier_fractions(cfg)
    sink     = cfg.get("sink_tokens", 4)
    recent   = cfg.get("recent_tokens", 32)
    reassign = max(1, cfg.get("reassign_every", 8))
    adaptive = method == "adaptive"
    if method in _BITS:
        tier_set = (_BITS[method],)
    else:
        tier_set = (8, 4, 2) if p2 > 0 else (8, 4)

    kv_cache = AdaptiveKVCache(
        n_layers=n_layers, group_size=cfg["quant_group_size"],
        storage_device=cfg.get("kv_storage_device", None),
        k_channel_norm=cfg.get("k_channel_norm", True),
        tiers=tier_set)

    hook = ema = allocator = None
    if adaptive:
        hook = AttentionHook(model, layer_weights=cfg.get("layer_weights"),
                             window=cfg.get("importance_window", 16))
        ema = EMATracker(alpha=cfg["ema_alpha"])
        allocator = TierAllocator(p8=p8, p4=p4, p2=p2, sink_tokens=sink, recent_tokens=recent)

    gen, history = [], []
    try:
        # ---- prefill (single pass) ----
        t0 = time.perf_counter()
        if adaptive:
            hook.register()
        with torch.no_grad():
            # output_attentions stays False: eager attention still hands the weights
            # to the importance hook, while output_attentions=True makes transformers
            # 5.x keep every layer's (heads x seq x seq) matrix -> OOM at long context.
            out = model(input_ids, use_cache=True, output_attentions=False, logits_to_keep=1)
        if adaptive:
            ema.update(hook.get_scores())
            hook.remove(); hook.reset()

        P = input_ids.shape[1]
        if method == "recency":
            cur = _tiers_recency(P, p8, p4, sink, p2)
        elif adaptive:
            cur = np.full(P, 8, dtype=np.int32)
        else:
            cur = np.full(P, _BITS[method], dtype=np.int32)

        for li in range(n_layers):
            K, V = _layer_kv(out.past_key_values, li)
            kv_cache.append_tokens(li, K, V, cur)
        history.append(_snapshot(0, cur))
        logits = out.logits[0, -1, :].clone()
        _check_logits(logits)
        del out
        _sync(device)
        prefill_s = time.perf_counter() - t0

        # ---- decode ----
        for step in range(max_new):
            if forced is not None:
                tok_id = int(forced[step])
                scorer(step, logits, tok_id)
            else:
                tok_id = _greedy(logits)
            gen.append(tok_id)
            if forced is None and tok_id == tokenizer.eos_token_id:
                break
            if step == max_new - 1:
                break

            n_fwd = step + 1
            tok = torch.tensor([[tok_id]], device=device)
            dyn = _build_dynamic_cache(kv_cache, n_layers, device)

            if adaptive:
                hook.register()
            with torch.no_grad():
                out_step = model(tok, past_key_values=dyn, use_cache=True,
                                 output_attentions=False)
            if adaptive:
                ema.update(hook.get_scores())          # length = cache_len + 1
                hook.remove(); hook.reset()

            cache_len = len(cur)

            # periodic demotion-only reassignment
            if n_fwd % reassign == 0:
                if adaptive:
                    new = allocator.demote(cur, ema.get()[:cache_len])   # quota-capped, demote only
                elif method == "recency":
                    new = np.minimum(cur, _tiers_recency(cache_len, p8, p4, sink, p2))
                else:
                    new = cur
                if not np.array_equal(new, cur):
                    kv_cache.reassign_tiers(new)
                    cur = new
                history.append(_snapshot(n_fwd, cur))

            new_tier = np.array([8 if method in ("adaptive", "recency")
                                 else _BITS[method]], dtype=np.int32)
            _append_last(kv_cache, out_step.past_key_values, n_layers, new_tier)
            cur = np.concatenate([cur, new_tier])
            logits = out_step.logits[0, -1, :].clone()
            _check_logits(logits)
            del out_step, dyn
    finally:
        if hook is not None:
            hook.remove()          # never leak hooks, even on OOM

    return {"gen": gen, "kv_mem": measure_kv_bytes(kv_cache), "prefill_s": prefill_s,
            "eff_bits": compute_effective_bits(kv_cache),
            "tier_dist": kv_cache.tier_counts(), "tier_history": history,
            "final_tiers": cur.tolist()}


# ---------------------------------------------------------------------------
# Engine: per-channel key cache (default)
# ---------------------------------------------------------------------------

def _channel_policy(cur, imp, method, cfg, p, allocator, G, do_reassign):
    """Give a tier to tokens that left the recent window, then apply the method's tier rule.
    cur: (n,) tiers with 16 = FP16 residual. imp: (n,) EMA importance (adaptive only)."""
    p8, p4, p2 = p
    n = len(cur)
    recent = cfg.get("recent_tokens", 32)
    sink = cfg.get("sink_tokens", 4)
    new = cur.copy()
    c = max(0, n - recent)                            # positions < c have left the recent window
    commit = np.nonzero(new[:c] == FP16_TIER)[0]
    if len(commit):
        new[commit] = _BITS.get(method, 8)            # uniform: its bits; adaptive/recency: enter at INT8
    if method == "adaptive" and (do_reassign or len(commit)):
        new = allocator.demote(new, imp, extra_pinned=(new == FP16_TIER))
    elif method == "recency" and (do_reassign or len(commit)):
        rec = _tiers_recency(n, p8, p4, sink, p2)
        m = new != FP16_TIER
        new[m] = np.minimum(new[m], rec[m])
    return new


def _run_quant_channel(model, tokenizer, input_ids, method, cfg, max_new, forced, scorer):
    device   = input_ids.device
    n_layers = len(model.model.layers)
    p8, p4, p2 = _tier_fractions(cfg)
    G        = int(cfg.get("key_group_tokens", 32))
    reassign = max(1, cfg.get("reassign_every", 8))
    adaptive = method == "adaptive"
    tier_set = (_BITS[method],) if method in _BITS else ((8, 4, 2) if p2 > 0 else (8, 4))

    kv_cache = ChannelKVCache(n_layers=n_layers, group_size=cfg["quant_group_size"],
                              key_group_tokens=G, tiers=tier_set,
                              storage_device=cfg.get("kv_storage_device", None))
    hook = ema = allocator = None
    if adaptive:
        hook = AttentionHook(model, layer_weights=cfg.get("layer_weights"),
                             window=cfg.get("importance_window", 16))
        ema = EMATracker(alpha=cfg["ema_alpha"])
        allocator = TierAllocator(p8=p8, p4=p4, p2=p2, sink_tokens=cfg.get("sink_tokens", 4),
                                  recent_tokens=cfg.get("recent_tokens", 32))

    def imp_of(n):
        return ema.get()[:n] if adaptive else None

    gen, history = [], []
    try:
        t0 = time.perf_counter()
        if adaptive:
            hook.register()
        with torch.no_grad():
            out = model(input_ids, use_cache=True, output_attentions=False, logits_to_keep=1)
        if adaptive:
            ema.update(hook.get_scores())
            hook.remove(); hook.reset()
        for li in range(n_layers):
            K, V = _layer_kv(out.past_key_values, li)
            kv_cache.append_tokens(li, K, V)
        cur = np.full(input_ids.shape[1], FP16_TIER, dtype=np.int32)
        new = _channel_policy(cur, imp_of(len(cur)), method, cfg, (p8, p4, p2), allocator, G, True)
        if not np.array_equal(new, cur):
            kv_cache.set_tiers(new)
            cur = new
        history.append(_snapshot(0, cur))
        logits = out.logits[0, -1, :].clone()
        _check_logits(logits)
        del out
        _sync(device)
        prefill_s = time.perf_counter() - t0

        for step in range(max_new):
            if forced is not None:
                tok_id = int(forced[step])
                scorer(step, logits, tok_id)
            else:
                tok_id = _greedy(logits)
            gen.append(tok_id)
            if forced is None and tok_id == tokenizer.eos_token_id:
                break
            if step == max_new - 1:
                break

            n_fwd = step + 1
            tok = torch.tensor([[tok_id]], device=device)
            dyn = _build_dynamic_cache(kv_cache, n_layers, device)
            if adaptive:
                hook.register()
            with torch.no_grad():
                out_step = model(tok, past_key_values=dyn, use_cache=True, output_attentions=False)
            if adaptive:
                ema.update(hook.get_scores())          # length = cache_len + 1
                hook.remove(); hook.reset()

            for li in range(n_layers):                 # the new token enters the FP16 residual
                K, V = _layer_kv(out_step.past_key_values, li)
                kv_cache.append_tokens(li, K[:, :, -1:, :], V[:, :, -1:, :])
            cur = np.concatenate([cur, [FP16_TIER]]).astype(np.int32)
            do_re = n_fwd % reassign == 0
            new = _channel_policy(cur, imp_of(len(cur)), method, cfg, (p8, p4, p2), allocator, G, do_re)
            if not np.array_equal(new, cur):
                kv_cache.set_tiers(new)
                cur = new
            if do_re:
                history.append(_snapshot(n_fwd, cur))
            logits = out_step.logits[0, -1, :].clone()
            _check_logits(logits)
            del out_step, dyn
    finally:
        if hook is not None:
            hook.remove()

    return {"gen": gen, "kv_mem": measure_kv_bytes_channel(kv_cache), "prefill_s": prefill_s,
            "eff_bits": compute_effective_bits_channel(kv_cache),
            "tier_dist": kv_cache.tier_counts(), "tier_history": history,
            "final_tiers": cur.tolist()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_method(model, tokenizer, input_ids, method: str, cfg: dict,
               max_new: int, forced: Optional[List[int]] = None,
               ref_logprobs: Optional[torch.Tensor] = None,
               collect_logprobs: bool = False) -> dict:
    """Core entry: returns engine dict + timing/VRAM. Used by generate_* and metrics.

    Teacher-forced runs (forced given) also return per-step "nll"; with
    ref_logprobs also "kl" and "top1" vs that reference; with collect_logprobs
    the (T, V) log-prob tensor itself (use the FP16 run's as the reference).
    """
    assert method in METHODS, f"Unknown method: {method}"
    if forced is not None:
        max_new = len(forced)
    scorer = _StepScorer(ref_logprobs, collect_logprobs)
    torch.manual_seed(cfg["seed"])
    device = input_ids.device
    _reset_peak(device)
    _sync(device)
    t0 = time.perf_counter()
    if method == "fp16":
        r = _run_fp16(model, tokenizer, input_ids, cfg, max_new, forced, scorer)
    elif cfg.get("key_quant", "per_channel") == "per_channel":
        r = _run_quant_channel(model, tokenizer, input_ids, method, cfg, max_new, forced, scorer)
    elif cfg["key_quant"] == "per_token":
        r = _run_quant(model, tokenizer, input_ids, method, cfg, max_new, forced, scorer)
    else:
        raise ValueError(f"key_quant must be per_channel or per_token, got {cfg['key_quant']}")
    _sync(device)
    r["latency"] = time.perf_counter() - t0
    r["decode_s"] = r["latency"] - r["prefill_s"]
    r.update(scorer.result())
    r["peak_vram"] = _peak(device)
    return r


def _to_result(method, prompt, tokenizer, input_ids, r, cfg) -> GenerationResult:
    n = len(r["gen"])
    return GenerationResult(
        method=method, prompt=prompt,
        generated_text=tokenizer.decode(r["gen"], skip_special_tokens=True),
        input_token_ids=input_ids[0].tolist(), output_token_ids=r["gen"],
        kv_memory=r["kv_mem"], effective_bits=r["eff_bits"],
        peak_vram_bytes=r["peak_vram"], gen_latency_s=r["latency"],
        tokens_per_sec=n / r["latency"] if r["latency"] > 0 else 0.0,
        tier_dist=r["tier_dist"], config_snapshot=cfg, is_simulated=False,
        tier_history=r["tier_history"], nll=r["nll"],
        final_tiers=r.get("final_tiers", []),
        prefill_s=r["prefill_s"], decode_s=r["decode_s"])


def generate_adaptive(model, tokenizer, prompt, cfg, max_new_tokens=None) -> GenerationResult:
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    r = run_method(model, tokenizer, ids, "adaptive", cfg, max_new_tokens or cfg["max_new_tokens"])
    return _to_result("adaptive", prompt, tokenizer, ids, r, cfg)


def generate_baseline(model, tokenizer, prompt, method, cfg, max_new_tokens=None) -> GenerationResult:
    assert method in METHODS and method != "adaptive", f"Unknown baseline: {method}"
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    r = run_method(model, tokenizer, ids, method, cfg, max_new_tokens or cfg["max_new_tokens"])
    return _to_result(method, prompt, tokenizer, ids, r, cfg)