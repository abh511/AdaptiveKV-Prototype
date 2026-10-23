"""
dashboard/app.py  -  AdaptiveKV live comparison (single page)

One click runs FP16 and Adaptive on the same prompt and shows, side by side:
output text, KV memory, GPU peak memory, latency, tokens, speed, and the
percentage change. Every number is measured live; nothing is typed in.

Run:  streamlit run dashboard/app.py      (stop and restart after code changes)
"""

import math
import os
import sys

import pandas as pd
import streamlit as st
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer                        # noqa: E402
from adaptivekv.generate import generate_adaptive, generate_baseline, load_config   # noqa: E402
from adaptivekv.metrics import compute_token_agreement, first_divergence             # noqa: E402

st.set_page_config(page_title="AdaptiveKV", page_icon="🧠", layout="centered")

@st.cache_resource(show_spinner="Loading model and warming up…")
def load_model():
    cfg = load_config(os.path.join(ROOT, "config.yaml"))
    dtype_name = cfg.get("dtype", "bfloat16")
    if dtype_name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        dtype_name = "float16"
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=getattr(torch, dtype_name),
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation=cfg.get("attn_impl", "eager"), trust_remote_code=True).eval()
    # warm-up so the first timed run is not penalised by CUDA start-up
    warm = {**cfg, "max_new_tokens": 4}
    generate_baseline(model, tok, "Hello", "fp16", warm, 4)
    generate_adaptive(model, tok, "Hello", warm, 4)
    return cfg, tok, model, dtype_name


def fmt_bytes(b: float) -> str:
    return f"{b / 1024 ** 2:.2f} MB" if b >= 1024 ** 2 else f"{b / 1024:.1f} KB"


def pct(new: float, old: float) -> str:
    if not old or math.isnan(old) or math.isnan(new):
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


st.title("🧠 AdaptiveKV")
st.caption("FP16 vs importance-based mixed-precision KV cache · live comparison")

cfg, tokenizer, model, dtype_name = load_model()

prompt = st.text_area("Prompt", value="What is the capital of France?", height=80)
c1, c2 = st.columns(2)
max_new = c1.slider("Max new tokens", 8, 128, 48)
use_chat = c2.checkbox("Use chat template (for Instruct models)", value=True)

if st.button("▶ Run comparison", type="primary"):
    if not prompt.strip():
        st.warning("Enter a prompt first.")
        st.stop()

    text_in = prompt
    if use_chat and tokenizer.chat_template:
        text_in = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    run_cfg = {**cfg, "max_new_tokens": max_new}

    try:
        with st.spinner("Running FP16 then Adaptive…"):
            ref = generate_baseline(model, tokenizer, text_in, "fp16", run_cfg)
            res = generate_adaptive(model, tokenizer, text_in, run_cfg)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # ── outputs ──────────────────────────────────────────────────────────
    st.subheader("Output")
    o1, o2 = st.columns(2)
    o1.markdown("**FP16**")
    o1.info(ref.generated_text or "(empty)")
    o2.markdown("**Adaptive**")
    o2.success(res.generated_text or "(empty)")

    agree = compute_token_agreement(res.output_token_ids, ref.output_token_ids)
    div = first_divergence(res.output_token_ids, ref.output_token_ids)
    st.caption(f"Token agreement with FP16: **{agree * 100:.1f}%**"
               + ("" if div is None else f" · first difference at generated token {div}"))

    # ── comparison table ─────────────────────────────────────────────────
    mf, ma = ref.kv_memory, res.kv_memory
    n_f, n_a = mf["fp16_tokens"], sum(res.tier_dist.values())
    bpt_f, bpt_a = mf["total_bytes"] / n_f, ma["total_bytes"] / n_a
    ratio = bpt_f / bpt_a if bpt_a else float("nan")
    gen_f, gen_a = len(ref.output_token_ids), len(res.output_token_ids)

    rows = [
        ("KV memory stored", f"{fmt_bytes(mf['total_bytes'])}  ({n_f} tokens)",
         f"{fmt_bytes(ma['total_bytes'])}  ({n_a} tokens)", ""),
        ("KV memory per token", f"{fmt_bytes(bpt_f)}", f"{fmt_bytes(bpt_a)}",
         f"{pct(bpt_a, bpt_f)}  ({ratio:.2f}× smaller)"),
        ("Effective bits per value", f"{ref.effective_bits:.2f}", f"{res.effective_bits:.2f}",
         pct(res.effective_bits, ref.effective_bits)),
        ("GPU peak memory (incl. weights)", fmt_bytes(ref.peak_vram_bytes), fmt_bytes(res.peak_vram_bytes),
         pct(res.peak_vram_bytes, ref.peak_vram_bytes)),
        ("Tokens generated", str(gen_f), str(gen_a), ""),
        ("Latency", f"{ref.gen_latency_s:.2f} s", f"{res.gen_latency_s:.2f} s",
         pct(res.gen_latency_s, ref.gen_latency_s)),
        ("Speed", f"{ref.tokens_per_sec:.1f} tok/s", f"{res.tokens_per_sec:.1f} tok/s",
         pct(res.tokens_per_sec, ref.tokens_per_sec)),
    ]
    st.subheader("Comparison")
    m1, m2, m3 = st.columns(3)
    m1.metric("KV memory per token", fmt_bytes(bpt_a), pct(bpt_a, bpt_f), delta_color="inverse")
    m2.metric("Effective bits", f"{res.effective_bits:.2f}",
              pct(res.effective_bits, ref.effective_bits), delta_color="inverse")
    m3.metric("Speed", f"{res.tokens_per_sec:.1f} tok/s", pct(res.tokens_per_sec, ref.tokens_per_sec))
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "FP16", "Adaptive", "Change"]),
                 hide_index=True, width="stretch")
    st.caption("Negative memory change = reduction. Memory change uses bytes per token, so it "
               "stays fair when the two runs produce different numbers of tokens. Compare speed "
               "(tok/s), not latency, when the token counts differ.")

    with st.expander("Settings used / notes"):
        keys = ("model_id", "p8", "p4", "p2", "int2_enabled", "ema_alpha", "quant_group_size",
                "key_quant", "key_group_tokens", "sink_tokens", "recent_tokens", "reassign_every",
                "importance_window")
        st.json({k: cfg.get(k) for k in keys} | {"dtype": dtype_name})
        st.caption("KV memory = packed bytes stored between steps. GPU peak memory is not reduced "
                   "(and can increase): each step rebuilds the full-precision cache for attention.")