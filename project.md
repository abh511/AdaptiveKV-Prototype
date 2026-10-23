# AdaptiveKV — Project Brief (for PPT and Report)

**Title:** AdaptiveKV: Software-Based Memory-Wall Mitigation in LLM Inference
**Subtitle:** Attention-guided mixed-precision (INT8 / INT4 / INT2) KV-cache quantization on a 4 GB consumer GPU

> Every number in this file is copied from a generated report in `results/` (source named next to each table).
> Those reports were produced on 2026-10-13 by running the code on this machine. If you re-run any
> experiment, re-copy the numbers from the regenerated report — do not edit them by hand.

---

## 1. One-paragraph summary (use for the abstract / opening slide)

When a large language model (LLM) generates text, it keeps the Key and Value vectors of every previous
token in memory (the **KV cache**) so it does not recompute them. This cache grows with every token and
becomes a major memory and bandwidth bottleneck — the **memory wall**. AdaptiveKV reduces the cache's
memory by storing each token at a different precision depending on how much attention the model pays to it:
important tokens stay at INT8, less important ones drop to INT4 or INT2. It is pure software (PyTorch on
top of Hugging Face Qwen2.5), runs on a 4 GB RTX 3050 laptop GPU, and every result is measured against an
FP16 baseline with confidence intervals and significance tests.

---

## 2. Problem statement

- An LLM's KV cache stores, for every token, every layer: `2 (K and V) × KV-heads × head_dim × bytes`.
- For Qwen2.5-0.5B-Instruct (24 layers, 2 KV heads, head_dim 64, FP16) that is **12,288 bytes per token**
  (verified from real tensor sizes; see README "Environment").
- The cache grows **linearly with context length** and is read in full at every generation step, so on
  small GPUs it limits context length and batch size, and dominates memory traffic.
- Measured stored KV for FP16 on 0.5B: 3.36 MiB at 256 tokens → 24.36 MiB at 2048 tokens
  (`results/PAPER_REPORT.md` §3.6).

**Question the project answers:** can we store most of the cache in fewer bits, choosing *which* tokens get
fewer bits from the model's own attention, without measurably hurting output quality?

---

## 3. Objectives

1. Build a mixed-precision KV cache with packed INT8 / INT4 / INT2 storage.
2. Decide each token's precision from an **attention-based importance score** smoothed by an EMA.
3. Measure memory **from real stored tensors** (`.nbytes`), including all scale/zero-point overhead.
4. Compare honestly against FP16, uniform INT8/INT4/INT2, and a **recency** heuristic.
5. Test quality three ways: perplexity/KL vs FP16, long-context passkey retrieval, and whether attention
   importance actually predicts future attention.
6. Provide a live dashboard demo (FP16 vs Adaptive on any prompt).

---

## 4. Core idea

```
Important tokens  → INT8  (high precision)
Medium tokens     → INT4
Unimportant tokens→ INT2  (low precision)
"Important" = receives a lot of attention from the model, averaged over time (EMA)
```

**Movement is one-way.** Tokens are re-ranked every step, but a token can only move *down* a tier
(INT8 → INT4 → INT2), never back up: once it is stored at lower precision its exact values are gone.
Do not show arrows in both directions on the slides.

**Default configuration:** two tiers, INT8/INT4 at 0.50/0.50 (7.41 effective bits, 2.16× smaller,
+0.1% perplexity). INT2 is an experiment that did not pay off (see §10).

Plus two protections widely used in KV-cache research:
- **Attention sinks:** the first 4 tokens always get disproportionate attention → pinned at INT8.
- **Recent window:** the newest 8 tokens stay at FP16 until they age out.

---

## 5. Workflow (draw this as the main diagram slide)

```
 ┌────────────┐     ┌──────────────┐     ┌─────────────────────────┐
 │ User prompt│ ──▶ │ Qwen2.5 LLM  │ ──▶ │ Attention hook (every   │
 └────────────┘     │ (eager attn) │     │ layer): reduce attention│
                    └──────────────┘     │ to 1 score per token,   │
                          ▲              │ discard the matrix      │
                          │              └───────────┬─────────────┘
                          │                          ▼
                          │              ┌─────────────────────────┐
                          │              │ EMA importance tracker  │
                          │              │ imp = α·score+(1-α)·imp │
                          │              │ α = 0.1                 │
                          │              └───────────┬─────────────┘
                          │                          ▼
                          │              ┌─────────────────────────┐
                          │              │ Allocator: rank tokens, │
                          │              │ assign tiers under fixed│
                          │              │ quotas p8/p4/p2;        │
                          │              │ demotion only           │
                          │              └───────────┬─────────────┘
                          │                          ▼
                          │              ┌─────────────────────────┐
                          │              │ Quantize + bit-pack     │
                          │              │ Keys: per channel       │
                          │              │ Values: per token       │
                          │              │ INT4 = 2/byte, INT2=4/byte│
                          │              └───────────┬─────────────┘
                          │                          ▼
                          │              ┌─────────────────────────┐
                          └───────────── │ Dequantize for attention│
                       next token        │ → generate next token   │
                       (loop)            └─────────────────────────┘
```

### Step-by-step (one generation step)

1. **Forward pass.** The model processes the newest token using the cached K/V of all previous tokens.
2. **Importance capture.** A hook on each attention module takes the attention the newest query gives every
   cached token, averages over heads and layers → one score per token. The full attention matrix is
   discarded immediately (storing it would run a 4 GB GPU out of memory).
3. **EMA smoothing.** `importance_t = 0.1 × score_t + 0.9 × importance_(t-1)`; a new token starts at its raw score.
4. **Tier assignment.** Tokens leaving the FP16 recent window are ranked by importance. The top `p8` fraction
   stays INT8, the next `p4` INT4, the rest INT2. Tiers are re-checked every step but can only be **demoted**
   (never promoted, because the original values are gone), and demotions are capped by the quotas.
5. **Storage.** Values are quantized per token (groups of 64). Keys are quantized **per channel** (KIVI-style):
   keys of the same tier inside a 32-token block share one scale and zero-point per channel. A key waits at
   FP16 until its block is complete. INT4/INT2 codes are bit-packed.
6. **Attention read.** The cache is dequantized back to FP16 for the attention call, and the next token is produced.
7. **Accounting.** Memory = sum of `.nbytes` of every stored tensor (codes, scales, zero-points, FP16 window,
   pending keys, 1-byte tier index). Effective bits = bytes × 8 / number of FP16 values represented.

---

## 6. System architecture (modules)

| Module | Role |
|---|---|
| `adaptivekv/quant.py` | INT8/INT4/INT2 asymmetric quantize/dequantize + bit packing |
| `adaptivekv/importance.py` | Memory-safe attention hook + EMA tracker |
| `adaptivekv/allocator.py` | Quota-capped, demotion-only tier assignment (2 or 3 tiers) |
| `adaptivekv/cache_channel.py` | `ChannelKVCache` — per-channel keys, per-token values (default) |
| `adaptivekv/cache.py` | `AdaptiveKVCache` — older per-token scheme (kept as ablation) |
| `adaptivekv/generate.py` | One decode engine for fp16 / uniform / recency / adaptive |
| `adaptivekv/metrics.py` | Perplexity through the quantized cache, token agreement |
| `adaptivekv/safety.py` | GPU temperature (pause at 80 °C) and RAM guard |
| `decision_test.py` | Quality vs memory: PPL / KL / top-1, paired statistics |
| `passkey.py` | Needle-in-a-haystack retrieval through each cache |
| `importance_analysis.py` | Does attention importance predict future attention? |
| `benchmarks/` | Memory/speed vs context length, extrapolation, figures, reports |
| `dashboard/app.py` | Streamlit live demo: FP16 vs Adaptive side by side |
| `tests/` | pytest suite (71 tests) |
| `config.yaml` | Every tunable parameter (nothing hard-coded) |

---

## 7. Tools, hardware and setup

| Item | Value |
|---|---|
| Language / framework | Python 3.11.9, PyTorch 2.11.0+cu128, transformers 5.18.0 |
| UI | Streamlit |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM (CUDA 12.8) |
| RAM | 16 GB |
| Models | Qwen2.5-0.5B-Instruct (fully on GPU), Qwen2.5-3B-Instruct (weights partly offloaded to CPU) |
| Dataset | WikiText-2 test set (`data/wikitext2_test.txt`) |
| Key parameters | EMA α 0.1, group size 64, sink 4, recent 8, key block 32 tokens, default tiers INT8/INT4 at 0.50/0.50 |

---

## 8. Baselines and experiment design

**Methods compared (all measured with the same code and same accounting):**
- **FP16** — the normal, unquantized cache (reference).
- **Uniform INT8 / INT4 / INT2** — every token at the same precision.
- **Recency** — newest tokens get the highest precision, same fractions and same sink pins as adaptive.
- **Adaptive** — attention-guided (this project).
- **Per-token keys** variants — the earlier design, kept as an ablation.

**Metrics:**
| Metric | Meaning | Better |
|---|---|---|
| Perplexity (PPL) | How well the model predicts real text, measured *through* the quantized cache | lower |
| KL divergence vs FP16 | How far the next-token distribution moves from FP16 (nats/token) | lower |
| Top-1 agreement | Share of steps where the most likely next token equals FP16's | higher |
| Effective bits | Real stored bits per value, including all overheads | lower |
| Compression | FP16 bytes / stored bytes | higher |
| Passkey accuracy | Can the model still recall a 5-digit key hidden in long text | higher |

**Statistics:** bootstrap 95% CI over windows (2000 resamples, seed 42); paired Wilcoxon signed-rank test
for adaptive vs recency; exact McNemar test for passkey; Wilson CI for accuracies.

**Quality run (0.5B):** 20 non-overlapping WikiText-2 windows, each 512 tokens prefilled + 64 tokens scored.
**Quality run (3B):** 3 windows (time budget, CPU offload).
**Passkey:** 60 prompts — contexts 512/1024/2048 × depths 0.1–0.9 × 4 trials.

---

## 9. Results

### 9.1 Main quality/memory table — Qwen2.5-0.5B-Instruct (source: `results/PAPER_REPORT.md` §3.2a)

| Variant | Eff. bits | Compression | PPL | ΔPPL vs FP16 | KL | Top-1 |
|---|---|---|---|---|---|---|
| FP16 | 16.00 | 1.00× | 13.531 | +0.0% | 0.0000 | 1.000 |
| Uniform INT8 | 8.99 | 1.78× | 13.533 | +0.0% | 0.0000 | 0.998 |
| Uniform INT4 | 5.13 | 3.12× | 13.609 | +0.6% | 0.0052 | 0.963 |
| Uniform INT2 | 3.20 | 5.01× | 16.577 | +22.5% | 0.2401 | 0.758 |
| Adaptive 8/4 0.50/0.50 | 7.41 | 2.16× | 13.542 | +0.1% | 0.0006 | 0.988 |
| Recency 8/4 0.50/0.50 | 7.08 | 2.26× | 13.531 | −0.0% | 0.0009 | 0.988 |
| Adaptive 8/4 0.25/0.75 | 6.35 | 2.52× | 13.561 | +0.2% | 0.0013 | 0.990 |
| Adaptive 8/4/2 0.10/0.40/0.50 | 4.92 | 3.25× | 13.947 | +3.1% | 0.0314 | 0.931 |
| Recency 8/4/2 0.20/0.30/0.50 | 4.92 | 3.25× | 13.832 | +2.2% | 0.0454 | 0.902 |
| Adaptive 8/4/2 0.05/0.15/0.80 | 3.92 | 4.09× | 14.356 | +6.1% | 0.0730 | 0.877 |

Figure: `results/figures/fig1_pareto_kl_Qwen2.5-0.5B-Instruct.png` (bits vs KL trade-off).

### 9.2 Key finding 1 — how keys are quantized decides everything (source: §3.1)

| Model | Variant | KL, per-token keys | KL, per-channel keys | Improvement |
|---|---|---|---|---|
| 0.5B | Uniform INT4 | 0.6545 | 0.005169 | 127× |
| 0.5B | Uniform INT2 | 8.154 | 0.2401 | 34× |
| 0.5B | Adaptive 8/4 0.50/0.50 | 0.2336 | 0.0006334 | 369× |
| 3B | Uniform INT4 | 8.006 | 0.003497 | 2290× |

With per-token keys, uniform INT4 on 3B **collapses** (PPL +263,677%); with per-channel keys it is +1.1%.
Figure: `results/figures/fig2_key_scheme_ablation.png`.

### 9.3 Key finding 2 — attention-guided beats recency at equal memory (source: §3.3)

At the **same 4.92 effective bits** on 0.5B (adaptive 8/4/2 0.10/0.40/0.50 vs recency 8/4/2 0.20/0.30/0.50, 20 windows):

| Measure | Adaptive − Recency | 95% CI | Wilcoxon p | Result |
|---|---|---|---|---|
| KL to FP16 | −0.0140 | [−0.0219, −0.0061] | 0.0027 | adaptive significantly better |
| Top-1 agreement | +0.029 | [+0.013, +0.045] | 0.0062 | adaptive significantly better |
| NLL per token | +0.0083 | [−0.0154, +0.0309] | 0.388 | no significant difference |

On 3B the direction is the same (ΔKL −0.0099) but with only 3 windows it is **not** statistically significant (p = 0.25).
Figure: `results/figures/fig3_bitmatched_Qwen2.5-0.5B-Instruct.png`.

### 9.4 Key finding 3 — why it works: importance predicts future attention (source: §3.5)

Share of the model's *future* attention that falls on tokens each policy keeps at INT8:

| Kept share (p8) | Adaptive (EMA) | Recency | Oracle (best possible) | Adaptive − Recency, p |
|---|---|---|---|---|
| 0.25 | 0.578 | 0.506 | 0.633 | +0.072, p = 1.91e-06 |
| 0.50 | 0.792 | 0.704 | 0.836 | +0.088, p = 1.91e-06 |

The EMA reaches 91–95% of the oracle; recency reaches 80–84%.
Figure: `results/figures/fig6_importance_recall_Qwen2.5-0.5B-Instruct.png`.
*(Re-run on the current code on 2026-10-22: the numbers are identical to the earlier run, and `audit.py`
no longer flags this report.)*

### 9.5 Long-context passkey retrieval (source: §3.4)

| Method | Eff. bits | Correct / 60 |
|---|---|---|
| FP16, INT8, INT4, all two-tier adaptive & recency | 5.04–16.00 | 60 / 60 |
| Adaptive 8/4/2 0.20/0.30/0.50 | 5.57 | 60 / 60 |
| Recency 8/4/2 0.20/0.30/0.50 | 4.82 | 57 / 60 |
| Adaptive 8/4/2 0.10/0.40/0.50 | 5.03 | 60 / 60 |
| Recency 8/4/2 0.10/0.40/0.50 | 4.42 | 57 / 60 |

Adaptive never lost the key; recency missed 3 with INT2 in the mix. McNemar p = 0.25, so this is
**a trend, not a significant result** (too few disagreements). It is also **not an equal-memory comparison**:
at the same tier fractions adaptive used more bits (5.57 vs 4.82, and 5.03 vs 4.42), so part of the gap may be
the extra memory. Do not present it as a win at equal memory. Figure: `results/figures/fig5_passkey_Qwen2.5-0.5B-Instruct.png`.

### 9.6 Memory and speed vs context length (0.5B, source: §3.6)

| Method | Context | Stored KV (MiB) | Compression | Decode (ms/token) | Peak VRAM (GB) |
|---|---|---|---|---|---|
| FP16 | 2048 | 24.36 | 1.00× | 14.4 | 1.42 |
| Uniform INT8 | 2048 | 13.43 | 1.81× | 46.0 | 1.42 |
| Uniform INT4 | 2048 | 7.39 | 3.30× | 46.7 | 1.42 |
| Recency | 2048 | 10.42 | 2.34× | 68.2 | 1.42 |
| Adaptive | 2048 | 11.03 | 2.21× | 69.8 | 1.42 |

Stored bytes shrink 2.2× (adaptive) to 3.3× (INT4); decoding is **slower** and peak VRAM is **unchanged** (see Limitations).
Figure: `results/figures/fig4_kv_memory_vs_context.png`.

### 9.7 Extrapolated (NOT measured — label clearly on any slide)

Measured compression ratio × FP16 KV size from each model's config, at 32,768 tokens context:

| Model | FP16 KV (MiB) | Adaptive KV (MiB, extrapolated) | INT4 KV (MiB, extrapolated) |
|---|---|---|---|
| Qwen2.5-0.5B | 384.0 | 177.9 | 122.8 |
| Qwen2.5-3B | 1152.0 | 533.8 | 368.4 |
| Qwen2.5-7B | 1792.0 | 830.4 | 573.1 |

### 9.8 Verification

- Test suite: **71 passed, 0 failed** (`results/VERIFICATION.md`, `results/pytest_report.xml`).
- The FP16 decode engine reproduces plain Hugging Face greedy output exactly.
- Hardware safety logged every run (e.g. max 75 °C, 0 thermal pauses in the main 0.5B quality run).

---

## 10. Conclusions (honest version — the coordinator will check this)

1. **Per-channel key quantization is the enabling step.** Without it, INT4 and INT2 caches are unusable;
   with it, uniform INT4 costs only +0.6% perplexity at 3.12× compression.
2. **Attention guidance works in the mixed-precision regime.** At equal memory it gives significantly
   lower KL and higher top-1 agreement than recency, and it captures more of the attention the model will use.
3. **It does not beat the best uniform format.** Under the pre-registered Pareto rule, no INT2-containing
   mix beat uniform INT4 at equal or fewer bits. Because per-channel INT4 is already near-lossless, the
   default is two-tier INT8/INT4 and INT2 is reported as a negative ablation.
4. **Savings are in stored bytes, not peak VRAM or speed** — that requires fused GPU kernels.

---

## 11. Limitations

| Limitation | Why |
|---|---|
| Peak VRAM not reduced | The cache is dequantized to FP16 for each attention call |
| Slower than FP16 (≈ 5× ms/token) | Pure PyTorch dequantization every step; no fused kernels |
| Adaptive uses more bits than recency at the same fractions | Scattered tiers store more per-channel key scale sets — hence the bit-matched comparison |
| 3B results are a confirmation only | 3 windows, CPU-offloaded weights |
| Quality measured on WikiText-2 + synthetic passkey | No downstream benchmarks (QA, summarisation) |
| Demotion only, no promotion | Original precision is discarded once a token is demoted |

## 12. Future work

- Fused dequantize-attention CUDA/Triton kernel → real peak-memory and speed gains.
- Larger models fully on GPU and longer contexts (8K–32K) to replace the extrapolated numbers with measurements.
- Downstream benchmarks (LongBench, QA).
- Learned or per-layer importance weights; per-head tiers.
- Comparison with published methods (KIVI, H2O, KVQuant) under the same accounting.

---

## 13. Live demo (dashboard)

```bash
cd ~/Desktop/AdaptiveKV-prototype
.venv/bin/python -m streamlit run dashboard/app.py --server.port 8501
# open http://localhost:8501
```

Type a prompt → **Run comparison**. It shows FP16 and Adaptive outputs side by side, token agreement,
and measured KV memory per token, effective bits, GPU peak memory, latency and speed with % change.
Suggested demo prompt: "What is the capital of France?" (short), then a long paragraph + question (longer cache).

## 14. How to reproduce every result

```bash
source .venv/bin/activate
python -m pytest tests -q --junitxml=results/pytest_report.xml
python decision_test.py
python passkey.py
python importance_analysis.py
python benchmarks/run_baselines.py --config config_bench.yaml && python benchmarks/summarise.py
python benchmarks/extrapolate.py
python benchmarks/make_report.py && python benchmarks/make_figures.py && python benchmarks/make_paper_report.py
# 3B model: add --model Qwen/Qwen2.5-3B-Instruct --gpu-mem-gb 2.8
```

---

## 15. Suggested PPT outline (≈ 15 slides)

| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | Title | Title, subtitle, your name, guide, college, date | — |
| 2 | Motivation | LLMs are memory-bound; KV cache grows with context | Bar: FP16 KV 3.36 → 24.36 MiB (256 → 2048 tokens) |
| 3 | What is a KV cache | K/V stored per token per layer; 12,288 B/token for 0.5B | Simple block diagram |
| 4 | Problem & objectives | §2 question + §3 objectives | Bullets |
| 5 | Core idea | Important → INT8, others → INT4 (INT2 tested); demotion only; sinks + recent window | §4 box, one-way arrows |
| 6 | Workflow | Full pipeline | §5 diagram |
| 7 | Quantization details | Asymmetric groups, bit packing, per-channel keys vs per-token values | Small matrix sketch |
| 8 | Architecture & tools | Module table, hardware | §6/§7 tables |
| 9 | Experimental setup | Models, data, baselines, metrics, statistics | §8 |
| 10 | Result 1: key granularity | 127× / 2290× KL improvement | `fig2_key_scheme_ablation.png` |
| 11 | Result 2: quality vs memory | Main table (trimmed) | `fig1_pareto_kl_Qwen2.5-0.5B-Instruct.png` |
| 12 | Result 3: adaptive vs recency | Equal-bits test, p = 0.0027 | `fig3_bitmatched_…png` + `fig6_importance_recall_…png` |
| 13 | Result 4: passkey & memory scaling | 60/60 vs 57/60 (not significant, and adaptive used more bits); MiB vs context | `fig5_passkey_…png`, `fig4_kv_memory_vs_context.png` |
| 14 | Conclusions & limitations | §10 + §11 | Two columns |
| 15 | Future work + live demo | §12, then switch to dashboard | Screenshot of dashboard |

Tip: use the PNGs in `results/figures/` for slides and the PDFs (vector) for the written report.

## 16. Suggested report chapters

1. **Introduction** — LLM inference, memory wall, motivation, objectives, contributions.
2. **Background & literature review** — Transformers and attention, KV cache, quantization basics
   (symmetric/asymmetric, group-wise), related work: KIVI (per-channel keys), H2O / heavy-hitter eviction,
   StreamingLLM (attention sinks), KVQuant.
3. **Proposed system** — §4–§6: idea, workflow, importance + EMA, allocator rules, storage format, accounting.
4. **Implementation** — modules, config, hooks, memory-safe attention capture, safety guard, tests.
5. **Experimental setup** — §7–§8.
6. **Results & discussion** — §9 tables + figures, with interpretation.
7. **Conclusion, limitations & future work** — §10–§12.
8. **References** and **Appendix** (full tables from `results/PAPER_REPORT.md`, test list from `results/VERIFICATION.md`).

A short coordinator PDF is at `results/report.pdf`, built by `benchmarks/make_report_pdf.py`.
Fill in the names when you build it:
`python benchmarks/make_report_pdf.py --authors "..." --guide "..." --college "..."`.
---

## 17. Likely coordinator questions — prepared answers

- **"Are these numbers real?"** Yes. Every number comes from a run on this machine; memory is summed from
  real tensor `.nbytes`; reports are generated by scripts and never hand-edited; extrapolations are labelled.
- **"Why not just use uniform INT4?"** Honest answer: with per-channel keys, uniform INT4 is already
  near-lossless and is on the Pareto frontier. Attention guidance is better than recency *when mixing
  precisions*, which matters when going below 4 bits or protecting long-range facts (passkey).
- **"Does it save GPU memory?"** Stored KV bytes, yes (2.2–3.3×). Peak VRAM, no — the prototype dequantizes
  before attention. A fused kernel is the next step.
- **"Why is it slower?"** Dequantization runs as separate PyTorch ops every step; production systems fuse it into attention.
- **"Why effective bits above 8/4/2?"** Scales, zero-points, FP16 recent window, pending keys and tier index are all counted.
- **"Why EMA?"** A single step's attention is noisy; EMA keeps tokens that are consistently important
  (shown to capture 91–95% of the oracle's future attention).
- **"Why demotion only?"** Once a token is stored at lower precision its exact values are gone; promoting it would not recover them.

## 18. Glossary

| Term | Meaning |
|---|---|
| KV cache | Stored Key/Value vectors of past tokens reused at every generation step |
| Memory wall | Performance limited by memory size/bandwidth rather than compute |
| Quantization | Storing numbers with fewer bits (e.g. FP16 → INT4) using a scale and zero-point |
| Per-channel / per-token | Whether one scale is shared along a feature channel or along one token |
| EMA | Exponential moving average — a smoothed running score |
| Attention sink | Early tokens that receive large attention regardless of content |
| Perplexity | exp(average negative log-likelihood); lower = better prediction |
| KL divergence | Distance between two probability distributions |
| Pareto frontier | Set of options where nothing else is both cheaper and better |
| GQA | Grouped-query attention: few KV heads shared by many query heads |
