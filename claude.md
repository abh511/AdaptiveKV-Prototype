# AdaptiveKV: Software-Based Memory-Wall Mitigation in LLM Inference

## What this project is (read this first)

When an LLM generates text, it stores the Key and Value vectors (the "KV cache") of every
token it has already seen, so it does not recompute them. This cache grows linearly with
context length and becomes a major memory and bandwidth cost: the "memory wall".

**AdaptiveKV reduces KV-cache memory by storing different cached tokens at different numeric
precision, depending on how important each token is to the model.** Important tokens keep
high precision; less important tokens are stored in fewer bits. It is purely software
(PyTorch, no custom kernels) and runs on top of a Hugging Face Qwen model.

This is a university-style prototype. The project coordinator is strict and audits every
number, so correctness and honest measurement matter more than impressive-looking results.

## The core idea

```
User prompt
   -> Qwen LLM
   -> attention scores
   -> attention-based importance per cached token
   -> EMA importance tracking (smoothed over time)
   -> importance ranking
   -> precision tiers:  INT8 (high importance) | INT4 (medium) | INT2 (low)
   -> adaptive KV cache (packed integer storage)
   -> next-token generation  (loop; tiers are re-evaluated as generation proceeds)
```

1. **Monitor** attention-based importance of each cached token.
2. **Track** importance over time with an exponential moving average (EMA).
3. **Rank** tokens by importance and **assign** INT8 for the top fraction `p8`, INT4 for the next
   fraction `p4`, INT2 for the rest.
4. **Keep it adaptive**: re-evaluate and re-assign precision as more tokens are generated.
5. **Output**: the generated text, reduced KV-cache memory, and a lower effective bits-per-value,
   while trying to maintain model quality.

## Key concepts

- **Importance score**: how much attention the model pays to a cached token (averaged over
  layers/heads, from the most recent query rows). Higher = more important.
- **EMA**: `importance_t = alpha * score_t + (1 - alpha) * importance_(t-1)`; a new token starts at its raw score.
- **Tier**: the bit width a token's K and V are stored at (8, 4 or 2). One tier per token, shared by all layers.
- **Quantization**: asymmetric, per group of values (scale + zero-point per group, stored FP16). INT4 is packed
  2 values per byte, INT2 packed 4 per byte. Dequantized to full precision when attention reads the cache.
- **Effective bits**: real stored bytes (data + scales + zero-points + metadata) * 8 / number of FP16 values
  represented. Always larger than the nominal 8/4/2 because of scale overhead.
- **Attention sink**: the first few tokens receive disproportionate attention and are normally protected.
- **Recent window**: the newest tokens are normally protected.
- **Baselines for comparison**: FP16 cache, uniform INT8, uniform INT4, uniform INT2, and a recency heuristic
  (newest tokens get higher precision), all measured the same way.

## Setup and hardware

- Model: `Qwen/Qwen2.5-0.5B-Instruct` (GQA: few KV heads; read layers/heads/head_dim from `model.config`, never assume).
- Hardware: NVIDIA RTX 3050 laptop GPU, 4 GB VRAM, 16 GB RAM, Ubuntu, Python venv at `.venv`.
- Attention weights require `attn_implementation="eager"`; fused kernels do not return them.
- Never store every layer's full attention matrix at once on this GPU (it runs out of memory).
  Reduce attention to a per-token vector inside a hook, then discard it.

## Project layout

```
adaptivekv/quant.py        quantize/dequantize + bit packing (INT8/INT4/INT2)
adaptivekv/importance.py   attention hook + EMA tracker
adaptivekv/allocator.py    ranking and tier assignment
adaptivekv/cache.py        adaptive KV cache (packed storage) + memory accounting
adaptivekv/generate.py     generation loop / decode engine for all methods
adaptivekv/metrics.py      perplexity (through the quantized cache), token agreement
dashboard/app.py           Streamlit live comparison page (FP16 vs Adaptive)
benchmarks/                benchmark + report scripts (write only to results/)
tests/                     pytest suite
results/                   generated CSV/JSON only; never edit by hand
data/wikitext2_test.txt    real text used for evaluation
config.yaml                every tunable parameter (p8, p4, ema_alpha, group size, windows, ...)
sweep.py  passkey.py  diagnose_cache.py  audit.py   evaluation / diagnosis scripts
```

## Commands

```bash
source .venv/bin/activate
pytest tests/ -v                       # tests
streamlit run dashboard/app.py         # dashboard (fully stop and restart after code changes)
python sweep.py                        # perplexity through each KV-cache method
python passkey.py                      # needle retrieval test
python benchmarks/run_benchmark.py     # memory / speed benchmark across context lengths
python benchmarks/make_report.py       # builds VERIFICATION.md from results/*.csv
python diagnose_cache.py               # isolates model vs engine vs cache problems
python audit.py                        # read-only project health check
```

## Ground rules (non-negotiable)

1. **No fabricated numbers.** Every number shown anywhere comes from code run on this machine. No estimates,
   no "~N% savings", no hardcoded results or hardcoded bytes-per-token. If something cannot be measured, say so.
2. **Memory is measured from real stored tensors** (sum of `.nbytes`), not from a formula. Anything extrapolated
   must be labelled EXTRAPOLATED and kept separate from measured results.
3. **Never claim quality is maintained without measured evidence** against the FP16 baseline. Report drops honestly.
4. **Read the code and library versions before changing anything.** Do not assume APIs. Ask if unsure.
5. **Tunable choices live in `config.yaml`**, not buried in code.
6. **Never weaken or delete a test to make it pass.** Show real, unedited terminal output.
7. **Do not change the project's core idea or design without asking me first.** Propose, explain the evidence, wait.
8. `results/` is generated output. Regenerate it; never edit it by hand.

## Honest limitations to keep in mind

- The cache is dequantized to full precision for each attention call, so **peak GPU memory is not reduced**; only the
  bytes stored between steps shrink. Real peak-memory savings would need fused dequantize-attention kernels.
- The 0.5B model's KV cache is small in absolute terms, so absolute savings look small; ratios and effective bits are the result.

---

## Status notes (optional; delete this section if you want only the original idea)

Measured on this machine so far, so work is not repeated:
- The base model and the decode engine are correct (FP16 engine output equals plain Hugging Face greedy output).
- INT8 KV reproduces FP16 output closely (about 1% reconstruction error). INT4 is usable (11-16% error).
- **INT2 with plain per-token group quantization is noise** (57% key / 85% value error) and destroyed generation quality.
  Methods that make INT2 work (e.g. OSCAR) rely on calibrated rotations, clipping and custom kernels on larger models.
- An earlier "demote any token ever ranked low" rule made far more tokens INT2 than configured; demotion is now
  quota-capped and never promotes.
- Current default is INT8/INT4 two-tier (`int2_enabled: false`); INT2 remains available as an ablation.
- Not yet known: whether attention-based importance beats the recency heuristic at equal bits. `sweep.py` and
  `passkey.py` exist to answer that. Results of the latest runs are in `results/`.