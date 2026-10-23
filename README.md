# AdaptiveKV

**Software-Based Memory-Wall Mitigation in LLM Inference**

Mixed-precision KV cache: each cached token is stored at INT8 / INT4 / INT2,
chosen by an EMA of the attention it receives (attention-guided), under fixed
tier quotas. Keys are quantized **per channel** (KIVI-style; values per token),
which is what makes INT4/INT2 usable at all.

**Results:** see `results/PAPER_REPORT.md` (paper draft with figures) and
`results/VERIFICATION.md` (every test case + every measured number). Both are
generated from `results/`; this README deliberately contains no copied numbers,
so it cannot go stale.

---
## Ground rules (enforced)

1. Every number in any output is computed from a real model run.  
2. Memory is measured from `tensor.nbytes` (packed integers) — not formulas.  
3. Effective bits include scale + zero-point overhead.  
4. Quality claims require measured evidence against FP16 baseline.  
5. Slower-than-FP16 results are reported honestly (see §Results).  
6. Extrapolated values are clearly labelled and kept separate.  

---

## Environment (measured)

| Item | Value |
|------|-------|
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128 |
| transformers | 5.18.0 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 3.68 GB VRAM |
| Model | Qwen/Qwen2.5-0.5B-Instruct, FP16, eager attention |
| FP16 KV bytes/token | 12288 (2 × 2 KV-heads × 64 head-dim × 2 bytes × 24 layers, **verified**) |

---

---

## Quick Start

```bash
source .venv/bin/activate
python benchmarks/download_data.py                 # WikiText-2 (already in data/)

python -m pytest tests -q --junitxml=results/pytest_report.xml   # test suite
python decision_test.py        # quality vs memory: PPL / KL / top-1 through each cache, paired stats
python passkey.py              # needle retrieval through each cache
python importance_analysis.py  # does attention importance predict future attention?
python benchmarks/run_baselines.py --config config_bench.yaml && python benchmarks/summarise.py
python benchmarks/extrapolate.py                   # EXTRAPOLATED projections (labelled)

python benchmarks/make_report.py                   # results/VERIFICATION.md
python benchmarks/make_figures.py                  # results/figures/*.png|pdf
python benchmarks/make_paper_report.py             # results/PAPER_REPORT.md

./run_queue.sh logs/run "python decision_test.py" "python passkey.py"   # sequential GPU queue
./run_dashboard.sh                                 # Streamlit dashboard
```

Larger model with CPU offload (4 GB GPU): add `--model Qwen/Qwen2.5-3B-Instruct --gpu-mem-gb 2.8`.
Every experiment pauses at 80 °C GPU and stops below 1.5 GB free RAM (`safety` in config.yaml).

---

## Honest Limitations

| Issue | Detail |
|-------|--------|
| Peak VRAM not reduced | The cache is dequantized to FP16 for each attention call; only stored bytes shrink |
| Slower than FP16 | Pure PyTorch dequantization every step; fused kernels would be needed |
| Adaptive costs extra bits at equal fractions | Scattered tiers store more per-channel key scale sets; compare at matched bits |
| 3B results | Few windows and CPU-offloaded weights (time budget); a confirmation, not a precise estimate |
| Demotion re-quantizes | From stored values (lossy); promotion is not supported |

---

## File Layout

```
config.yaml              all parameters (none buried in code)
adaptivekv/
  quant.py               INT8/INT4/INT2 quantize/dequant + bit packing
  importance.py          attention hook (no output_attentions needed) + EMA tracker
  allocator.py           quota-capped, demotion-only tier assignment (2 or 3 tiers)
  cache_channel.py       ChannelKVCache: per-channel keys, per-token values/tiers (default)
  cache.py               AdaptiveKVCache: previous per-token scheme (ablation)
  generate.py            decode engine for fp16 / uniform / recency / adaptive
  metrics.py             perplexity helpers, token agreement
  safety.py              in-process GPU-temperature / RAM guard
decision_test.py  passkey.py  importance_analysis.py   paper experiments
benchmarks/              run_baselines, summarise, extrapolate, make_report, make_figures, make_paper_report
tests/                   unit + integration tests (count and results: results/VERIFICATION.md)
results/                 generated only, never hand-edited (archive_* = superseded runs)
```
