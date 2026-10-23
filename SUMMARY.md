# AdaptiveKV — Summary

Mixed-precision (INT8/INT4/INT2) KV cache guided by attention importance, with
per-channel key quantization, on Qwen2.5 (0.5B fully on GPU; 3B with CPU offload)
on a 4 GB RTX 3050.

All numbers live in generated files (this summary copies none, so it cannot go stale):

| What | Where |
|---|---|
| Paper draft: abstract, method, results, figures, limitations | `results/PAPER_REPORT.md` |
| Every test case and every measured number | `results/VERIFICATION.md` |
| Per-experiment reports | `results/DECISION_REPORT_*.md`, `PASSKEY_REPORT_*.md`, `IMPORTANCE_REPORT_*.md` |
| Figures (PNG + vector PDF) | `results/figures/` |
| Superseded runs (old 3-tier code) | `results/archive_3tier_pre_2026-10-13/` |

Regenerate everything: see "Quick Start" in `README.md`.
