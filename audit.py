"""
audit.py - read-only check of the whole project. Changes nothing.

Run from the project root:   python audit.py
Then paste the full output back.
"""
import re
import sys
import pathlib

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
issues, oks = [], []


def read(rel):
    p = root / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def issue(f, msg):
    issues.append(f"ISSUE  {f}: {msg}")


def ok(f, msg):
    oks.append(f"ok     {f}: {msg}")


# ---- 1. core files: are they the fixed versions? -------------------------
fixed_markers = {
    "adaptivekv/generate.py":   ("def run_method", "shared engine (fixed version)"),
    "adaptivekv/cache.py":      ("class _TierBank", "tier-bank cache (fixed version)"),
    "adaptivekv/quant.py":      ("FP16_MIN_NORMAL", "robust scale/zero-point (fixed version)"),
    "adaptivekv/importance.py": ("drop_attn", "memory-safe attention hook (fixed version)"),
    "adaptivekv/allocator.py":  ("def pinned_mask", "fixed allocator"),
    "adaptivekv/metrics.py":    ("def compute_perplexity_kv", "KV-aware perplexity (fixed version)"),
    "dashboard/app.py":         ("Run comparison", "comparison dashboard (latest version)"),
}
for f, (marker, label) in fixed_markers.items():
    t = read(f)
    if t is None:
        issue(f, "file missing")
    elif marker in t:
        ok(f, label)
    else:
        issue(f, f"NOT the fixed version (marker '{marker}' not found). Overwritten or never copied.")

for f in ("sweep.py", "diagnose_cache.py"):
    if read(f) is None:
        issue(f, "not in the project root yet (copy it from the chat)")

# ---- 2. hacks that hide the real method (e.g. edits by an AI agent) -------
g = read("adaptivekv/generate.py") or ""
hacks = [
    (r"np\.full\(\s*prefill_len\s*,\s*8", "prefill forced to INT8 -> prompt is never compressed"),
    (r"new_tok_tier\s*=\s*np\.array\(\[8\]\)\s*#?.*(Always|always)", "generated tokens forced INT8 -> not adaptive"),
    (r"#\s*DISABLED", "reassignment commented out ('DISABLED') -> tiers never change"),
    (r"fp16_per_token\s*=\s*12288|\b12288\b", "hardcoded 12288 bytes/token"),
]
for pat, msg in hacks:
    if re.search(pat, g):
        issue("adaptivekv/generate.py", msg)

# ---- 3. config.yaml ---------------------------------------------------------
cfg_text = read("config.yaml")
if cfg_text is None:
    issue("config.yaml", "missing")
else:
    cfg = {}
    for line in cfg_text.splitlines():
        m = re.match(r"^([A-Za-z_0-9]+):\s*(.+?)\s*(#.*)?$", line)
        if m:
            cfg[m.group(1)] = m.group(2)
    print("config.yaml values:", {k: cfg.get(k, "<MISSING>") for k in (
        "model_id", "dtype", "attn_impl", "p8", "p4", "quant_group_size", "sink_tokens",
        "recent_tokens", "reassign_every", "importance_window", "k_channel_norm", "ema_alpha")})
    for k, why in (("sink_tokens", "0 lets the attention-sink token drop below the protected precision"),):
        if cfg.get(k, "0") in ("0", "null", "None"):
            issue("config.yaml", f"{k} is 0/missing: {why}")
    for k in ("dtype", "importance_window", "k_channel_norm", "recent_tokens", "reassign_every"):
        if k not in cfg:
            issue("config.yaml", f"key '{k}' missing (code falls back to a default)")
    try:
        p8, p4 = float(cfg["p8"]), float(cfg["p4"])
        print(f"  -> configured INT8/INT4 shares = {p8:.0%}/{p4:.0%}")
        if abs(p8 + p4 - 1.0) > 1e-9:
            issue("config.yaml", "p8 + p4 must equal 100% because only INT8 and INT4 are supported")
    except (KeyError, ValueError):
        issue("config.yaml", "p8/p4 missing or not numeric")

# ---- 4. tests ---------------------------------------------------------------
tests = sorted((root / "tests").glob("*.py")) if (root / "tests").exists() else []
for p in tests:
    name = p.name
    t = p.read_text(encoding="utf-8", errors="replace")
    if name != "__init__.py" and not (name.startswith("test_") or name.endswith("_test.py")):
        issue(f"tests/{name}", "name does not match pytest pattern, so pytest will NOT run it")
    if re.search(r"_storage|TokenEntry", t):
        issue(f"tests/{name}", "uses old cache internals (_storage/TokenEntry) -> will fail with the new cache")
    if re.search(r"compute_perplexity\(", t):
        issue(f"tests/{name}", "tests the old single-forward perplexity")

# ---- 5. benchmarks ------------------------------------------------------------
bdir = root / "benchmarks"
for p in sorted(bdir.glob("*.py")) if bdir.exists() else []:
    t = p.read_text(encoding="utf-8", errors="replace")
    f = f"benchmarks/{p.name}"
    if re.search(r"compute_perplexity\(", t) and "compute_perplexity_kv" not in t:
        issue(f, "uses compute_perplexity (no KV quantization) -> identical perplexity for every method")
    if re.search(r"\b12288\b", t):
        issue(f, "hardcoded 12288 bytes/token")
    if re.search(r"torch\.float16", t) and "bfloat16" not in t:
        issue(f, "loads model in float16 (dashboard/sweep use the dtype in config.yaml)")
    if re.search(r"_storage|TokenEntry", t):
        issue(f, "uses old cache internals")

# ---- 6. stale numbers anywhere --------------------------------------------------
for p in list(root.glob("*.md")) + list((root / "results").glob("*")) if (root / "results").exists() else list(root.glob("*.md")):
    if p.is_file():
        t = p.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\b12288\b|~?\d+\s*%\s*(memory )?savings", t, flags=re.I):
            issue(str(p.relative_to(root)), "contains hardcoded/estimated numbers (12288 or '~N% savings')")

gm = (root / "adaptivekv/generate.py")
if gm.exists():
    for p in (root / "results").glob("*") if (root / "results").exists() else []:
        if p.is_file() and p.stat().st_mtime < gm.stat().st_mtime:
            issue(f"results/{p.name}", "older than the current generate.py -> produced by old code, regenerate")

# ---- report ---------------------------------------------------------------------
print()
for line in oks:
    print(line)
print()
for line in issues:
    print(line)
print(f"\n{len(oks)} ok, {len(issues)} issues")