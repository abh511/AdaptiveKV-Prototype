"""
benchmarks/make_figures.py
--------------------------
Paper figures, generated ONLY from files in results/ (no typed-in numbers).
Writes results/figures/<name>.png (200 dpi) and .pdf (vector, for LaTeX).

  fig1_pareto_kl_<model>        effective bits vs KL to FP16 (quality-memory trade-off)
  fig2_key_scheme_ablation      per-token vs per-channel keys, every model available
  fig3_bitmatched_<model>       per-window KL, adaptive vs recency at matched memory
  fig4_kv_memory_vs_context     stored KV bytes vs context length (benchmark)
  fig5_passkey_<model>          retrieval accuracy with Wilson 95% CIs
  fig6_importance_recall_<model> share of future attention kept at INT8 (EMA vs recency vs oracle)

Colors follow the method, never its rank, and are the same in every figure:
adaptive = blue, recency = orange, uniform = aqua, int8 = yellow (memory plot
only), FP16 / previous key scheme / oracle = muted gray. Marker shape is a
second cue, so identity never relies on color alone (validated palette:
validate_palette.js, light surface, all pairs: PASS; aqua is < 3:1 contrast,
so points carry direct labels and every figure has a table in the report).
Usage: python benchmarks/make_figures.py
"""

import csv
import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"adaptive": "#2a78d6", "recency": "#eb6834", "uniform": "#1baf7a", "int8": "#eda100",
       "ref": MUTED, "old": MUTED, "oracle": MUTED}
MARK = {"adaptive": "o", "recency": "s", "uniform": "^", "old": "X", "ref": "D"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2,
})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def family(variant: str, method: str) -> str:
    if "[per-token keys]" in variant:
        return "old"
    if method == "fp16":
        return "ref"
    if method in ("adaptive", "recency"):
        return method
    return "uniform"


def short(variant: str) -> str:
    v = variant.replace("[per-token keys]", "(old keys)").replace("_uniform", "")
    v = v.replace("adaptive ", "A ").replace("recency ", "R ")
    return re.sub(r"0\.(\d\d)", r".\1", v)


def place_labels(ax, items, fontsize=6.5):
    """Direct labels without overlaps: try a few offsets per point, keep the first free one."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    taken = []
    for x, y, _ in items:                          # markers are obstacles too
        px, py = ax.transData.transform((x, y))
        taken.append(Bbox.from_extents(px - 9, py - 9, px + 9, py + 9))
    if ax.get_legend() is not None:                # never write over the legend
        taken.append(ax.get_legend().get_window_extent(r))
    offsets = [(5, 3, "left"), (5, -10, "left"), (-5, 3, "right"), (-5, -10, "right"),
               (5, 11, "left"), (5, -18, "left"), (-5, 11, "right"), (-5, -18, "right"), (0, 14, "center"),
               (0, -22, "center"), (8, 20, "left"), (8, -27, "left"), (-8, 20, "right"), (-8, -27, "right"),
               (0, 26, "center"), (0, -32, "center")]
    for x, y, text in items:
        for dx, dy, ha in offsets:
            a = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            fontsize=fontsize, color=INK2)
            bb = a.get_window_extent(r).expanded(1.04, 1.15)
            if not any(bb.overlaps(t) for t in taken):
                taken.append(bb)
                break
            a.remove()
        else:
            taken.append(ax.annotate(text, (x, y), xytext=(5, 3), textcoords="offset points",
                                     fontsize=fontsize, color=INK2).get_window_extent(r))


def frontier(points):
    """Lower envelope (Pareto frontier) of (bits, loss) points, sorted by bits."""
    out, best = [], float("inf")
    for x, y in sorted(points):
        if y < best:
            out.append((x, y))
            best = y
    return out


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png/.pdf")


def model_tags(prefix):
    return sorted(re.sub(rf"^{prefix}_|_summary\.csv$|\.csv$", "", os.path.basename(p))
                  for p in glob.glob(os.path.join(RES, f"{prefix}_*.csv")))


# ---------------------------------------------------------------------------

def fig_pareto(tag):
    rows = [r for r in read_csv(os.path.join(RES, f"decision_{tag}_summary.csv")) if fnum(r.get("n_windows")) > 0]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    seen = set()
    labels, fam_pts = [], {}
    for r in rows:
        fam = family(r["variant"], r["method"])
        if fam == "ref":
            continue
        x, y = fnum(r["eff_bits"]), max(fnum(r["kl"]), 1e-5)
        lo, hi = max(fnum(r["kl_ci_lo"]), 1e-5), max(fnum(r["kl_ci_hi"]), 1e-5)
        lab = {"adaptive": "Adaptive (attention importance)", "recency": "Recency heuristic",
               "uniform": "Uniform precision", "old": "Previous key scheme (per-token)"}[fam]
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=MARK[fam], ms=7, color=COL[fam],
                    mec=SURFACE, mew=1.5, ecolor=COL[fam], elinewidth=1, capsize=0,
                    label=None if fam in seen else lab, zorder=3)
        seen.add(fam)
        labels.append((x, y, short(r["variant"])))
        fam_pts.setdefault(fam, []).append((x, y))
    for fam in ("adaptive", "recency"):
        f = frontier(fam_pts.get(fam, []))
        if len(f) > 1:
            fx, fy = zip(*f)
            ax.plot(fx, fy, color=COL[fam], lw=1.2, alpha=0.55, zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Effective bits per KV value (incl. scales, zero-points, FP16 window, index)")
    ax.set_ylabel("KL(FP16 ‖ method), nats/token  (log)")
    ax.set_title(f"Quality vs memory: {tag}")
    ax.legend(loc="upper right", fontsize=7.5)
    place_labels(ax, labels)
    ax.text(0.0, -0.17, "Lower-left is better. Vertical bars: bootstrap 95% CI over windows. "
            "Thin lines: each method's Pareto frontier.", transform=ax.transAxes,
            fontsize=7, color=MUTED)
    save(fig, f"fig1_pareto_kl_{tag}")


def fig_key_ablation(tags):
    pairs = [("int4_uniform", "int4_uniform [per-token keys]", "Uniform\nINT4"),
             ("int2_uniform", "int2_uniform [per-token keys]", "Uniform\nINT2"),
             ("adaptive 8/4 0.50/0.50", "adaptive 8/4 0.50/0.50 [per-token keys]", "Adaptive 8/4\n.50/.50")]
    data = {}
    for tag in tags:
        s = {r["variant"]: r for r in read_csv(os.path.join(RES, f"decision_{tag}_summary.csv"))
             if fnum(r.get("n_windows")) > 0}
        data[tag] = [(name, fnum(s[new]["kl"]) if new in s else None, fnum(s[old]["kl"]) if old in s else None)
                     for new, old, name in pairs]
    data = {t: d for t, d in data.items() if any(n is not None and o is not None for _, n, o in d)}
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(3.6 * len(data), 3.6), sharey=True, squeeze=False)
    for ax, (tag, d) in zip(axes[0], data.items()):
        xs = np.arange(len(d))
        w = 0.36
        for i, (name, new, old) in enumerate(d):
            for dx, val, c, lab in ((-w / 2 - 0.01, old, COL["old"], "Per-token keys (previous)"),
                                    (w / 2 + 0.01, new, COL["adaptive"], "Per-channel keys (this work)")):
                if val is None or math.isnan(val):
                    continue
                ax.bar(i + dx, val, w, color=c, label=lab if i == 0 else None, zorder=3)
                ax.annotate(f"{val:.3g}", (i + dx, val), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(xs, [n for n, _, _ in d], fontsize=7.5)
        ax.set_yscale("log")
        ax.set_title(tag, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("KL(FP16 ‖ method)  (log)")
    lo = min(v for d in data.values() for _, n, o in d for v in (n, o) if v)
    hi = max(v for d in data.values() for _, n, o in d for v in (n, o) if v)
    axes[0][0].set_ylim(lo / 3, hi * 8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Key quantization scheme: per-token vs per-channel", fontsize=10, fontweight="bold", color=INK)
    save(fig, "fig2_key_scheme_ablation")


def fig_bitmatched(tag):
    wpath = os.path.join(RES, f"decision_{tag}_windows.csv")
    rep = os.path.join(RES, f"DECISION_REPORT_{tag}.md")
    if not (os.path.exists(wpath) and os.path.exists(rep)):
        return
    m = re.search(r"## Question 2b.*?\n\|---.*?\n(.*?)\n\n", open(rep, encoding="utf-8").read(), re.S)
    if not m:
        return
    pairs = [[c.strip() for c in line.strip("|").split("|")][:2] for line in m.group(1).splitlines() if line.startswith("|")]
    rows = read_csv(wpath)
    pairs = [p for p in pairs if len(p) == 2][:4]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(max(4.2, 3.6 * len(pairs)), 3.8), squeeze=False)
    for ax, (a, r) in zip(axes[0], pairs):
        ka = {x["window"]: fnum(x["kl_mean"]) for x in rows if x["variant"] == a and not x["error"]}
        kr = {x["window"]: fnum(x["kl_mean"]) for x in rows if x["variant"] == r and not x["error"]}
        w = sorted(set(ka) & set(kr))
        xa, xr = np.array([kr[i] for i in w]), np.array([ka[i] for i in w])
        lim = [min(xa.min(), xr.min()) * 0.8, max(xa.max(), xr.max()) * 1.2]
        ax.plot(lim, lim, color=AXIS, lw=1, zorder=1)
        ax.scatter(xa, xr, s=28, color=COL["adaptive"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        better = int((xr < xa).sum())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"Recency KL  [{short(r)}]", fontsize=7.5)
        ax.set_ylabel(f"Adaptive KL  [{short(a)}]", fontsize=7.5)
        ax.set_title(f"{better}/{len(w)} windows: adaptive lower", fontsize=8.5)
    fig.suptitle(f"Matched memory, per window: {tag}\n(below the diagonal = adaptive better)",
                 fontsize=9.5, fontweight="bold", color=INK, y=1.04)
    save(fig, f"fig3_bitmatched_{tag}")


def fig_memory():
    path = os.path.join(RES, "summary.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    series = [("fp16", "FP16", COL["ref"], "D", (0, (4, 2))), ("int8_uniform", "Uniform INT8", COL["int8"], "v", "-"),
              ("int4_uniform", "Uniform INT4", COL["uniform"], "^", "-"),
              ("adaptive", "Adaptive", COL["adaptive"], "o", "-"), ("recency", "Recency", COL["recency"], "s", "-")]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ends = []
    for m, lab, c, mk, ls in series:
        pts = sorted((fnum(r["context_length"]), fnum(r["kv_bytes_real_mean"]) / 2 ** 20)
                     for r in rows if r["method"] == m and fnum(r.get("kv_bytes_real_mean")) > 0)
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, ls=ls, color=c, marker=mk, ms=6, mec=SURFACE, mew=1.2, label=lab, zorder=3)
        ends.append([x[-1], y[-1], lab])
    merged = []                                   # identical byte counts -> one combined end label
    for e in sorted(ends, key=lambda e: e[1]):
        if merged and abs(e[1] - merged[-1][1]) <= 0.03 * max(e[1], 1e-9):
            merged[-1][2] += " / " + e[2]
        else:
            merged.append(e)
    fig.canvas.draw()
    prev_py = None
    for x, y, lab in merged:                       # bottom-up; keep >= 11 px between end labels
        py = ax.transData.transform((x, y))[1]
        dy = 0 if prev_py is None else max(0, prev_py + 11 - py)
        prev_py = py + dy
        ax.annotate(lab, (x, y), xytext=(6, dy * 72 / fig.dpi), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048], ["256", "512", "1024", "2048"])
    ax.set_xlabel("Context length (prompt tokens; + generated tokens)")
    ax.set_ylabel("Stored KV cache (MiB, measured .nbytes)")
    ax.set_title("KV cache memory vs context length")
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "fig4_kv_memory_vs_context")


def fig_passkey(tag):
    rows = read_csv(os.path.join(RES, f"passkey_{tag}.csv"))
    order = []
    for r in rows:
        if r["variant"] not in order:
            order.append(r["variant"])
    z = 1.959964
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(order) + 1.0))
    for i, v in enumerate(order):
        rs = [r for r in rows if r["variant"] == v and not r["error"]]
        n = len(rs)
        if n == 0:
            continue
        k = sum(int(r["correct"]) for r in rs)
        p = k / n
        d = 1 + z * z / n
        c0 = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        fam = family(v, rs[0]["method"])
        ax.barh(i, p, 0.62, color=COL[fam], zorder=3)
        ax.errorbar(p, i, xerr=[[p - max(0, c0 - h)], [min(1, c0 + h) - p]], fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=0, zorder=4)
        ax.annotate(f"{k}/{n}", (1.0, i), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK2, annotation_clip=False)
    ax.set_yticks(range(len(order)), [short(v) for v in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Passkey retrieval accuracy (Wilson 95% CI)")
    ax.set_title(f"Needle retrieval through each KV cache: {tag}")
    ax.grid(axis="y", visible=False)
    save(fig, f"fig5_passkey_{tag}")


def fig_importance(tag):
    rows = read_csv(os.path.join(RES, f"importance_{tag}.csv"))
    p8s = sorted({fnum(r["p8"]) for r in rows})
    pol = [("ema", "EMA attention importance (adaptive)", COL["adaptive"]),
           ("recency", "Recency (newest tokens)", COL["recency"]),
           ("oracle", "Oracle (actual future attention)", COL["oracle"])]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    w = 0.26
    for j, (k, lab, c) in enumerate(pol):
        for i, p8 in enumerate(p8s):
            per_w = {}
            for r in rows:
                if fnum(r["p8"]) == p8 and not math.isnan(fnum(r[k])):
                    per_w.setdefault(r["window"], []).append(fnum(r[k]))
            v = np.mean([np.mean(x) for x in per_w.values()])
            ax.bar(i + (j - 1) * (w + 0.02), v, w, color=c, label=lab if i == 0 else None, zorder=3)
            ax.annotate(f"{v:.2f}", (i + (j - 1) * (w + 0.02), v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(p8s)), [f"keep {p:.0%} at INT8" for p in p8s])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of next-16-step attention\non the kept (unpinned) tokens")
    ax.set_title(f"Does importance predict future attention? {tag}")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(axis="x", visible=False)
    save(fig, f"fig6_importance_recall_{tag}")


def main():
    print(f"writing figures to {OUT}")
    dtags = model_tags("decision")
    dtags = sorted({t for t in dtags if not t.endswith("_windows")})
    for t in dtags:
        if os.path.exists(os.path.join(RES, f"decision_{t}_summary.csv")):
            fig_pareto(t)
            fig_bitmatched(t)
    fig_key_ablation([t for t in dtags if os.path.exists(os.path.join(RES, f"decision_{t}_summary.csv"))])
    fig_memory()
    for t in model_tags("passkey"):
        fig_passkey(t)
    for t in model_tags("importance"):
        fig_importance(t)


if __name__ == "__main__":
    main()
