"""
tests/test_paper_metrics.py
---------------------------
Metric code used for the paper: per-step KL / top-1 scoring, Pareto frontier,
bootstrap CI, Wilson interval, and pinned-token attention recall.
"""

import math

import numpy as np
import pytest
import torch

from adaptivekv.generate import _StepScorer
from decision_test import pareto_front, boot_ci, make_windows
from passkey import wilson
from importance_analysis import recall_at, ema_series


# ---- per-step scoring --------------------------------------------------------

def test_scorer_kl_zero_and_top1_when_identical():
    logits = torch.randn(3, 50)
    ref = torch.log_softmax(logits, -1)
    sc = _StepScorer(ref, collect=False)
    for i in range(3):
        sc(i, logits[i], tok_id=0)
    r = sc.result()
    assert max(r["kl"]) < 1e-6 and all(r["top1"])
    assert r["nll"] == pytest.approx([-ref[i, 0].item() for i in range(3)], abs=1e-5)


def test_scorer_kl_matches_definition_and_is_positive():
    torch.manual_seed(0)
    ref = torch.log_softmax(torch.randn(1, 20), -1)
    logits = torch.randn(20)
    sc = _StepScorer(ref, collect=True)
    sc(0, logits, tok_id=3)
    lp = torch.log_softmax(logits, -1)
    expected = float((ref[0].exp() * (ref[0] - lp)).sum())
    r = sc.result()
    assert r["kl"][0] == pytest.approx(expected, rel=1e-5) and r["kl"][0] > 0
    assert r["top1"][0] == (int(lp.argmax()) == int(ref[0].argmax()))
    assert r["logprobs"].shape == (1, 20)


# ---- statistics ----------------------------------------------------------------

def test_pareto_front():
    pts = {"a": (4.0, 1.0), "b": (8.0, 0.1), "c": (8.0, 0.5), "d": (5.0, 1.0)}
    assert pareto_front(pts) == {"a", "b"}          # c dominated by b, d by a


def test_boot_ci_contains_mean_and_degenerates_for_constant():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 1, 50)
    lo, hi = boot_ci(x, np.mean, 1000, rng)
    assert lo < x.mean() < hi
    assert boot_ci(np.full(10, 2.0), np.mean, 200, rng) == (2.0, 2.0)


def test_wilson_interval():
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and 0.65 < lo < 0.75
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi


def test_make_windows_non_overlapping():
    w = make_windows(list(range(1000)), 5, 100)
    assert len(w) == 5 and all(len(x) == 100 for x in w)
    starts = [x[0] for x in w]
    assert all(b - a >= 100 for a, b in zip(starts, starts[1:]))


# ---- importance recall -------------------------------------------------------

def _trace(n0=10, steps=6, hot=5):
    """Attention trace where position `hot` always receives most of the mass,
    and position 0 (sink) receives a lot too."""
    tr = []
    for s in range(steps + 1):
        n = n0 + s
        v = np.full(n, 0.01)
        v[0] = 0.5
        v[hot] = 0.3
        tr.append(v / v.sum())
    return tr


def test_recall_excludes_pinned_sink():
    tr = _trace()
    ema = ema_series(tr, 0.1)
    r = recall_at(tr, ema, 0, 3, p8=0.3, sink=1, recent=2)
    # 3 kept, 3 pinned -> k_free = 0 -> undefined, reported as nan
    assert r["k_free"] == 0 and math.isnan(r["ema"])
    r = recall_at(tr, ema, 0, 3, p8=0.5, sink=1, recent=2)
    assert r["k_free"] == 2
    assert r["ema"] == pytest.approx(r["oracle"])       # EMA finds the hot token
    assert r["ema"] > r["recency"]                      # recency keeps only newest unpinned


def test_recall_without_pins_is_inflated_by_sink():
    tr = _trace()
    ema = ema_series(tr, 0.1)
    unpinned = recall_at(tr, ema, 0, 3, p8=0.2, sink=1, recent=0)
    raw = recall_at(tr, ema, 0, 3, p8=0.2, sink=0, recent=0)
    assert raw["ema"] > unpinned["ema"]


# ---- hardware safety guard --------------------------------------------------

def test_guard_pauses_until_cool_and_stops_on_low_ram(monkeypatch):
    import adaptivekv.safety as sf
    temps = iter([82, 78, 71, 69])
    monkeypatch.setattr(sf, "gpu_temp_c", lambda: next(temps))
    monkeypatch.setattr(sf, "free_ram_mb", lambda: 8000)
    monkeypatch.setattr(sf.time, "sleep", lambda s: None)
    g = sf.Guard({"safety": {"pause_temp_c": 80, "resume_temp_c": 70}})
    g()
    assert g.pauses == 1 and g.max_seen == 82
    monkeypatch.setattr(sf, "free_ram_mb", lambda: 500)
    with pytest.raises(sf.SafetyStop):
        g()
