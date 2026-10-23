"""
tests/test_allocator.py
-----------------------
Unit tests for TierAllocator.

Hand calculations shown in comments.
"""

import pytest
import math
import numpy as np
from adaptivekv.allocator import TierAllocator, INT8, INT4


# ---------------------------------------------------------------------------
# T3.2 — Basic tier counts
# ---------------------------------------------------------------------------

def test_tier_counts_basic():
    """
    p8=0.2, p4=0.8, n=10
    n8 = ceil(0.2*10) = ceil(2.0) = 2
    n4 = 10 - 2 = 8
    """
    alloc = TierAllocator(p8=0.2, p4=0.8)
    importance = np.arange(10, dtype=float)  # 0..9, higher = more important
    tiers = alloc.assign(importance)

    assert tiers.shape == (10,)
    n8 = (tiers == INT8).sum()
    n4 = (tiers == INT4).sum()
    assert n8 == 2, f"Expected 2 INT8, got {n8}"
    assert n4 == 8, f"Expected 8 INT4, got {n4}"


def test_tier_top_tokens_get_int8():
    """The 2 highest-importance tokens must be assigned INT8."""
    alloc = TierAllocator(p8=0.2, p4=0.8)
    importance = np.array([0.1, 0.9, 0.5, 0.3, 0.8, 0.2, 0.4, 0.7, 0.6, 0.0])
    tiers = alloc.assign(importance)
    # Positions 1 (0.9) and 4 (0.8) should be INT8
    assert tiers[1] == INT8, f"pos 1 (imp=0.9) should be INT8, got {tiers[1]}"
    assert tiers[4] == INT8, f"pos 4 (imp=0.8) should be INT8, got {tiers[4]}"


# ---------------------------------------------------------------------------
# T3.3 — Small sequence edge cases
# ---------------------------------------------------------------------------

def test_tier_counts_n3():
    """
    p8=0.2, p4=0.8, n=3
    n8 = ceil(0.2*3) = ceil(0.6) = 1
    n4 = 3 - 1 = 2
    """
    alloc = TierAllocator(p8=0.2, p4=0.8)
    importance = np.array([0.3, 0.8, 0.5])
    tiers = alloc.assign(importance)
    assert (tiers == INT8).sum() == 1
    assert (tiers == INT4).sum() == 2


def test_tier_counts_n1():
    """
    p8=0.2, p4=0.8, n=1
    n8 = ceil(0.2) = 1
    Only 1 token, 1 INT8 slot.
    """
    alloc = TierAllocator(p8=0.2, p4=0.8)
    importance = np.array([0.7])
    tiers = alloc.assign(importance)
    assert len(tiers) == 1
    assert tiers[0] == INT8


def test_all_tokens_assigned():
    """Every token gets exactly one tier."""
    alloc = TierAllocator(p8=0.2, p4=0.8)
    for n in [1, 2, 5, 10, 20, 100]:
        importance = np.random.rand(n)
        tiers = alloc.assign(importance)
        valid = {INT8, INT4}
        assert all(t in valid for t in tiers), f"Unknown tier in n={n}"
        assert len(tiers) == n


# ---------------------------------------------------------------------------
# T3.4 — Sink token pinning
# ---------------------------------------------------------------------------

def test_sink_pinning():
    """
    sink_tokens=2: positions 0 and 1 must be INT8 regardless of importance.
    """
    alloc = TierAllocator(p8=0.0, p4=1.0, sink_tokens=2)
    # Give sink tokens lowest importance
    importance = np.array([0.01, 0.02, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    tiers = alloc.assign(importance)
    assert tiers[0] == INT8, f"Sink pos 0 should be INT8, got {tiers[0]}"
    assert tiers[1] == INT8, f"Sink pos 1 should be INT8, got {tiers[1]}"


# ---------------------------------------------------------------------------
# T3.5 — Recent token pinning
# ---------------------------------------------------------------------------

def test_recent_pinning():
    """
    recent_tokens=2: last 2 positions must be INT8 regardless of importance.
    """
    alloc = TierAllocator(p8=0.0, p4=1.0, recent_tokens=2)
    n = 10
    # Give recent tokens lowest importance
    importance = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.01, 0.02])
    tiers = alloc.assign(importance)
    assert tiers[8] == INT8, f"Recent pos 8 should be INT8, got {tiers[8]}"
    assert tiers[9] == INT8, f"Recent pos 9 should be INT8, got {tiers[9]}"


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------

def test_tie_breaking_by_position():
    """
    With equal importance, lower position index wins higher tier.
    """
    alloc = TierAllocator(p8=0.2, p4=0.8)
    n = 10
    importance = np.ones(n)  # all equal
    tiers = alloc.assign(importance)
    # n8=2: positions 0 and 1 should get INT8
    assert tiers[0] == INT8, f"pos 0 should be INT8, got {tiers[0]}"
    assert tiers[1] == INT8, f"pos 1 should be INT8, got {tiers[1]}"
