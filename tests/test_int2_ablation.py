"""
tests/test_int2_ablation.py
---------------------------
INT2 ablation (int2_enabled: true): packing, three-tier allocation quotas,
three-tier cache storage, and the recency baseline with an INT2 tier.
The default two-tier behaviour must be unchanged when INT2 is off.
"""

import numpy as np
import pytest
import torch

from adaptivekv.quant import quantize, dequantize, packed_last_dim
from adaptivekv.allocator import TierAllocator
from adaptivekv.cache import AdaptiveKVCache, measure_kv_bytes, compute_effective_bits
from adaptivekv.generate import _tiers_recency, _tier_fractions

GS = 32


# ---- quantizer -------------------------------------------------------------

def test_int2_pack_roundtrip_exact_codes():
    # values already on the 4-level grid of each group -> must come back exactly
    g = torch.Generator().manual_seed(0)
    levels = torch.randint(0, 4, (3, 2, 64), generator=g).float()
    levels[..., 0] = 0.0
    levels[..., 1] = 3.0                       # force min 0 / max 3 in every group
    levels[..., 32] = 0.0
    levels[..., 33] = 3.0
    p, s, z = quantize(levels, 2, GS)
    assert p.dtype == torch.uint8 and p.shape == (3, 2, 16)
    assert torch.equal(dequantize(p, s, z, 2, GS).float(), levels)


def test_int2_error_bounded_by_half_scale():
    x = torch.randn(4, 2, 64)
    p, s, z = quantize(x, 2, GS)
    xr = dequantize(p, s, z, 2, GS).float()
    half = s.float().repeat_interleave(GS, dim=-1) / 2
    assert ((xr - x).abs() <= half + 1e-3).all()


def test_int2_packed_dim():
    assert packed_last_dim(64, 2) == 16


# ---- allocator --------------------------------------------------------------

def test_three_tier_targets_and_assign():
    a = TierAllocator(0.2, 0.3, p2=0.5)
    assert a.targets3(10) == (2, 3, 5)
    t = a.assign(np.arange(10, dtype=float))
    assert (t == 8).sum() == 2 and (t == 4).sum() == 3 and (t == 2).sum() == 5
    assert (t[8:] == 8).all() and (t[5:8] == 4).all() and (t[:5] == 2).all()


def test_two_tier_targets_unchanged_when_p2_zero():
    a = TierAllocator(0.3, 0.7)
    for n in range(1, 50):
        n8, n4, n2 = a.targets3(n)
        assert (n8, n4) == a.targets(n) and n2 == 0


def test_fractions_must_sum_to_one_with_p2():
    with pytest.raises(ValueError, match="sum to 1"):
        TierAllocator(0.2, 0.3, p2=0.2)


def test_three_tier_demote_respects_quotas_over_many_steps():
    rng = np.random.default_rng(0)
    a = TierAllocator(0.3, 0.3, p2=0.4, sink_tokens=4, recent_tokens=8)
    cur = np.full(200, 8, dtype=np.int32)
    for _ in range(100):
        cur = a.demote(cur, rng.random(len(cur)))
        n8, n4, n2 = a.targets3(len(cur))
        assert (cur == 2).sum() <= n2
        assert (cur == 8).sum() >= n8
        assert (cur[:4] == 8).all() and (cur[-8:] == 8).all()   # pins untouched
        cur = np.concatenate([cur, [8]])


def test_demote_never_promotes():
    rng = np.random.default_rng(1)
    a = TierAllocator(0.4, 0.3, p2=0.3)
    cur = a.assign(rng.random(100))
    new = a.demote(cur, rng.random(100))
    assert (new <= cur).all()


# ---- recency baseline -------------------------------------------------------

def test_recency_three_tier_newest_highest():
    t = _tiers_recency(10, 0.2, 0.3, sink=1, p2=0.5)
    assert t.tolist() == [8, 2, 2, 2, 2, 4, 4, 4, 8, 8]


def test_recency_two_tier_unchanged():
    t = _tiers_recency(10, 0.2, 0.8, sink=1)
    assert t.tolist() == [8, 4, 4, 4, 4, 4, 4, 4, 8, 8]


def test_tier_fractions_ignore_p2_when_disabled():
    assert _tier_fractions({"p8": 0.5, "p4": 0.5, "p2": 0.3}) == (0.5, 0.5, 0.0)
    assert _tier_fractions({"p8": 0.2, "p4": 0.3, "p2": 0.5, "int2_enabled": True}) == (0.2, 0.3, 0.5)
    with pytest.raises(ValueError):
        _tier_fractions({"p8": 0.2, "p4": 0.3, "p2": 0.3, "int2_enabled": True})


# ---- cache ------------------------------------------------------------------

def test_three_tier_cache_memory_and_order():
    c = AdaptiveKVCache(1, GS, k_channel_norm=False, tiers=(8, 4, 2))
    n = 12
    k = torch.randn(1, 2, n, 64)
    v = torch.randn(1, 2, n, 64)
    c.append_tokens(0, k, v, np.full(n, 8))
    new = np.array([8, 2, 4, 2, 8, 4, 2, 8, 4, 2, 8, 8])
    c.reassign_tiers(new)
    c.check_consistency()
    assert c.tier_counts() == {8: 5, 4: 3, 2: 4}
    m = measure_kv_bytes(c)
    # per token per layer: data (K+V) * 2 heads * 64*bits/8 B ; scales (K,V)*(s,z) * 2 heads * 2 groups * 2 B
    assert m["int8_data_bytes"] == 5 * 2 * 2 * 64
    assert m["int4_data_bytes"] == 3 * 2 * 2 * 32
    assert m["int2_data_bytes"] == 4 * 2 * 2 * 16
    assert m["int2_scale_bytes"] == 4 * 32
    assert m["total_bytes"] == sum(m[f"int{b}_{x}_bytes"] for b in (8, 4, 2) for x in ("data", "scale")) + n
    K, _ = c.get_kv_fp16(0, torch.device("cpu"))
    err = (K.float() - k[0]).abs().amax(dim=(0, 1, 3))        # per token
    assert err[new == 8].max() < 0.05                        # INT8 rows stayed in place


def test_default_cache_rejects_int2():
    c = AdaptiveKVCache(1, GS)
    with pytest.raises(ValueError):
        c.append_tokens(0, torch.randn(1, 2, 3, 64), torch.randn(1, 2, 3, 64), np.array([8, 4, 2]))


def test_int2_uniform_effective_bits():
    c = AdaptiveKVCache(1, 64, k_channel_norm=False, tiers=(2,))
    c.append_tokens(0, torch.randn(1, 2, 10, 64), torch.randn(1, 2, 10, 64), np.full(10, 2))
    # 2 bits data + (16+16)/64 = 0.5 bits scale/zp + index 8 bits per token / (2*2*64) values
    expected = 2 + 0.5 + 8 * 10 / (10 * 2 * 64 * 2)
    assert abs(compute_effective_bits(c) - expected) < 1e-9
