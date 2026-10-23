"""
Run:  pytest tests/test_core.py -v
Every expected value is hand-checkable (see comments).
"""
import numpy as np
import pytest
import torch

from adaptivekv.quant import quantize, dequantize, _pack_int4, _unpack_int4
from adaptivekv.allocator import TierAllocator
from adaptivekv.importance import EMATracker
from adaptivekv.cache import AdaptiveKVCache, measure_kv_bytes, compute_effective_bits

GS = 32


@pytest.mark.parametrize("bits", [8, 4])
def test_roundtrip_error_bounded_by_half_scale(bits):
    torch.manual_seed(0)
    x = torch.randn(5, 2, 64)
    p, s, z = quantize(x, bits, GS)
    xr = dequantize(p, s, z, bits, GS).float()
    # per-element error <= scale/2 (+ tiny fp16 slack)
    sc = s.float().repeat_interleave(GS, dim=-1)
    assert ((x - xr).abs() <= sc * 0.5 * 1.01 + 5e-3).all()


def test_packing_lossless():
    c4 = torch.randint(0, 16, (3, 2, 64), dtype=torch.int32)
    assert torch.equal(_unpack_int4(_pack_int4(c4), 64), c4)
    c2 = torch.randint(0, 4, (3, 2, 64), dtype=torch.int32)


def test_packed_sizes():
    x = torch.randn(4, 2, 64)
    assert quantize(x, 8, GS)[0].shape[-1] == 64   # 1 byte / value
    assert quantize(x, 4, GS)[0].shape[-1] == 32   # 2 values / byte


def test_flat_group_no_nan():
    x = torch.full((1, 1, 32), 3.0)               # zero range: old code produced NaN
    p, s, z = quantize(x, 4, GS)
    xr = dequantize(p, s, z, 4, GS).float()
    assert torch.isfinite(xr).all() and (xr - 3.0).abs().max() < 1e-2


def test_ema_hand_computed():
    # alpha=0.5: step1 [1,1] ; step2 scores [0,0,4]
    # existing: 0.5*0+0.5*1 = 0.5 each ; new position = raw 4
    e = EMATracker(0.5)
    e.update(np.array([1.0, 1.0]))
    e.update(np.array([0.0, 0.0, 4.0]))
    assert np.allclose(e.get(), [0.5, 0.5, 4.0])


def test_allocator_counts():
    # n=10, p8=0.2 -> 2 INT8 ; all remaining tokens are INT4
    t = TierAllocator(0.2, 0.8).assign(np.arange(10, dtype=float))
    assert (t == 8).sum() == 2 and (t == 4).sum() == 8
    assert t[9] == 8 and t[8] == 8                  # highest importance -> INT8


def test_allocator_requires_two_tier_fractions_to_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        TierAllocator(0.5, 0.0)


def test_allocator_pins():
    t = TierAllocator(0.1, 0.9, sink_tokens=2, recent_tokens=3).assign(np.zeros(20))
    assert (t[:2] == 8).all() and (t[-3:] == 8).all()


def _fill(cache, n_layers, n, H=2, D=64, tiers=None, outlier=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    ks, vs = [], []
    for li in range(n_layers):
        k = torch.randn(1, H, n, D, generator=g)
        if outlier:
            k[..., 5] *= 50.0                      # one huge outlier channel (like Qwen keys)
        v = torch.randn(1, H, n, D, generator=g)
        cache.append_tokens(li, k, v, tiers if tiers is not None else np.full(n, 8))
        ks.append(k); vs.append(v)
    return ks, vs


def test_memory_accounting_equals_nbytes_and_bits_by_hand():
    c = AdaptiveKVCache(2, GS, k_channel_norm=False)
    _fill(c, 2, 10, tiers=np.full(10, 4))
    m = measure_kv_bytes(c)
    # H=2, D=64, gs=32 -> 2 groups/head, INT4 packed dim = 32
    # data  per token per layer: (K+V) * 2 heads * 32 B            = 128 B -> 10 tok * 2 layers = 2560
    # scale per token per layer: (K,V)*(scale,zp) * 2 heads*2 grp*2B = 32 B  -> 10 tok * 2 layers =  640
    # index = 10 B (1 byte/token, counted once) ; no aux (k_channel_norm=False)
    assert m["int4_data_bytes"] == 2560 and m["int4_scale_bytes"] == 640
    assert m["total_bytes"] == 2560 + 640 + 10
    assert m["fp16_equivalent"] == 2 * 10 * 2 * 64 * 2 * 2          # layers*tok*H*D*(K+V)*2B
    assert abs(compute_effective_bits(c) - m["total_bytes"] * 8 / (2 * 10 * 2 * 64 * 2)) < 1e-9


def test_all_int8_close_to_original():
    c = AdaptiveKVCache(1, GS)
    ks, vs = _fill(c, 1, 16)
    K, V = c.get_kv_fp16(0, torch.device("cpu"))
    assert (K.float() - ks[0]).abs().max() < 0.05 and (V.float() - vs[0]).abs().max() < 0.05


def test_token_order_preserved_with_mixed_tiers_and_reassign():
    c = AdaptiveKVCache(1, GS, k_channel_norm=False)
    n = 12
    ks, _ = _fill(c, 1, n, tiers=np.full(n, 8))
    new = np.array([8, 8, 4, 4, 4, 8, 8, 4, 8, 8, 4, 8])
    assert c.reassign_tiers(new) == int((new != 8).sum())
    c.check_consistency()
    assert c.tier_counts() == {8: int((new == 8).sum()), 4: int((new == 4).sum())}
    K, _ = c.get_kv_fp16(0, torch.device("cpu"))
    err = (K.float() - ks[0]).abs().amax(dim=(0, 1, 3))        # per token
    # INT8 tokens must stay accurate => order was not scrambled
    assert err[new == 8].max() < 0.05


def test_channel_norm_helps_int4_keys_with_outlier_channel():
    n = 64
    errs = {}
    for flag in (False, True):
        c = AdaptiveKVCache(1, GS, k_channel_norm=flag)
        ks, _ = _fill(c, 1, n, tiers=np.full(n, 4), outlier=True)
        K, _ = c.get_kv_fp16(0, torch.device("cpu"))
        normal = [i for i in range(64) if i != 5]
        errs[flag] = (K.float() - ks[0])[..., normal].abs().mean().item()
    assert errs[True] < errs[False]            # normal channels far more accurate with norm
    print("mean abs err on normal channels:", errs)