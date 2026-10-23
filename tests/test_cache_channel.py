"""
tests/test_cache_channel.py
---------------------------
ChannelKVCache: per-channel key quantization with per-token tiers.
"""

import numpy as np
import pytest
import torch

from adaptivekv.cache_channel import (ChannelKVCache, FP16_TIER, quantize_keys_per_channel,
                                      measure_kv_bytes_channel, compute_effective_bits_channel)
from adaptivekv.cache import AdaptiveKVCache
from adaptivekv.quant import dequantize

H, D, GS, G = 2, 64, 32, 8


def _kv(n, seed=0, outlier=False):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, H, n, D, generator=g)
    if outlier:
        k[..., 7] = k[..., 7] * 40 + 60          # large, mostly constant channel (like Qwen keys)
    v = torch.randn(1, H, n, D, generator=g)
    return k.half(), v.half()                   # real model K/V are fp16


def _fill(c, n_layers, n, **kw):
    ks, vs = [], []
    for li in range(n_layers):
        k, v = _kv(n, seed=li, **kw)
        c.append_tokens(li, k, v)
        ks.append(k); vs.append(v)
    return ks, vs


def _read(c, li=0):
    K, V = c.get_kv_fp16(li, torch.device("cpu"))
    return K.float(), V.float()


def test_residual_is_exact_before_commit():
    c = ChannelKVCache(1, GS, key_group_tokens=G)
    ks, vs = _fill(c, 1, 20)
    K, V = _read(c)
    assert torch.equal(K, ks[0].float()) and torch.equal(V, vs[0].float())
    assert c.tier_counts() == {8: 0, 4: 0, FP16_TIER: 20}


def test_commit_order_preserved_and_int8_accurate():
    c = ChannelKVCache(1, GS, key_group_tokens=G, tiers=(8, 4))
    ks, _ = _fill(c, 1, 24)
    new = np.array([8, 4] * 8 + [FP16_TIER] * 8)
    c.set_tiers(new)
    c.check_consistency()
    K, _ = _read(c)
    err = (K - ks[0].float()).abs().amax(dim=(0, 1, 3))
    assert err[new == 8].max() < 0.05                      # INT8 rows in the right place
    assert err[new == FP16_TIER].max() == 0                # residual stored exactly
    assert err[new == 4].max() > err[new == 8].max()


def test_per_channel_beats_per_token_on_outlier_channel():
    n = 32
    k, v = _kv(n, outlier=True)
    ch = ChannelKVCache(1, GS, key_group_tokens=n, tiers=(4,))
    ch.append_tokens(0, k, v)
    ch.set_tiers(np.full(n, 4))
    pt = AdaptiveKVCache(1, GS, k_channel_norm=False, tiers=(4,))
    pt.append_tokens(0, k, v, np.full(n, 4))
    kf = k.float()
    e_ch = ((_read(ch)[0] - kf).norm() / kf.norm()).item()
    Kp, _ = pt.get_kv_fp16(0, torch.device("cpu"))
    e_pt = ((Kp.float() - kf).norm() / kf.norm()).item()
    assert e_ch < 0.5 * e_pt


def test_quantize_keys_per_channel_shapes_and_roundtrip():
    k = torch.randn(10, H, D)
    for bits in (8, 4, 2):
        p, s, z = quantize_keys_per_channel(k, bits)
        assert p.shape == (10, H, D * bits // 8) and s.shape == (H, D) and z.shape == (H, D)
    p, s, z = quantize_keys_per_channel(k, 8)
    r = p.to(torch.int32).float() * s.float() + z.float()
    assert (r - k).abs().max() < 0.05


def test_demotion_leaves_other_blocks_byte_identical():
    c = ChannelKVCache(1, GS, key_group_tokens=G, tiers=(8, 4, 2))
    _fill(c, 1, 32)
    t0 = np.full(32, 8)
    c.set_tiers(t0)
    K0, V0 = _read(c)
    t1 = t0.copy()
    t1[3] = 4                                           # block 0 changes; blocks 1..3 must not
    t1[20] = 2                                          # block 2 changes
    c.set_tiers(t1)
    c.check_consistency()
    K1, V1 = _read(c)
    untouched = [i for i in range(32) if i // G not in (0, 2)]
    assert torch.equal(K1[:, :, untouched], K0[:, :, untouched])
    assert torch.equal(V1[:, :, untouched], V0[:, :, untouched])
    stayed_vals = [i for i in range(32) if i not in (3, 20)]
    assert torch.equal(V1[:, :, stayed_vals], V0[:, :, stayed_vals])   # values are per token


def test_promotion_rejected_and_bad_tier_rejected():
    c = ChannelKVCache(1, GS, key_group_tokens=G, tiers=(8, 4))
    _fill(c, 1, 8)
    c.set_tiers(np.full(8, 4))
    with pytest.raises(ValueError, match="promotion"):
        c.set_tiers(np.full(8, 8))
    with pytest.raises(ValueError):
        c.set_tiers(np.full(8, 2))


def test_memory_accounting_by_hand():
    c = ChannelKVCache(2, GS, key_group_tokens=G, tiers=(8, 4))
    _fill(c, 2, 20)
    t = np.array([8] * 8 + [4] * 8 + [FP16_TIER] * 4)
    c.set_tiers(t)
    m = measure_kv_bytes_channel(c)
    L = 2
    # INT8: 8 tokens, K and V codes 64 B per head -> 8*2*64*2 per layer
    assert m["int8_data_bytes"] == L * 8 * H * 64 * 2
    assert m["int4_data_bytes"] == L * 8 * H * 32 * 2
    # key scales: 1 subset per tier here -> (H, D) scale + zp fp16; values: per token (H, D/GS) scale + zp
    key_sc = H * D * 2 * 2
    val_sc8 = 8 * H * (D // GS) * 2 * 2
    assert m["int8_scale_bytes"] == L * (key_sc + val_sc8)
    assert m["fp16_residual_bytes"] == L * 4 * H * D * 2 * 2
    assert m["index_bytes"] == 20
    assert m["total_bytes"] == (m["int8_data_bytes"] + m["int4_data_bytes"] + m["int8_scale_bytes"]
                                + m["int4_scale_bytes"] + m["fp16_residual_bytes"] + 20)
    assert m["fp16_equivalent"] == L * 20 * H * D * 2 * 2
    assert abs(compute_effective_bits_channel(c) - m["total_bytes"] * 8 / m["n_original_values"]) < 1e-12


def test_layers_stay_consistent():
    c = ChannelKVCache(3, GS, key_group_tokens=G, tiers=(8, 4, 2))
    _fill(c, 3, 16)
    c.set_tiers(np.array([8] * 4 + [4] * 4 + [2] * 8))
    c.set_tiers(np.array([8] * 2 + [4] * 6 + [2] * 8))
    c.check_consistency()
    assert c.tier_counts() == {8: 2, 4: 6, 2: 8, FP16_TIER: 0}


def test_pending_keys_before_block_is_sealed():
    c = ChannelKVCache(1, GS, key_group_tokens=G, tiers=(8, 4))
    ks, vs = _fill(c, 1, 20)
    t = np.array([8] * 8 + [4] * 4 + [FP16_TIER] * 8)       # block 0 sealed, block 1 only partly tiered
    c.set_tiers(t)
    c.check_consistency()
    K, V = _read(c)
    kf, vf = ks[0].float(), vs[0].float()
    assert torch.equal(K[:, :, 8:12], kf[:, :, 8:12])        # pending keys stay exact FP16
    assert not torch.equal(V[:, :, 8:12], vf[:, :, 8:12])    # their values are already quantized (INT4)
    m = measure_kv_bytes_channel(c)
    assert m["fp16_pending_key_bytes"] == 4 * H * D * 2
    assert m["fp16_residual_bytes"] == 8 * H * D * 2 * 2
    # sealing block 1 quantizes its keys per channel and empties the pending buffer
    t2 = np.array([8] * 8 + [4] * 8 + [FP16_TIER] * 4)
    c.set_tiers(t2)
    c.check_consistency()
    assert measure_kv_bytes_channel(c)["fp16_pending_key_bytes"] == 0
    K2, V2 = _read(c)
    assert torch.equal(V2[:, :, 8:12], V[:, :, 8:12])        # values not re-quantized on sealing


def test_demotion_inside_unsealed_block_keeps_key_exact():
    c = ChannelKVCache(1, GS, key_group_tokens=G, tiers=(8, 4))
    ks, _ = _fill(c, 1, 12)
    c.set_tiers(np.array([8] * 10 + [FP16_TIER] * 2))
    c.set_tiers(np.array([8] * 8 + [4, 8] + [FP16_TIER] * 2))   # demote position 8 (block 1 unsealed)
    c.check_consistency()
    K, _ = _read(c)
    assert torch.equal(K[:, :, 8], ks[0].float()[:, :, 8])
