"""
tests/test_quant.py
-------------------
Unit tests for adaptivekv/quant.py.

Hand calculations are shown in comments so they can be verified independently.
"""

import pytest
import torch
import numpy as np
from adaptivekv.quant import quantize, dequantize, _pack_int4, _unpack_int4


# ---------------------------------------------------------------------------
# T1.3 — INT4 packing is lossless on integer codes
# ---------------------------------------------------------------------------

def test_int4_packing_lossless():
    """
    Hand check:
      input codes = [0, 15, 7, 8] (4-bit values in [0,15])
      byte 0 = (0 & 0xF) | ((15 & 0xF) << 4) = 0 | 0xF0 = 0xF0 = 240
      byte 1 = (7 & 0xF) | ((8  & 0xF) << 4) = 7 | 0x80 = 0x87 = 135
      unpack byte 0: lo = 240 & 0xF = 0, hi = (240 >> 4) & 0xF = 15
      unpack byte 1: lo = 135 & 0xF = 7, hi = (135 >> 4) & 0xF = 8
      → [0, 15, 7, 8]  ✓
    """
    codes = torch.tensor([0, 15, 7, 8, 1, 14, 3, 12], dtype=torch.int32)
    packed = _pack_int4(codes)
    restored = _unpack_int4(packed, original_last_dim=8)
    assert torch.equal(codes, restored), f"INT4 pack/unpack mismatch: {codes} vs {restored}"
    # Check byte values for first two hand-calculated bytes
    # codes[0]=0, codes[1]=15 → byte = 0 | (15<<4) = 240
    assert int(packed[0].item()) == 240, f"Expected 240, got {packed[0].item()}"
    # codes[2]=7, codes[3]=8 → byte = 7 | (8<<4) = 7 | 128 = 135
    assert int(packed[1].item()) == 135, f"Expected 199, got {packed[1].item()}"


# ---------------------------------------------------------------------------
# T1.5 — INT8 quantize/dequantize round-trip
# ---------------------------------------------------------------------------

def test_roundtrip_int8():
    """
    For a random FP16 tensor, max quantization error ≤ scale / 2.

    scale = (max - min) / 255.
    For a random uniform [0,1] tensor of 64 values:
      expected max error = scale / 2 ≈ (1 / 255) / 2 ≈ 0.002.

    We verify: abs(x - dequantize(quantize(x))) ≤ scale / 2  for all elements.
    """
    torch.manual_seed(0)
    x = torch.rand(2, 64, dtype=torch.float16)
    group_size = 64

    packed, scale, zero_point = quantize(x, bits=8, group_size=group_size)

    # Verify packed dtype
    assert packed.dtype == torch.uint8, f"Expected uint8, got {packed.dtype}"
    # INT8 packed shape == original shape
    assert packed.shape == x.shape

    x_dq = dequantize(packed, scale, zero_point, bits=8, group_size=group_size)

    # Max error ≤ scale/2 per element
    # scale shape: (2, 1) for input (2, 64) with group_size=64, then reshape gives (2, 1, 1)
    # Expand scale to match x shape for comparison
    max_scale = scale.float().max().item()
    max_err = (x.float() - x_dq.float()).abs().max().item()
    allowed = max_scale / 2
    # Tolerance: FP16 scale is rounded, so effective bound has FP16 rounding error (~1e-4).
    assert max_err <= allowed + 1e-4, \
        f"INT8 max error {max_err:.6f} > allowed {allowed:.6f} + 1e-4"


# ---------------------------------------------------------------------------
# T1.6 — INT4 quantize/dequantize round-trip
# ---------------------------------------------------------------------------

def test_roundtrip_int4():
    """
    For INT4 (15 levels): scale = (max-min)/15.
    Max error ≤ scale/2 per element.
    """
    torch.manual_seed(1)
    x = torch.rand(2, 64, dtype=torch.float16)
    group_size = 64

    packed, scale, zero_point = quantize(x, bits=4, group_size=group_size)

    assert packed.dtype == torch.uint8
    # INT4: packed last dim = original // 2
    assert packed.shape[-1] == x.shape[-1] // 2

    x_dq = dequantize(packed, scale, zero_point, bits=4, group_size=group_size)
    assert x_dq.shape == x.shape

    max_scale = scale.float().max().item()
    max_err = (x.float() - x_dq.float()).abs().max().item()
    allowed = max_scale / 2
    assert max_err <= allowed + 1e-4, \
        f"INT4 max error {max_err:.6f} > allowed {allowed:.6f}"


# ---------------------------------------------------------------------------
# Additional: multi-dim and group_size < head_dim
# ---------------------------------------------------------------------------

def test_quantize_multidim_shape():
    """Shape contract: packed dim scales by 1/bits*8, scale has n_groups groups."""
    x = torch.randn(3, 4, 64).half()
    group_size = 32  # 2 groups per last dim
    for bits in (8, 4):
        packed, scale, zp = quantize(x, bits, group_size)
        expected_packed_last = 64 // (8 // bits)
        assert packed.shape == (3, 4, expected_packed_last), \
            f"bits={bits}: expected packed shape (3,4,{expected_packed_last}), got {packed.shape}"
        assert scale.shape == (3, 4, 2), \
            f"bits={bits}: expected scale shape (3,4,2), got {scale.shape}"


def test_dequantize_restores_shape():
    """dequantize output shape equals original input shape."""
    x = torch.randn(1, 2, 64).half()
    for bits in (8, 4):
        packed, scale, zp = quantize(x, bits, group_size=64)
        x_dq = dequantize(packed, scale, zp, bits, group_size=64)
        assert x_dq.shape == x.shape, \
            f"bits={bits}: expected {x.shape}, got {x_dq.shape}"
        assert x_dq.dtype == torch.float16
