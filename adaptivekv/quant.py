"""
adaptivekv/quant.py
-------------------
Asymmetric per-group quantization and bit-packing for INT8 / INT4 / INT2.
INT2 (4 values per byte) is only used by the optional INT2 ablation.

Fixes vs previous version:
  * Device-agnostic: works on whatever device the input tensor is on
    (previous version forced everything to CPU).
  * Scale underflow fixed: the old clamp(1e-8) became 0 in FP16 -> division by
    zero -> NaN codes for flat groups. Scale is now clamped to FP16 min-normal.
  * Zero-point rounding fixed: x_min cast to FP16 could round UP above x_min,
    which clipped the smallest values. Zero-point is now rounded DOWN so the
    group minimum is always representable (no clipping).

Leading dimensions are arbitrary: quantize() on (n_tokens, kv_heads, head_dim)
groups only along the last dimension.

Public API (unchanged):
  quantize(x, bits, group_size)  -> (packed, scale, zero_point)
  dequantize(packed, scale, zero_point, bits, group_size) -> FP16 tensor
"""

import torch
from torch import Tensor
from typing import Tuple

FP16_MIN_NORMAL = 6.1035e-05   # smallest normal FP16; avoids scale == 0 after cast


def _check_group_size(last_dim: int, group_size: int) -> None:
    if last_dim % group_size != 0:
        raise ValueError(
            f"Last dimension {last_dim} must be divisible by group_size {group_size}")


def _check_bits(bits: int) -> None:
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4 or 8; got {bits}")


# ---------------------------------------------------------------------------
# Codes <-> floats
# ---------------------------------------------------------------------------

def _quantize_to_codes(x: Tensor, bits: int, group_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """x -> (int32 codes in [0, 2^bits-1], fp16 scale, fp16 zero_point)."""
    _check_bits(bits)
    _check_group_size(x.shape[-1], group_size)

    x_fp = x.float()
    orig_shape = x_fp.shape
    n_groups = orig_shape[-1] // group_size
    xg = x_fp.reshape(*orig_shape[:-1], n_groups, group_size)

    x_min = xg.amin(dim=-1)
    x_max = xg.amax(dim=-1)
    max_val = float(2 ** bits - 1)

    # Zero-point in FP16, rounded DOWN so it never exceeds x_min.
    zp16 = x_min.to(torch.float16)
    too_high = zp16.float() > x_min
    shifted = (x_min - x_min.abs() * (2.0 ** -10)).to(torch.float16)
    zp16 = torch.where(too_high, shifted, zp16)
    zp = zp16.float()

    # Scale computed from the stored zero-point, clamped away from zero.
    range_val = (x_max - zp).clamp(min=0.0)
    scale16 = (range_val / max_val).clamp(min=FP16_MIN_NORMAL).to(torch.float16)
    scale = scale16.float()

    codes = ((xg - zp.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp(0, max_val)
    codes = codes.to(torch.int32).reshape(orig_shape)
    return codes, scale16, zp16


def _dequantize_from_codes(codes: Tensor, scale: Tensor, zero_point: Tensor,
                           bits: int, group_size: int) -> Tensor:
    _check_bits(bits)
    orig_shape = codes.shape
    n_groups = orig_shape[-1] // group_size
    cg = codes.float().reshape(*orig_shape[:-1], n_groups, group_size)
    x = cg * scale.float().unsqueeze(-1) + zero_point.float().unsqueeze(-1)
    return x.reshape(orig_shape).to(torch.float16)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def _pack_int8(codes: Tensor) -> Tensor:
    return codes.to(torch.uint8)


def _unpack_int8(packed: Tensor) -> Tensor:
    return packed.to(torch.int32)


def _pack_int4(codes: Tensor) -> Tensor:
    """2 values per byte: low nibble = even index, high nibble = odd index."""
    if codes.shape[-1] % 2 != 0:
        raise ValueError("INT4 packing requires even last dimension")
    lo = (codes[..., 0::2] & 0xF).to(torch.uint8)
    hi = (codes[..., 1::2] & 0xF).to(torch.uint8)
    return lo | (hi << 4)


def _unpack_int4(packed: Tensor, original_last_dim: int) -> Tensor:
    lo = (packed & 0xF).to(torch.int32)
    hi = ((packed >> 4) & 0xF).to(torch.int32)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], original_last_dim)


def _pack_int2(codes: Tensor) -> Tensor:
    """4 values per byte: index 4j+i goes to bits [2i, 2i+1] of byte j."""
    if codes.shape[-1] % 4 != 0:
        raise ValueError("INT2 packing requires last dimension divisible by 4")
    out = torch.zeros(*codes.shape[:-1], codes.shape[-1] // 4, dtype=torch.uint8, device=codes.device)
    for i in range(4):
        out |= (codes[..., i::4] & 0x3).to(torch.uint8) << (2 * i)
    return out


def _unpack_int2(packed: Tensor, original_last_dim: int) -> Tensor:
    parts = [((packed >> (2 * i)) & 0x3).to(torch.int32) for i in range(4)]
    return torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], original_last_dim)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def packed_last_dim(last_dim: int, bits: int) -> int:
    """Last-dim size of the packed uint8 tensor."""
    _check_bits(bits)
    return last_dim * bits // 8


def quantize(x: Tensor, bits: int, group_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Quantize to packed uint8 storage on the SAME device as x.

    Returns:
        packed: uint8, last dim = x.shape[-1] * bits / 8
        scale, zero_point: FP16, shape (..., n_groups)
    """
    x = x.detach()
    codes, scale, zp = _quantize_to_codes(x, bits, group_size)
    if bits == 8:
        packed = _pack_int8(codes)
    elif bits == 4:
        packed = _pack_int4(codes)
    else:
        packed = _pack_int2(codes)
    return packed, scale, zp


def dequantize(packed: Tensor, scale: Tensor, zero_point: Tensor,
               bits: int, group_size: int) -> Tensor:
    """Inverse of quantize(); returns FP16 with the original shape."""
    _check_bits(bits)
    if bits == 8:
        codes = _unpack_int8(packed)
    elif bits == 4:
        codes = _unpack_int4(packed, packed.shape[-1] * 2)
    else:
        codes = _unpack_int2(packed, packed.shape[-1] * 4)
    return _dequantize_from_codes(codes, scale, zero_point, bits, group_size)