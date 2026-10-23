"""
adaptivekv/cache_channel.py
---------------------------
ChannelKVCache: mixed-precision KV cache with PER-CHANNEL key quantization
(KIVI-style) and per-token precision tiers (INT8 / INT4 / INT2).

Why: keys of Qwen2.5 have a few large-magnitude channels. Quantizing each key
per token (one scale per token and group) breaks INT4/INT2 (measured: INT4 KL
vs FP16 1.71 on 0.5B and 6.19 = collapse on 3B), while quantizing each key
CHANNEL across a group of tokens is nearly lossless (KL 0.006 / 0.002).
Values are quantized per token (INT4 values alone are nearly lossless).

Layout per layer
  * FP16 residual: the newest tokens (tier code 16, the recent window) keep
    keys and values unquantized.
  * When the engine assigns a token its tier (it has left the recent window),
    its VALUE is quantized at once (per token, groups of `group_size`).
  * Its KEY waits at FP16 ("pending key") until its position block of
    `key_group_tokens` (G) tokens is SEALED, i.e. every position of the block
    exists and has a tier. Then the block's keys are quantized per channel:
    ONE scale + zero-point per (block, kv_head, channel) shared by the tier-b
    tokens of that block (a "subset"). So short sequences still get tiers, and
    at most G-1 keys per layer wait at FP16 (counted in memory).
  * Tier per position: tier_of (1 byte/token, shared by all layers). Block =
    position // G and sealing is derived from tier_of, so no extra index.

set_tiers(new) applies commits (16 -> b) and demotions (b -> c < b) in one call.
Only subsets whose membership changes are re-quantized, from the currently
stored values (lossy, like any re-quantization); every other token's bytes are
untouched. Promotion is rejected.

Memory accounting counts every stored tensor: key/value codes, key scales and
zero-points, value scales and zero-points, the FP16 residual and the tier index.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from .quant import (_quantize_to_codes, _pack_int8, _pack_int4, _pack_int2,
                    _unpack_int8, _unpack_int4, _unpack_int2, quantize, dequantize)

FP16_TIER = 16
QUANT_TIERS = (8, 4, 2)

_PACK = {8: _pack_int8, 4: _pack_int4, 2: _pack_int2}


def _unpack(packed: Tensor, bits: int, D: int) -> Tensor:
    if bits == 8:
        return _unpack_int8(packed)
    if bits == 4:
        return _unpack_int4(packed, D)
    return _unpack_int2(packed, D)


def quantize_keys_per_channel(k: Tensor, bits: int) -> Tuple[Tensor, Tensor, Tensor]:
    """k: (m, H, D) float -> packed (m, H, D*bits/8) uint8, scale/zp (H, D) fp16.
    One scale and zero-point per (head, channel) over the m tokens."""
    x = k.float().permute(1, 2, 0).contiguous()                  # (H, D, m): group = all m tokens
    codes, scale, zp = _quantize_to_codes(x, bits, x.shape[-1])  # codes (H, D, m), scale (H, D, 1)
    codes = codes.permute(2, 0, 1).contiguous()                  # (m, H, D)
    return _PACK[bits](codes), scale[..., 0], zp[..., 0]


class _Layer:
    def __init__(self):
        self.H: Optional[int] = None
        self.D: Optional[int] = None
        self.dtype = torch.float16
        self.device = None
        self.tier_of = np.zeros(0, dtype=np.int8)
        self.res_k: Optional[Tensor] = None      # (r, H, D) model dtype, positions where tier_of == 16
        self.res_v: Optional[Tensor] = None
        self.pend_k: Optional[Tensor] = None     # (q, H, D) model dtype, tiered positions of unsealed blocks
        self.kp: Dict[int, Tensor] = {}          # (n_b, H, D*b/8) uint8, position-sorted
        self.ks: Dict[int, Tensor] = {}          # (S_b, H, D) fp16, one row per block present in tier b
        self.kz: Dict[int, Tensor] = {}
        self.vp: Dict[int, Tensor] = {}          # (n_b, H, D*b/8) uint8
        self.vs: Dict[int, Tensor] = {}          # (n_b, H, D/group) fp16
        self.vz: Dict[int, Tensor] = {}


class ChannelKVCache:
    def __init__(self, n_layers: int, group_size: int, key_group_tokens: int = 32,
                 tiers: Tuple[int, ...] = (8, 4), storage_device=None):
        tiers = tuple(int(b) for b in tiers)
        if not tiers or not set(tiers) <= set(QUANT_TIERS):
            raise ValueError(f"tiers must be a subset of {QUANT_TIERS}, got {tiers}")
        if key_group_tokens < 1:
            raise ValueError("key_group_tokens must be >= 1")
        self.tiers = tiers
        self.n_layers = n_layers
        self.group_size = group_size
        self.G = int(key_group_tokens)
        self.storage_device = torch.device(storage_device) if storage_device else None
        self._layers: List[_Layer] = [_Layer() for _ in range(n_layers)]

    # ---- helpers -------------------------------------------------------------

    def _init_layer(self, st: _Layer, k_t: Tensor, dtype, device) -> None:
        st.H, st.D = k_t.shape[1], k_t.shape[2]
        if st.D % self.group_size != 0:
            raise ValueError(f"head_dim {st.D} not divisible by group_size {self.group_size}")
        if st.D % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for INT2 packing")
        st.dtype, st.device = dtype, device
        st.res_k = torch.empty(0, st.H, st.D, dtype=dtype, device=device)
        st.res_v = torch.empty_like(st.res_k)
        st.pend_k = torch.empty_like(st.res_k)
        Gv = st.D // self.group_size
        for b in self.tiers:
            pd = st.D * b // 8
            st.kp[b] = torch.empty(0, st.H, pd, dtype=torch.uint8, device=device)
            st.ks[b] = torch.empty(0, st.H, st.D, dtype=torch.float16, device=device)
            st.kz[b] = torch.empty_like(st.ks[b])
            st.vp[b] = torch.empty(0, st.H, pd, dtype=torch.uint8, device=device)
            st.vs[b] = torch.empty(0, st.H, Gv, dtype=torch.float16, device=device)
            st.vz[b] = torch.empty_like(st.vs[b])

    def sealed(self, tier_of: np.ndarray) -> np.ndarray:
        """(n,) bool: position lies in a complete block whose positions all have a tier."""
        n = len(tier_of)
        nb = n // self.G
        ok = np.zeros(n, dtype=bool)
        if nb:
            full = (tier_of[:nb * self.G] != FP16_TIER).reshape(nb, self.G).all(1)
            ok[:nb * self.G] = np.repeat(full, self.G)
        return ok

    def _deq_keys(self, st: _Layer, b: int, kpos: np.ndarray) -> Tensor:
        """Dequantize the sealed tier-b keys (positions kpos, ascending)."""
        codes = _unpack(st.kp[b], b, st.D).float()                          # (n_b, H, D)
        _, inv = np.unique(kpos // self.G, return_inverse=True)
        sub = torch.as_tensor(inv, dtype=torch.long, device=codes.device)
        return codes * st.ks[b].float()[sub] + st.kz[b].float()[sub]

    def _full_fp(self, st: _Layer, tier_of: np.ndarray) -> Tuple[Tensor, Tensor]:
        """(seq, H, D) float32 K and V of the whole layer, from storage."""
        seq = len(tier_of)
        K = torch.empty(seq, st.H, st.D, dtype=torch.float32, device=st.device)
        V = torch.empty_like(K)
        sealed = self.sealed(tier_of)
        p16 = np.nonzero(tier_of == FP16_TIER)[0]
        if len(p16):
            idx = torch.as_tensor(p16, dtype=torch.long, device=st.device)
            K[idx] = st.res_k.float()
            V[idx] = st.res_v.float()
        pend = np.nonzero((tier_of != FP16_TIER) & ~sealed)[0]
        if len(pend):
            K[torch.as_tensor(pend, dtype=torch.long, device=st.device)] = st.pend_k.float()
        for b in self.tiers:
            vpos = np.nonzero(tier_of == b)[0]
            if len(vpos):
                V[torch.as_tensor(vpos, dtype=torch.long, device=st.device)] = dequantize(
                    st.vp[b], st.vs[b], st.vz[b], b, self.group_size).float()
            kpos = np.nonzero((tier_of == b) & sealed)[0]
            if len(kpos):
                K[torch.as_tensor(kpos, dtype=torch.long, device=st.device)] = self._deq_keys(st, b, kpos)
        return K, V

    # ---- write ---------------------------------------------------------------

    def append_tokens(self, layer: int, k: Tensor, v: Tensor) -> None:
        """k, v: (1, kv_heads, n_new, head_dim). New tokens enter the FP16 residual."""
        st = self._layers[layer]
        dev = self.storage_device or k.device
        k_t = k[0].detach().permute(1, 0, 2).to(dev)
        v_t = v[0].detach().permute(1, 0, 2).to(dev)
        if st.H is None:
            self._init_layer(st, k_t, k.dtype, dev)
        st.res_k = torch.cat([st.res_k, k_t.to(st.dtype)], 0)
        st.res_v = torch.cat([st.res_v, v_t.to(st.dtype)], 0)
        st.tier_of = np.concatenate([st.tier_of, np.full(k_t.shape[0], FP16_TIER, dtype=np.int8)])

    def set_tiers(self, new_tiers: np.ndarray) -> int:
        """Commit (16 -> b) and demote (b -> c < b). Returns number of changed positions."""
        new = np.asarray(new_tiers).astype(np.int8)
        if len(new) != self.seq_len:
            raise ValueError(f"new_tiers len {len(new)} != seq_len {self.seq_len}")
        allowed = set(self.tiers) | {FP16_TIER}
        if not set(np.unique(new).tolist()) <= allowed:
            raise ValueError(f"tiers must be in {sorted(allowed)}, got {np.unique(new)}")
        n_changed = 0
        for li, st in enumerate(self._layers):
            old = st.tier_of
            if np.any(new > old):
                raise ValueError("promotion is not supported (re-quantization cannot recover information)")
            changed = np.nonzero(new != old)[0]
            if li == 0:
                n_changed = int(len(changed))
            if len(changed) == 0:
                continue
            K, V = self._full_fp(st, old)            # values as currently stored
            self._apply(st, old, new, K, V)
            st.tier_of = new.copy()
        return n_changed

    def _apply(self, st: _Layer, old, new, K: Tensor, V: Tensor) -> None:
        dev = st.device
        blk = np.arange(len(new)) // self.G
        s_old, s_new = self.sealed(old), self.sealed(new)
        # pending FP16 keys: tiered positions of unsealed blocks (rebuilt from exact FP16 values)
        pend_old = (old != FP16_TIER) & ~s_old
        pend_new = (new != FP16_TIER) & ~s_new
        if not np.array_equal(pend_old, pend_new):
            idx = torch.as_tensor(np.nonzero(pend_new)[0], dtype=torch.long, device=dev)
            st.pend_k = K.index_select(0, idx).to(st.dtype).contiguous()
        # FP16 residual: keep the rows that stay at 16
        p16_old = np.nonzero(old == FP16_TIER)[0]
        keep16 = np.nonzero(new[p16_old] == FP16_TIER)[0]
        if len(keep16) != len(p16_old):
            idx = torch.as_tensor(keep16, dtype=torch.long, device=dev)
            st.res_k = st.res_k.index_select(0, idx).contiguous()
            st.res_v = st.res_v.index_select(0, idx).contiguous()

        for b in self.tiers:
            self._apply_keys(st, b, (old == b) & s_old, (new == b) & s_new, blk, K)
            self._apply_values(st, b, old == b, new == b, V)

    def _apply_keys(self, st: _Layer, b: int, in_old, in_new, blk, K: Tensor) -> None:
        """Sealed tier-b keys: keep clean (block, tier) subsets byte-exact, re-quantize dirty ones."""
        if np.array_equal(in_old, in_new):
            return
        dev = st.device
        pos_old = np.nonzero(in_old)[0]
        pos_new = np.nonzero(in_new)[0]
        dirty = np.unique(blk[np.nonzero(in_old != in_new)[0]])         # blocks whose tier-b subset changed
        ub_old = np.unique(blk[pos_old])
        keep_rows = np.nonzero(~np.isin(blk[pos_old], dirty))[0]
        keep_sub = np.nonzero(~np.isin(ub_old, dirty))[0]
        rows_pos = [pos_old[keep_rows]]
        kp_parts = [st.kp[b].index_select(0, torch.as_tensor(keep_rows, dtype=torch.long, device=dev))]
        sub_blk = [ub_old[keep_sub]]
        si = torch.as_tensor(keep_sub, dtype=torch.long, device=dev)
        ks_parts = [st.ks[b].index_select(0, si)]
        kz_parts = [st.kz[b].index_select(0, si)]
        for beta in dirty:
            members = pos_new[blk[pos_new] == beta]
            if len(members) == 0:
                continue
            kp, ks, kz = quantize_keys_per_channel(K[torch.as_tensor(members, device=dev)], b)
            rows_pos.append(members)
            kp_parts.append(kp)
            sub_blk.append(np.array([beta]))
            ks_parts.append(ks.unsqueeze(0))
            kz_parts.append(kz.unsqueeze(0))
        order = torch.as_tensor(np.argsort(np.concatenate(rows_pos), kind="stable"), dtype=torch.long, device=dev)
        st.kp[b] = torch.cat(kp_parts, 0).index_select(0, order).contiguous()
        sorder = torch.as_tensor(np.argsort(np.concatenate(sub_blk), kind="stable"), dtype=torch.long, device=dev)
        st.ks[b] = torch.cat(ks_parts, 0).index_select(0, sorder).contiguous()
        st.kz[b] = torch.cat(kz_parts, 0).index_select(0, sorder).contiguous()

    def _apply_values(self, st: _Layer, b: int, in_old, in_new, V: Tensor) -> None:
        """Tier-b values, per token: staying rows keep their bytes, entering rows are quantized."""
        if np.array_equal(in_old, in_new):
            return
        dev = st.device
        pos_old = np.nonzero(in_old)[0]
        stay = np.nonzero(in_new[pos_old])[0]
        enter = np.nonzero(in_new & ~in_old)[0]
        si = torch.as_tensor(stay, dtype=torch.long, device=dev)
        vp = [st.vp[b].index_select(0, si)]
        vs = [st.vs[b].index_select(0, si)]
        vz = [st.vz[b].index_select(0, si)]
        if len(enter):
            p, s, z = quantize(V[torch.as_tensor(enter, device=dev)], b, self.group_size)
            vp.append(p); vs.append(s); vz.append(z)
        vorder = torch.as_tensor(np.argsort(np.concatenate([pos_old[stay], enter]), kind="stable"),
                                 dtype=torch.long, device=dev)
        st.vp[b] = torch.cat(vp, 0).index_select(0, vorder).contiguous()
        st.vs[b] = torch.cat(vs, 0).index_select(0, vorder).contiguous()
        st.vz[b] = torch.cat(vz, 0).index_select(0, vorder).contiguous()

    # ---- read ----------------------------------------------------------------

    def get_kv_fp16(self, layer: int, device) -> Tuple[Tensor, Tensor]:
        """Dequantize the whole layer. Returns (1, H, seq, D) K and V (model dtype) on `device`."""
        st = self._layers[layer]
        if len(st.tier_of) == 0:
            raise RuntimeError(f"No tokens stored for layer {layer}")
        K, V = self._full_fp(st, st.tier_of)
        K = K.to(device=device, dtype=st.dtype)
        V = V.to(device=device, dtype=st.dtype)
        return K.permute(1, 0, 2).unsqueeze(0).contiguous(), V.permute(1, 0, 2).unsqueeze(0).contiguous()

    # ---- properties ----------------------------------------------------------

    @property
    def seq_len(self) -> int:
        return len(self._layers[0].tier_of)

    def tier_counts(self) -> Dict[int, int]:
        t = self._layers[0].tier_of
        out = {b: int((t == b).sum()) for b in self.tiers}
        out[FP16_TIER] = int((t == FP16_TIER).sum())
        return out

    def check_consistency(self) -> None:
        ref = self._layers[0].tier_of
        for li, st in enumerate(self._layers):
            assert np.array_equal(st.tier_of, ref), f"layer {li} tier map differs"
            assert st.res_k.shape[0] == int((st.tier_of == FP16_TIER).sum()), f"layer {li} residual size"
            sealed = self.sealed(st.tier_of)
            assert st.pend_k.shape[0] == int(((st.tier_of != FP16_TIER) & ~sealed).sum()), f"layer {li} pending"
            for b in self.tiers:
                vpos = np.nonzero(st.tier_of == b)[0]
                kpos = np.nonzero((st.tier_of == b) & sealed)[0]
                assert st.vp[b].shape[0] == len(vpos), f"layer {li} tier {b} value rows"
                assert st.kp[b].shape[0] == len(kpos), f"layer {li} tier {b} key rows"
                assert st.ks[b].shape[0] == len(np.unique(kpos // self.G)), f"layer {li} tier {b} subsets"


def measure_kv_bytes_channel(cache: ChannelKVCache) -> dict:
    """Sum of real tensor .nbytes: codes, scales, zero-points, FP16 residual, tier index."""
    data = {b: 0 for b in QUANT_TIERS}
    scale = {b: 0 for b in QUANT_TIERS}
    residual = 0
    pending = 0
    n_vals = 0
    for st in cache._layers:
        if st.H is None:
            continue
        residual += st.res_k.nbytes + st.res_v.nbytes
        pending += st.pend_k.nbytes
        for b in cache.tiers:
            data[b] += st.kp[b].nbytes + st.vp[b].nbytes
            scale[b] += st.ks[b].nbytes + st.kz[b].nbytes + st.vs[b].nbytes + st.vz[b].nbytes
        n_vals += len(st.tier_of) * st.H * st.D * 2
    index_bytes = cache._layers[0].tier_of.nbytes
    total = sum(data.values()) + sum(scale.values()) + residual + pending + index_bytes
    return {
        "int8_data_bytes": data[8], "int4_data_bytes": data[4], "int2_data_bytes": data[2],
        "int8_scale_bytes": scale[8], "int4_scale_bytes": scale[4], "int2_scale_bytes": scale[2],
        "fp16_residual_bytes": residual,
        "fp16_pending_key_bytes": pending,
        "aux_bytes": 0,
        "index_bytes": index_bytes,
        "total_bytes": total,
        "n_original_values": n_vals,
        "fp16_equivalent": n_vals * 2,
        "is_simulated": False,
        "key_quant": "per_channel",
        "key_group_tokens": cache.G,
    }


def compute_effective_bits_channel(cache: ChannelKVCache) -> float:
    m = measure_kv_bytes_channel(cache)
    return m["total_bytes"] * 8.0 / m["n_original_values"] if m["n_original_values"] else 0.0
