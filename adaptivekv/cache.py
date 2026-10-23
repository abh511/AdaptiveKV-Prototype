"""
adaptivekv/cache.py
-------------------
AdaptiveKVCache: K/V stored as packed integers, one precision tier per token.

What changed vs the previous version
  1. TIER BANKS instead of one tiny tensor-set per token. Each layer keeps one
     contiguous bank per tier (8/4). Dequantization is ONE batched op per bank
     (previously ~seq_len x 2 x n_layers separate ops per step = very slow), and
     the byte accounting is exact (no per-allocation padding hiding real memory).
  2. KEY CHANNEL NORMALISATION (k_channel_norm=True). Qwen keys have a few huge
     outlier channels. With per-token groups, one outlier stretches the range of
     the      whole group, so INT4 keys collapse to garbage. We divide K by a
     per-(kv_head, channel) scale (median |k| from the first appended tokens)
     before quantizing and multiply back on read. The scale vector is stored
     and COUNTED in memory (aux_bytes). V is not normalised.
  3. storage_device: packed data lives on `storage_device`. Default None = the
     device of the incoming tensors (GPU), so the stored bytes really are VRAM.
     Use "cpu" to keep the packed cache in system RAM.
  4. Memory accounting includes data + scales + zero-points + channel scales +
     tier-index metadata. FP16-equivalent bytes are computed from real tensor
     shapes (no hardcoded 12288).

Honest limitation: get_kv_fp16() dequantizes the whole cache into FP16 for the
attention call, so peak VRAM during a step is NOT reduced by this design; the
saving is in what is STORED between steps.

Tiers are identical across layers by construction; the tier index (1 byte per
token) is therefore counted once.

`tiers` selects the active precisions: (8, 4) by default, (8, 4, 2) for the
INT2 ablation (int2_enabled in config.yaml).
"""

import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from .quant import quantize, dequantize, packed_last_dim

TIERS = (8, 4)                 # default two-tier set
ALL_TIERS = (8, 4, 2)          # every precision the quantizer supports
_NAMES = ("kp", "ks", "kz", "vp", "vs", "vz")


# ---------------------------------------------------------------------------
# One contiguous bank of tokens at a single precision
# ---------------------------------------------------------------------------

class _TierBank:
    def __init__(self, bits: int, H: int, D: int, group_size: int, device):
        G = D // group_size
        pd = packed_last_dim(D, bits)
        self.bits = bits
        self.n = 0
        self.kp = torch.empty(0, H, pd, dtype=torch.uint8, device=device)
        self.vp = torch.empty(0, H, pd, dtype=torch.uint8, device=device)
        self.ks = torch.empty(0, H, G, dtype=torch.float16, device=device)
        self.kz = torch.empty(0, H, G, dtype=torch.float16, device=device)
        self.vs = torch.empty(0, H, G, dtype=torch.float16, device=device)
        self.vz = torch.empty(0, H, G, dtype=torch.float16, device=device)

    def append(self, kp, ks, kz, vp, vs, vz) -> None:
        for name, t in zip(_NAMES, (kp, ks, kz, vp, vs, vz)):
            setattr(self, name, torch.cat([getattr(self, name), t], dim=0))
        self.n += kp.shape[0]

    def take(self, rows: Tensor) -> None:
        """Keep only `rows` (in the given order)."""
        for name in _NAMES:
            setattr(self, name, getattr(self, name).index_select(0, rows).contiguous())
        self.n = int(rows.numel())

    def data_bytes(self) -> int:
        return self.kp.nbytes + self.vp.nbytes

    def scale_bytes(self) -> int:
        return self.ks.nbytes + self.kz.nbytes + self.vs.nbytes + self.vz.nbytes


class _LayerStore:
    def __init__(self):
        self.H: Optional[int] = None
        self.D: Optional[int] = None
        self.banks: Dict[int, _TierBank] = {}
        self.tier_of = np.zeros(0, dtype=np.int8)   # tier id per token position
        self.k_chan: Optional[Tensor] = None        # (H, D) fp16 or None
        self.dtype = torch.float16                  # dtype returned on read (model dtype)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class AdaptiveKVCache:
    def __init__(self, n_layers: int, group_size: int,
                 storage_device=None, k_channel_norm: bool = True,
                 tiers: Tuple[int, ...] = TIERS):
        tiers = tuple(int(b) for b in tiers)
        if not tiers or not set(tiers) <= set(ALL_TIERS):
            raise ValueError(f"tiers must be a subset of {ALL_TIERS}, got {tiers}")
        self.tiers = tiers
        self.n_layers = n_layers
        self.group_size = group_size
        self.k_channel_norm = k_channel_norm
        self.storage_device = torch.device(storage_device) if storage_device else None
        self._layers: List[_LayerStore] = [_LayerStore() for _ in range(n_layers)]

    # ---- helpers ---------------------------------------------------------

    def _sdev(self, t: Tensor):
        return self.storage_device if self.storage_device is not None else t.device

    def _init_layer(self, st: _LayerStore, k_t: Tensor, device) -> None:
        """k_t: (n_new, H, D) float32. First append decides shapes + channel scale."""
        st.H, st.D = k_t.shape[1], k_t.shape[2]
        if st.D % self.group_size != 0:
            raise ValueError(f"head_dim {st.D} not divisible by group_size {self.group_size}")
        for b in self.tiers:
            st.banks[b] = _TierBank(b, st.H, st.D, self.group_size, device)
        if self.k_channel_norm:
            med = k_t.abs().median(dim=0).values                      # (H, D)
            floor = (0.05 * med.mean()).clamp(min=1e-3)
            st.k_chan = med.clamp(min=float(floor)).to(torch.float16)

    def _quant_rows(self, st: _LayerStore, k_t: Tensor, v_t: Tensor, bits: int):
        """k_t, v_t: (m, H, D) in the STORED space (K already channel-normalised)."""
        kp, ks, kz = quantize(k_t, bits, self.group_size)
        vp, vs, vz = quantize(v_t, bits, self.group_size)
        return kp, ks, kz, vp, vs, vz

    def _positions(self, st: _LayerStore, bits: int, device) -> Tensor:
        return torch.as_tensor(np.nonzero(st.tier_of == bits)[0], dtype=torch.long, device=device)

    # ---- write -----------------------------------------------------------

    def append_tokens(self, layer: int, k: Tensor, v: Tensor, tiers: np.ndarray) -> None:
        """
        k, v: (1, kv_heads, n_new, head_dim). Only the NEW tokens.
        tiers: (n_new,) values in {8, 4}.
        """
        st = self._layers[layer]
        dev = self._sdev(k)
        k_t = k[0].detach().to(dev).permute(1, 0, 2).float()   # (n_new, H, D)
        v_t = v[0].detach().to(dev).permute(1, 0, 2).float()
        n_new = k_t.shape[0]
        tiers = np.asarray(tiers).astype(np.int64)
        assert len(tiers) == n_new, f"tiers len {len(tiers)} != n_new {n_new}"
        if not np.isin(tiers, self.tiers).all():
            raise ValueError(f"tiers must be in {self.tiers}, got {np.unique(tiers)}")

        if st.H is None:
            st.dtype = k.dtype
            self._init_layer(st, k_t, dev)
        if st.k_chan is not None:
            k_t = k_t / st.k_chan.float().unsqueeze(0)

        for bits in self.tiers:
            sel = np.nonzero(tiers == bits)[0]
            if len(sel) == 0:
                continue
            idx = torch.as_tensor(sel, dtype=torch.long, device=dev)
            st.banks[bits].append(*self._quant_rows(st, k_t[idx], v_t[idx], bits))
        st.tier_of = np.concatenate([st.tier_of, tiers.astype(np.int8)])

    # ---- read ------------------------------------------------------------

    def get_kv_fp16(self, layer: int, device) -> Tuple[Tensor, Tensor]:
        """Dequantize the whole layer. Returns (1, H, seq, D) FP16 K and V on `device`."""
        st = self._layers[layer]
        seq = len(st.tier_of)
        if seq == 0:
            raise RuntimeError(f"No tokens stored for layer {layer}")
        K = torch.empty(seq, st.H, st.D, dtype=st.dtype, device=device)
        V = torch.empty_like(K)
        for bits, bank in st.banks.items():
            if bank.n == 0:
                continue
            pos = self._positions(st, bits, device)
            k = dequantize(bank.kp, bank.ks, bank.kz, bits, self.group_size)
            v = dequantize(bank.vp, bank.vs, bank.vz, bits, self.group_size)
            if st.k_chan is not None:
                k = (k.float() * st.k_chan.float().unsqueeze(0)).to(torch.float16)
            K[pos] = k.to(device=device, dtype=st.dtype)
            V[pos] = v.to(device=device, dtype=st.dtype)
        return (K.permute(1, 0, 2).unsqueeze(0).contiguous(),
                V.permute(1, 0, 2).unsqueeze(0).contiguous())

    # ---- re-assignment ---------------------------------------------------

    def reassign_tiers(self, new_tiers: np.ndarray) -> int:
        """
        Move tokens whose tier changed. Re-quantizes from the CURRENT stored
        values, so demotion is lossy and promotion cannot recover lost
        information (callers should demote only).
        Returns number of tokens changed (layer 0).
        """
        new = np.asarray(new_tiers).astype(np.int8)
        assert len(new) == self.seq_len, f"new_tiers len {len(new)} != seq_len {self.seq_len}"
        if not np.isin(new, self.tiers).all():
            raise ValueError(f"tiers must be in {self.tiers}")
        n_changed_layer0 = 0

        for li, st in enumerate(self._layers):
            old = st.tier_of
            changed = np.nonzero(old != new)[0]
            if li == 0:
                n_changed_layer0 = int(len(changed))
            if len(changed) == 0:
                continue
            dev = next(iter(st.banks.values())).kp.device
            m = len(changed)
            kfp = torch.empty(m, st.H, st.D, dtype=torch.float16, device=dev)
            vfp = torch.empty_like(kfp)
            where = np.full(len(old), -1, dtype=np.int64)
            where[changed] = np.arange(m)

            # 1) pull changed tokens out of their current banks
            for bits, bank in st.banks.items():
                if bank.n == 0:
                    continue
                pos = np.nonzero(old == bits)[0]                 # ascending, len == bank.n
                leave = np.nonzero(old[pos] != new[pos])[0]
                if len(leave) == 0:
                    continue
                keep = np.nonzero(old[pos] == new[pos])[0]
                rows = torch.as_tensor(leave, dtype=torch.long, device=dev)
                dst = torch.as_tensor(where[pos[leave]], dtype=torch.long, device=dev)
                kfp[dst] = dequantize(bank.kp.index_select(0, rows), bank.ks.index_select(0, rows),
                                      bank.kz.index_select(0, rows), bits, self.group_size)
                vfp[dst] = dequantize(bank.vp.index_select(0, rows), bank.vs.index_select(0, rows),
                                      bank.vz.index_select(0, rows), bits, self.group_size)
                bank.take(torch.as_tensor(keep, dtype=torch.long, device=dev))

            # 2) re-quantize into destination banks, keep each bank position-sorted
            for bits in self.tiers:
                dsel = np.nonzero(new[changed] == bits)[0]
                if len(dsel) == 0:
                    continue
                bank = st.banks[bits]
                existing_pos = np.nonzero((old == bits) & (old == new))[0]
                new_pos = changed[dsel]
                d = torch.as_tensor(dsel, dtype=torch.long, device=dev)
                bank.append(*self._quant_rows(st, kfp.index_select(0, d), vfp.index_select(0, d), bits))
                all_pos = np.concatenate([existing_pos, new_pos])
                order = np.argsort(all_pos, kind="stable")
                if not np.array_equal(order, np.arange(len(order))):
                    bank.take(torch.as_tensor(order, dtype=torch.long, device=dev))

            st.tier_of = new.copy()
        return n_changed_layer0

    # ---- properties ------------------------------------------------------

    @property
    def seq_len(self) -> int:
        return len(self._layers[0].tier_of)

    def tier_counts(self) -> Dict[int, int]:
        t = self._layers[0].tier_of
        return {b: int((t == b).sum()) for b in self.tiers}

    def check_consistency(self) -> None:
        """Assert all layers share identical tier assignment and bank sizes."""
        ref = self._layers[0].tier_of
        for li, st in enumerate(self._layers):
            assert np.array_equal(st.tier_of, ref), f"layer {li} tier map differs"
            for b, bank in st.banks.items():
                assert bank.n == int((st.tier_of == b).sum()), f"layer {li} bank {b} size mismatch"


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------

def _n_original_values(cache: AdaptiveKVCache) -> int:
    """Number of FP16 values this cache represents (K+V, all layers, from real shapes)."""
    total = 0
    for st in cache._layers:
        if st.H is not None:
            total += len(st.tier_of) * st.H * st.D * 2
    return total


def measure_kv_bytes(cache: AdaptiveKVCache) -> dict:
    """Sum of real tensor .nbytes (data + scales + zero-points + aux + index)."""
    data = {b: 0 for b in ALL_TIERS}
    scale = {b: 0 for b in ALL_TIERS}
    aux = 0
    for st in cache._layers:
        for b, bank in st.banks.items():
            data[b] += bank.data_bytes()
            scale[b] += bank.scale_bytes()
        if st.k_chan is not None:
            aux += st.k_chan.nbytes
    index_bytes = cache._layers[0].tier_of.nbytes      # shared across layers, counted once
    total = sum(data.values()) + sum(scale.values()) + aux + index_bytes
    n_vals = _n_original_values(cache)
    return {
        "int8_data_bytes": data[8], "int4_data_bytes": data[4],
        "int8_scale_bytes": scale[8], "int4_scale_bytes": scale[4],
        "int2_data_bytes": data[2], "int2_scale_bytes": scale[2],
        "aux_bytes": aux,
        "index_bytes": index_bytes,
        "total_bytes": total,
        "n_original_values": n_vals,
        "fp16_equivalent": n_vals * 2,
        "storage_device": (str(next(iter(cache._layers[0].banks.values())).kp.device)
                           if cache._layers[0].banks else "n/a"),
        "is_simulated": False,
    }


def compute_effective_bits(cache: AdaptiveKVCache) -> float:
    """Effective bits per KV value INCLUDING scales, zero-points, aux and index."""
    n = _n_original_values(cache)
    if n == 0:
        return 0.0
    return measure_kv_bytes(cache)["total_bytes"] * 8.0 / n