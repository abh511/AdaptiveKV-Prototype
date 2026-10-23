"""
adaptivekv/importance.py
------------------------
Hook-based per-token importance extraction and EMA tracking.

Importance S_t[i] = sum_l w_l * mean_{h, q in last W rows}( attn_l[h, q, i] )
with sum_l w_l = 1 (uniform by default).

Fixes vs previous version
  * Call the model with output_attentions=False. With eager attention the
    attention module still returns its weights, so this hook receives them,
    but transformers does not collect them. With output_attentions=True,
    transformers 5.x records every layer's [heads x seq x seq] tensor through
    its own output recorder (drop_attn below does NOT prevent that), which
    OOMs a 4GB card at ~2048 tokens. Measured on transformers 5.18: identical
    scores, prefill peak 1.46 GB instead of OOM at 2048 tokens.
  * drop_attn=True additionally replaces the weights in the layer output with
    None (harmless; kept for older transformers versions).
  * window W (config key `importance_window`): average the LAST W query rows
    instead of only the last one. A single last-token row is a very noisy
    importance signal during prefill.
  * Accumulation stays on the GPU (no per-layer CPU sync).
  * get_scores() raises if not all layers reported (e.g. SDPA attention
    returns no weights) instead of silently returning partial scores.
"""

import numpy as np
import torch
from typing import List, Optional


class AttentionHook:
    def __init__(self, model, layer_weights: Optional[List[float]] = None,
                 window: int = 1, drop_attn: bool = True):
        self._model = model
        self._hooks = []
        self._acc: Optional[torch.Tensor] = None
        self._n_layers = len(model.model.layers)
        self._n_calls = 0
        self.window = max(1, int(window))
        self.drop_attn = drop_attn

        if layer_weights is not None:
            if len(layer_weights) != self._n_layers:
                raise ValueError("layer_weights length must equal num layers")
            w = np.array(layer_weights, dtype=np.float64)
            self._layer_weights = w / w.sum()
        else:
            self._layer_weights = np.ones(self._n_layers) / self._n_layers

    def register(self) -> None:
        self.remove()
        self._acc = None
        self._n_calls = 0

        for layer_idx, layer in enumerate(self._model.model.layers):
            weight = float(self._layer_weights[layer_idx])

            def make_hook(weight: float):
                def hook(module, inputs, output):
                    if not isinstance(output, (tuple, list)) or len(output) < 2:
                        return None
                    attn = output[1]                     # (1, Hq, Sq, Sk) or None
                    if attn is None:
                        return None
                    w = min(self.window, attn.shape[2])
                    contrib = attn[0, :, -w:, :].float().mean(dim=(0, 1)) * weight   # (Sk,)
                    contrib = contrib.detach()
                    self._acc = contrib if self._acc is None else self._acc + contrib
                    self._n_calls += 1
                    if self.drop_attn:
                        return (output[0], None) + tuple(output[2:])
                    return None
                return hook

            self._hooks.append(layer.self_attn.register_forward_hook(make_hook(weight)))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_scores(self) -> np.ndarray:
        """Per-token scores of the current forward pass, shape (seq_len,)."""
        if self._acc is None or self._n_calls == 0:
            raise RuntimeError(
                "No attention weights collected. Load the model with "
                "attn_implementation='eager' (the eager attention module returns its "
                "weights to this hook; output_attentions is not needed).")
        if self._n_calls != self._n_layers:
            raise RuntimeError(
                f"Only {self._n_calls}/{self._n_layers} layers returned attention weights.")
        return self._acc.cpu().numpy().astype(np.float64)

    def reset(self) -> None:
        self._acc = None
        self._n_calls = 0


class EMATracker:
    """
    importance_t[i] = alpha * score_t[i] + (1 - alpha) * importance_{t-1}[i]
    A position seen for the first time is initialised to its raw score.
    """

    def __init__(self, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        self.alpha = alpha
        self._importance: Optional[np.ndarray] = None

    def update(self, scores: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        if self._importance is None:
            self._importance = scores.copy()
            return
        n_old = len(self._importance)
        if len(scores) < n_old:
            raise ValueError(f"scores shorter ({len(scores)}) than tracked ({n_old})")
        self._importance = np.concatenate([
            self.alpha * scores[:n_old] + (1 - self.alpha) * self._importance,
            scores[n_old:],                       # new positions: raw score
        ])

    def get(self) -> np.ndarray:
        if self._importance is None:
            raise RuntimeError("EMATracker has no data yet. Call update() first.")
        return self._importance.copy()

    def reset(self) -> None:
        self._importance = None