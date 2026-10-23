"""
adaptivekv/allocator.py
-----------------------
Tier assignment: rank tokens by importance and assign INT8 / INT4 (/ INT2).

  n8 = ceil(p8 * n)
  n4 = ceil((p8 + p4) * n) - n8
  n2 = n - n8 - n4              (0 unless the INT2 ablation sets p2 > 0)
  Ties: lower position index wins (stable sort).

With p2 = 0 this is exactly the two-tier rule (n4 = n - n8).

Optional pins (always INT8): first `sink_tokens`, last `recent_tokens`.
Pins consume the INT8 budget. If pins alone exceed ceil(p8*n), the INT8 share
is larger than p8 (documented behaviour; report it, don't hide it).

Fixes vs previous version
  * No forced "at least 1 token" in INT8/INT4: p4 = 0 now really gives no INT4.
  * Vectorised (no Python loops/sets).
"""

import math
import numpy as np

INT8, INT4, INT2 = 8, 4, 2


class TierAllocator:
    def __init__(self, p8: float, p4: float, sink_tokens: int = 0, recent_tokens: int = 0,
                 p2: float = 0.0):
        if p8 < 0 or p4 < 0 or p2 < 0 or abs(p8 + p4 + p2 - 1.0) > 1e-9:
            raise ValueError(f"p8, p4 and p2 must be non-negative and sum to 1, "
                             f"got p8={p8}, p4={p4}, p2={p2}")
        self.p8 = p8
        self.p4 = p4
        self.p2 = p2
        self.sink_tokens = max(0, int(sink_tokens))
        self.recent_tokens = max(0, int(recent_tokens))

    def pinned_mask(self, n: int) -> np.ndarray:
        pinned = np.zeros(n, dtype=bool)
        if self.sink_tokens > 0:
            pinned[:min(self.sink_tokens, n)] = True
        if self.recent_tokens > 0:
            pinned[max(0, n - self.recent_tokens):] = True
        return pinned

    def assign(self, importance: np.ndarray) -> np.ndarray:
        """importance: (n,) float. Returns (n,) int32 tiers in {8, 4, 2}."""
        importance = np.asarray(importance, dtype=np.float64)
        n = len(importance)
        tiers = np.full(n, INT4 if self.p2 == 0 else INT2, dtype=np.int32)
        if n == 0:
            return tiers

        pinned = self.pinned_mask(n)
        tiers[pinned] = INT8

        free = np.nonzero(~pinned)[0]
        if len(free) == 0:
            return tiers

        n8_total, n4_total, _ = self.targets3(n)
        n8_free = max(0, n8_total - int(pinned.sum()))

        order = free[np.argsort(-importance[free], kind="stable")]   # best first
        tiers[order[:n8_free]] = INT8
        tiers[order[n8_free:n8_free + n4_total]] = INT4
        return tiers

    # ------------------------------------------------------------------
    def targets3(self, n: int):
        """(n8, n4, n2) target counts for a cache of n tokens."""
        n8 = math.ceil(self.p8 * n - 1e-9)
        n84 = min(n, math.ceil((self.p8 + self.p4) * n - 1e-9))
        return n8, max(0, n84 - n8), max(0, n - n84)

    def targets(self, n: int):
        """(n8, n4) target counts for a cache of n tokens (two-tier view)."""
        n8 = math.ceil(self.p8 * n - 1e-9)
        return n8, max(0, n - n8)

    def demote(self, current: np.ndarray, importance: np.ndarray,
               extra_pinned: np.ndarray = None) -> np.ndarray:
        """
        Demotion-only update that CANNOT exceed the configured tier fractions.

        Why: demoting every token that is ever ranked low (min over history) is a
        Ranking is noisy, so with reassignment every step the INT4 share can
        drift. Here, the number of tokens at INT4 (and INT2) is capped at the
        target count; the least important unpinned tokens are demoted first,
        nothing is ever promoted, and pinned (sink / recent) tokens are never
        touched. INT2 is filled first (from INT8 or INT4), then INT4 from INT8,
        so the INT8 count never drops below its target.
        extra_pinned: additional positions that must not change (e.g. the
        not-yet-committed FP16 residual of the per-channel cache).
        """
        cur = np.asarray(current, dtype=np.int32)
        imp = np.asarray(importance, dtype=np.float64)
        n = len(cur)
        if len(imp) != n:
            raise ValueError(f"importance length {len(imp)} != tiers length {n}")
        new = cur.copy()
        if n == 0:
            return new
        _, n4, n2 = self.targets3(n)
        pinned = self.pinned_mask(n)
        if extra_pinned is not None:
            pinned = pinned | np.asarray(extra_pinned, dtype=bool)

        order = np.argsort(imp, kind="stable")        # least important first
        order = order[~pinned[order]]

        if n2 > 0:
            room2 = max(0, n2 - int((new == INT2).sum()))
            cand = order[new[order] > INT2]
            new[cand[:room2]] = INT2

        room4 = max(0, n4 - int((new == INT4).sum()))
        cand = order[new[order] > INT4]
        new[cand[:room4]] = INT4
        return new
