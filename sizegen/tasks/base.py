"""Common result container for the iterative-operator tasks (paper Def. 2).

Every F^op task is solved by repeating a local update U_N from an initial
state; ``iterates[k]`` is the k-step truncation, whose value at node v depends
only on the k-hop neighborhood B_k(v).  ``target`` is the (numerically exact)
fixed point f*.  ``rho`` is the known contraction rate when the task has one.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TaskResult:
    target: np.ndarray            # (N,) or (N, F) exact target f*
    inputs: np.ndarray            # (N, F_in) node input features for learning
    iterates: List[np.ndarray]    # iterates[k] = radius-k truncation m_k, k = 0..K
    rho: Optional[float] = None   # known contraction rate (None if non-contractive)
    meta: Dict = field(default_factory=dict)

    def truncation_mse(self, reduction: str = "mean") -> np.ndarray:
        """Squared radius-k truncation error eps^2(k), k = 0..K.

        Empirical counterpart of the paper's resolution profile (Lemma 4):
        eps_exp^2(k) = ||f* - m_k||^2 ~ rho^{2k} C(N).

        reduction:
          "mean" — per-node mean; for O(1)-valued tasks this saturates in N
                   (curves for different N coincide once k-balls stop growing).
          "sum"  — graph-level norm; the far field C(N) then grows with N on
                   F2 targets, which is what drives the paper's depth law
                   L(N) ~ log N (cf. PageRank T*(N) = O(log N), paper §7.2).
        """
        red = np.mean if reduction == "mean" else np.sum
        tgt = self.target
        return np.array(
            [float(red((m - tgt) ** 2)) for m in self.iterates]
        )

    def depth_to_tolerance(self, tau: float, reduction: str = "sum") -> int:
        """L(N): least k with eps(k) <= tau (paper Lemma 4). -1 if never."""
        eps = np.sqrt(self.truncation_mse(reduction=reduction))
        hits = np.where(eps <= tau)[0]
        return int(hits[0]) if hits.size else -1
