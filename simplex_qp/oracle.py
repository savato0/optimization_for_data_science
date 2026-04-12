from __future__ import annotations

import numpy as np

from .problem import Partition


def linear_minimization_oracle(gradient: np.ndarray, partition: Partition) -> np.ndarray:
    """Solve min_{s in D} <gradient, s> blockwise over the disjoint simplices."""

    grad = np.asarray(gradient, dtype=float).reshape(-1)
    partition.validate(grad.size)

    s_t = np.zeros_like(grad)
    for index_set in partition.index_sets:
        local_index = int(np.argmin(grad[index_set]))
        s_t[index_set[local_index]] = 1.0

    return s_t
