from __future__ import annotations

import numpy as np

from .problem import Partition


def linear_minimization_oracle(gradient: np.ndarray, partition: Partition) -> np.ndarray:
    """Solve min <gradient, s> over the product of simplices."""

    grad = np.asarray(gradient, dtype=float).reshape(-1)
    partition.validate(grad.size)

    s = np.zeros_like(grad)
    for block in partition.blocks:
        local_index = int(np.argmin(grad[block]))
        s[block[local_index]] = 1.0

    return s
