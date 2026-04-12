from __future__ import annotations

import numpy as np

from .problem import SimplexQP


def exact_line_search(
    problem: SimplexQP,
    x: np.ndarray,
    direction: np.ndarray,
    *,
    gradient: np.ndarray | None = None,
    atol: float = 1e-15,
) -> float:
    """Compute α_t* for x + α d with α in [0,1] using the report's closed form."""

    point = np.asarray(x, dtype=float).reshape(-1)
    d_t = np.asarray(direction, dtype=float).reshape(-1)
    grad = problem.gradient(point) if gradient is None else np.asarray(gradient, dtype=float)

    a_t = float(d_t @ problem.Q @ d_t)
    b_t = float(grad @ d_t)

    if a_t <= atol:
        return 1.0

    alpha_t = -b_t / (2.0 * a_t)
    return float(np.clip(alpha_t, 0.0, 1.0))
