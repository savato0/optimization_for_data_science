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
    """Minimize the quadratic objective along x + gamma * direction for gamma in [0, 1]."""

    point = np.asarray(x, dtype=float).reshape(-1)
    step_direction = np.asarray(direction, dtype=float).reshape(-1)
    grad = problem.gradient(point) if gradient is None else np.asarray(gradient, dtype=float)

    quadratic_term = float(step_direction @ problem.Q @ step_direction)
    linear_term = float(grad @ step_direction)

    if quadratic_term <= atol:
        return 1.0 if linear_term < 0.0 else 0.0

    gamma = -linear_term / (2.0 * quadratic_term)
    return float(np.clip(gamma, 0.0, 1.0))
