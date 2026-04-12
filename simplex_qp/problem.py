from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Partition:
    """Disjoint blocks defining a product of simplices over zero-based indices."""

    blocks: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        canonical_blocks: list[np.ndarray] = []
        for raw_block in self.blocks:
            block = np.asarray(raw_block, dtype=int).reshape(-1)
            if block.size == 0:
                raise ValueError("Each simplex block must contain at least one index.")
            if np.any(block < 0):
                raise ValueError("Partition indices must be non-negative.")
            if np.unique(block).size != block.size:
                raise ValueError("Each simplex block must contain unique indices.")
            canonical_blocks.append(block.copy())

        flat = np.concatenate(canonical_blocks)
        if np.unique(flat).size != flat.size:
            raise ValueError("Partition blocks must be pairwise disjoint.")

        self.blocks = tuple(canonical_blocks)

    @classmethod
    def from_block_sizes(cls, block_sizes: list[int] | tuple[int, ...]) -> "Partition":
        start = 0
        blocks: list[np.ndarray] = []
        for size in block_sizes:
            if size <= 0:
                raise ValueError("Block sizes must be strictly positive.")
            blocks.append(np.arange(start, start + size, dtype=int))
            start += size
        return cls(tuple(blocks))

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(int(block.size) for block in self.blocks)

    @property
    def dimension_hint(self) -> int:
        return max(int(block.max()) for block in self.blocks) + 1

    def validate(self, n: int) -> None:
        flat = np.concatenate(self.blocks)
        expected = np.arange(n, dtype=int)
        if flat.size != n:
            raise ValueError(f"Partition covers {flat.size} indices, expected {n}.")
        if np.array_equal(np.sort(flat), expected):
            return
        raise ValueError("Partition blocks must cover each index in {0, ..., n-1} exactly once.")

    def barycenter(self, n: int | None = None) -> np.ndarray:
        dimension = self.dimension_hint if n is None else int(n)
        self.validate(dimension)
        x = np.zeros(dimension, dtype=float)
        for block in self.blocks:
            x[block] = 1.0 / block.size
        return x

    def block_sums(self, x: np.ndarray) -> np.ndarray:
        point = np.asarray(x, dtype=float).reshape(-1)
        return np.array([point[block].sum() for block in self.blocks], dtype=float)


@dataclass(slots=True)
class SimplexQP:
    """Convex quadratic program over a product of simplices."""

    Q: np.ndarray
    q: np.ndarray
    partition: Partition
    name: str = "simplex_qp"

    def __post_init__(self) -> None:
        self.Q = np.asarray(self.Q, dtype=float)
        self.q = np.asarray(self.q, dtype=float).reshape(-1)

        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be a square matrix.")
        if self.Q.shape[0] != self.q.size:
            raise ValueError("Q and q must define the same dimension.")
        if not np.allclose(self.Q, self.Q.T, atol=1e-12):
            raise ValueError("Q must be symmetric for the Frank-Wolfe implementation.")

        self.partition.validate(self.dimension)

    @property
    def dimension(self) -> int:
        return int(self.q.size)

    def objective(self, x: np.ndarray) -> float:
        point = self._as_point(x)
        return float(point @ self.Q @ point + self.q @ point)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        point = self._as_point(x)
        return 2.0 * (self.Q @ point) + self.q

    def barycenter(self) -> np.ndarray:
        return self.partition.barycenter(self.dimension)

    def feasibility_metrics(self, x: np.ndarray) -> dict[str, float]:
        point = self._as_point(x)
        block_sum_errors = np.abs(self.partition.block_sums(point) - 1.0)
        min_value = float(point.min())
        return {
            "max_block_sum_error": float(block_sum_errors.max()) if block_sum_errors.size else 0.0,
            "max_nonnegativity_violation": float(max(0.0, -min_value)),
            "min_value": min_value,
        }

    def is_feasible(self, x: np.ndarray, atol: float = 1e-9) -> bool:
        metrics = self.feasibility_metrics(x)
        return (
            metrics["max_block_sum_error"] <= atol
            and metrics["max_nonnegativity_violation"] <= atol
        )

    def _as_point(self, x: np.ndarray) -> np.ndarray:
        point = np.asarray(x, dtype=float).reshape(-1)
        if point.size != self.dimension:
            raise ValueError(
                f"Expected a point of dimension {self.dimension}, got {point.size}."
            )
        return point
