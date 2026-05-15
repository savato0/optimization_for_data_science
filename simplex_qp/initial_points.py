from __future__ import annotations

from pathlib import Path

import numpy as np

from .problem import Partition


def generate_initial_points(
    partition: Partition,
    *,
    dimension: int,
    num_canonical_vertices: int,
    num_random_vertices: int,
    num_random_feasible: int = 0,
    seed: int,
) -> dict[str, np.ndarray]:
    if num_canonical_vertices < 0:
        raise ValueError("num_canonical_vertices must be non-negative.")
    if num_random_vertices < 0:
        raise ValueError("num_random_vertices must be non-negative.")
    if num_random_feasible < 0:
        raise ValueError("num_random_feasible must be non-negative.")

    partition.validate(dimension)
    points: dict[str, np.ndarray] = {
        "barycenter": partition.barycenter(dimension),
    }

    for local_index in _canonical_local_indices(partition, num_canonical_vertices):
        points[f"vertex_{local_index}"] = canonical_vertex(
            partition,
            local_index=local_index,
            dimension=dimension,
        )

    rng = np.random.default_rng(seed)
    for random_id in range(num_random_vertices):
        points[f"random_vertex_{random_id}"] = random_vertex(
            partition,
            rng=rng,
            dimension=dimension,
        )
    for random_id in range(num_random_feasible):
        points[f"random_feasible_{random_id}"] = random_feasible_point(
            partition,
            rng=rng,
            dimension=dimension,
        )

    return points


def canonical_vertex(partition: Partition, *, local_index: int, dimension: int) -> np.ndarray:
    if local_index < 0:
        raise ValueError("local_index must be non-negative.")

    partition.validate(dimension)
    x = np.zeros(dimension, dtype=float)
    for block in partition.blocks:
        if local_index >= block.size:
            raise ValueError(
                f"Cannot build vertex_{local_index}: block size {block.size} is too small."
            )
        x[block[local_index]] = 1.0
    return x


def random_vertex(
    partition: Partition,
    *,
    rng: np.random.Generator,
    dimension: int,
) -> np.ndarray:
    partition.validate(dimension)
    x = np.zeros(dimension, dtype=float)
    for block in partition.blocks:
        selected_index = int(rng.choice(block))
        x[selected_index] = 1.0
    return x


def random_feasible_point(
    partition: Partition,
    *,
    rng: np.random.Generator,
    dimension: int,
) -> np.ndarray:
    partition.validate(dimension)
    x = np.zeros(dimension, dtype=float)
    for block in partition.blocks:
        active_count = int(rng.integers(1, block.size + 1))
        active_indices = rng.choice(block, size=active_count, replace=False)
        if active_count == 1:
            x[int(active_indices[0])] = 1.0
            continue

        weights = rng.dirichlet(np.ones(active_count, dtype=float))
        x[active_indices] = weights
    return x


def save_initial_points(path: str | Path, points: dict[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **points)
    return target


def _canonical_local_indices(partition: Partition, count: int) -> list[int]:
    if count == 0:
        return []

    max_count = min(min(partition.block_sizes), count)
    last_local_index = min(partition.block_sizes) - 1
    return list(np.linspace(0, last_local_index, num=max_count, dtype=int))
