from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .problem import Partition, SimplexQP


DEFAULT_METADATA_FILENAMES = ("partition.json", "problem_metadata.json", "metadata.json")


def infer_equal_partition(n: int, num_blocks: int) -> Partition:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be strictly positive.")
    if n % num_blocks != 0:
        raise ValueError("n must be divisible by num_blocks to infer equal contiguous blocks.")
    block_size = n // num_blocks
    return Partition.from_block_sizes([block_size] * num_blocks)


def save_partition_metadata(
    target: str | Path,
    partition: Partition,
    *,
    extra: dict[str, object] | None = None,
) -> Path:
    path = Path(target)
    if path.suffix != ".json":
        path = path / "partition.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "index_base": 0,
        "dimension": partition.dimension_hint,
        "num_blocks": partition.num_blocks,
        "num_simplices": partition.num_simplices,
        "block_sizes": list(partition.block_sizes),
        "index_sets": [index_set.tolist() for index_set in partition.index_sets],
        "blocks": [block.tolist() for block in partition.blocks],
    }
    if extra is not None:
        payload.update(extra)

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_partition_metadata(target: str | Path) -> Partition:
    path = _resolve_metadata_path(target)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "index_sets" in payload or "blocks" in payload:
        blocks = payload.get("index_sets", payload.get("blocks"))
        index_base = int(payload.get("index_base", 0))
        shifted_blocks = [
            np.asarray(block, dtype=int) - index_base for block in blocks
        ]
        return Partition(tuple(shifted_blocks))

    if "block_sizes" in payload:
        return Partition.from_block_sizes(tuple(int(size) for size in payload["block_sizes"]))

    raise ValueError(f"Unsupported partition metadata format in {path}.")


def load_problem(
    data_folder: str | Path,
    matrix_name: str,
    vector_name: str,
    *,
    metadata_path: str | Path | None = None,
    num_blocks: int | None = None,
    write_inferred_metadata: bool = False,
) -> SimplexQP:
    folder = Path(data_folder)
    matrices_path = folder / "matrices.npz"
    vectors_path = folder / "vectors.npz"

    matrices = np.load(matrices_path)
    vectors = np.load(vectors_path)
    try:
        Q = matrices[matrix_name]
    except KeyError as exc:
        raise KeyError(f"Matrix '{matrix_name}' was not found in {matrices_path}.") from exc
    try:
        q = vectors[vector_name]
    except KeyError as exc:
        raise KeyError(f"Vector '{vector_name}' was not found in {vectors_path}.") from exc

    partition = _load_or_infer_partition(
        folder,
        len(q),
        metadata_path=metadata_path,
        num_blocks=num_blocks,
        write_inferred_metadata=write_inferred_metadata,
    )
    return SimplexQP(Q=Q, q=q, partition=partition, name=f"{matrix_name}_{vector_name}")


def load_initial_point(path: str | Path, *, key: str) -> np.ndarray:
    """Load a named initial point from the project initial_points.npz file."""

    source = Path(path)
    _validate_initial_points_path(source)
    if not key:
        raise ValueError("An initial point key is required when loading from .npz.")

    archive = np.load(source)
    try:
        values = archive[key]
    except KeyError as exc:
        available = ", ".join(archive.files)
        raise KeyError(f"Initial point '{key}' was not found in {source}. Available: {available}") from exc
    return np.asarray(values, dtype=float).reshape(-1)


def load_initial_point_keys(path: str | Path) -> list[str]:
    """Return the initial point keys stored in the project initial_points.npz file."""

    source = Path(path)
    _validate_initial_points_path(source)
    archive = np.load(source)
    return list(archive.files)


def _validate_initial_points_path(path: Path) -> None:
    if path.suffix.lower() != ".npz":
        raise ValueError("Initial points must be loaded from a .npz file.")


def linear_term_from_stationary_point(Q: np.ndarray, x_u: np.ndarray) -> np.ndarray:
    """Return q = -2Qx_u so that x_u is a stationary point of f(x) = x^TQx + q^Tx."""

    matrix = np.asarray(Q, dtype=float)
    point = np.asarray(x_u, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if matrix.shape[0] != point.size:
        raise ValueError("Q and x_u must define the same dimension.")
    return -2.0 * (matrix @ point)


def _load_or_infer_partition(
    folder: Path,
    n: int,
    *,
    metadata_path: str | Path | None,
    num_blocks: int | None,
    write_inferred_metadata: bool,
) -> Partition:
    try:
        if metadata_path is not None:
            partition = load_partition_metadata(metadata_path)
        else:
            partition = load_partition_metadata(folder)
    except FileNotFoundError:
        if num_blocks is None:
            raise FileNotFoundError(
                "Partition metadata was not found. Provide a metadata JSON file or pass num_blocks "
                "to infer the current equal-block dataset layout."
            ) from None
        partition = infer_equal_partition(n, num_blocks)
        if write_inferred_metadata:
            save_partition_metadata(folder, partition, extra={"source": "inferred_equal_blocks"})

    partition.validate(n)
    return partition


def _resolve_metadata_path(target: str | Path) -> Path:
    path = Path(target)
    if path.is_file():
        return path
    for filename in DEFAULT_METADATA_FILENAMES:
        candidate = path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No partition metadata found under {path}. Expected one of: "
        + ", ".join(DEFAULT_METADATA_FILENAMES)
    )
