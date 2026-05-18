from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simplex_qp import Partition, load_partition_metadata, resolve_problem_data_folder


TARGET_COLUMNS = [
    "target",
    "min_value",
    "negative_count",
    "max_block_sum_error",
    "l2_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize x_u targets used to generate q vectors.")
    parser.add_argument(
        "data_folder",
        help="Problem folder containing data/ or the data folder containing targets.npz and partition.json.",
    )
    parser.add_argument(
        "--targets-file",
        help="Optional explicit path to targets.npz. Defaults to DATA_FOLDER/targets.npz.",
    )
    parser.add_argument(
        "--metadata",
        help="Optional explicit partition metadata JSON. Defaults to DATA_FOLDER/partition.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_target_records(
        args.data_folder,
        targets_file=args.targets_file,
        metadata_path=args.metadata,
    )
    print(format_table(records), flush=True)


def load_target_records(
    data_folder: str | Path,
    *,
    targets_file: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    folder = resolve_problem_data_folder(data_folder)
    target_path = Path(targets_file) if targets_file is not None else folder / "targets.npz"
    partition = load_partition_metadata(metadata_path if metadata_path is not None else folder)

    archive = np.load(target_path)
    return [
        summarize_target(key, np.asarray(archive[key], dtype=float).reshape(-1), partition)
        for key in archive.files
    ]


def summarize_target(
    key: str,
    x_u: np.ndarray,
    partition: Partition,
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    point = np.asarray(x_u, dtype=float).reshape(-1)
    partition.validate(point.size)

    block_sums = partition.block_sums(point)
    block_sum_errors = np.abs(block_sums - 1.0)

    return {
        "target": key,
        "min_value": float(point.min()),
        "negative_count": int(np.sum(point < -atol)),
        "max_block_sum_error": float(block_sum_errors.max()) if block_sum_errors.size else 0.0,
        "l2_norm": float(np.linalg.norm(point)),
    }


def format_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No targets found."

    rows = [[_format_value(record.get(column)) for column in TARGET_COLUMNS] for record in records]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(TARGET_COLUMNS)
    ]
    header = "  ".join(column.ljust(widths[index]) for index, column in enumerate(TARGET_COLUMNS))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(TARGET_COLUMNS)))
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6e}"
    return str(value)


if __name__ == "__main__":
    main()
