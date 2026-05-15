from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simplex_qp import (
    FrankWolfeConfig,
    load_initial_point,
    load_initial_point_keys,
    load_problem,
    solve_frank_wolfe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Frank-Wolfe experiments on saved QP cases.")
    parser.add_argument("data_folder", help="Folder containing matrices.npz and vectors.npz.")
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case identifier in the form MATRIX_NAME:VECTOR_NAME. Repeat for multiple cases.",
    )
    parser.add_argument("--metadata", help="Path to a partition metadata JSON file.")
    parser.add_argument(
        "--num-blocks",
        type=int,
        help="Infer equal contiguous simplex blocks when metadata is not available.",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Persist inferred equal-block metadata as partition.json in the data folder.",
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol-gap", type=float, default=1e-6)
    parser.add_argument(
        "--x0-file",
        help="Optional initial_points.npz file.",
    )
    parser.add_argument(
        "--x0-key",
        action="append",
        help="Initial point key to read from --x0-file.",
    )
    parser.add_argument(
        "--all-x0",
        action="store_true",
        help="Run every initial point key stored in --x0-file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a Frank-Wolfe progress line every N iterations.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-iteration Frank-Wolfe progress logging.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Start a fresh output file instead of resuming from an existing one. "
            "By default, existing records are kept and matching matrix/vector/x0 runs are skipped."
        ),
    )
    parser.add_argument(
        "--include-solution",
        action="store_true",
        help="Include the full solution vector in the JSON payload.",
    )
    args = parser.parse_args()
    if args.all_x0 and not args.x0_file:
        parser.error("--all-x0 requires --x0-file.")
    if args.all_x0 and args.x0_key:
        parser.error("--all-x0 cannot be combined with --x0-key.")
    if args.x0_file and not args.x0_key and not args.all_x0:
        parser.error("--x0-file requires --x0-key or --all-x0.")
    if args.x0_key and not args.x0_file:
        parser.error("--x0-key requires --x0-file.")
    return args


def main() -> None:
    args = parse_args()
    x0_keys = _resolve_x0_keys(args.x0_file, args.x0_key, args.all_x0)
    output_path = Path(args.output) if args.output else Path(args.data_folder) / "fw_results.json"

    if args.overwrite:
        records: list[dict[str, Any]] = []
        _write_records(output_path, records)
        print(f"[FW] Starting fresh output file at {output_path}", flush=True)
    else:
        records = _load_existing_records(output_path)
        if records:
            print(
                f"[FW] Resuming from {output_path}: {len(records)} existing result(s).",
                flush=True,
            )
        else:
            _write_records(output_path, records)
            print(f"[FW] Writing checkpoints to {output_path}", flush=True)

    completed_runs = {_record_key(record) for record in records}
    for raw_case in args.case:
        matrix_name, vector_name = parse_case(raw_case)
        pending_x0_keys = [
            x0_key
            for x0_key in x0_keys
            if (matrix_name, vector_name, x0_key) not in completed_runs
        ]
        if not pending_x0_keys:
            print(
                f"[FW] Skipping {matrix_name} + {vector_name}: "
                "all requested x0 keys are already saved.",
                flush=True,
            )
            continue

        print(f"[FW] Loading case {matrix_name} + {vector_name}...", flush=True)
        problem = load_problem(
            args.data_folder,
            matrix_name,
            vector_name,
            metadata_path=args.metadata,
            num_blocks=args.num_blocks,
            write_inferred_metadata=args.write_metadata,
        )
        for x0_key in pending_x0_keys:
            x0 = (
                load_initial_point(args.x0_file, key=x0_key)
                if args.x0_file
                else "barycenter"
            )
            config = FrankWolfeConfig(
                max_iter=args.max_iter,
                tol_gap=args.tol_gap,
                x0=x0,
                line_search="exact",
                store_history=True,
                verbose=not args.quiet,
                progress_every=args.progress_every,
            )
            result = solve_frank_wolfe(problem, config)
            record = {
                "solver": "frank_wolfe",
                "matrix": matrix_name,
                "vector": vector_name,
                "x0_key": x0_key,
                **result.to_dict(include_solution=args.include_solution),
            }
            records.append(record)
            completed_runs.add(_record_key(record))
            _write_records(output_path, records)
            print(
                f"[FW] {matrix_name} + {vector_name} | x0={x0_key}: "
                f"status={result.status}, objective={result.objective:.6e}, "
                f"gap={result.gap:.3e}, iterations={result.iterations}, "
                f"runtime={result.runtime_seconds:.4f}s",
                flush=True,
            )
            print(
                f"[FW] Checkpoint saved: {len(records)} result(s) in {output_path}",
                flush=True,
            )

    print(f"Saved {len(records)} Frank-Wolfe result(s) to {output_path}", flush=True)


def parse_case(raw_case: str) -> tuple[str, str]:
    if ":" not in raw_case:
        raise ValueError(
            f"Invalid case '{raw_case}'. Expected MATRIX_NAME:VECTOR_NAME."
        )
    matrix_name, vector_name = raw_case.split(":", maxsplit=1)
    return matrix_name, vector_name


def _resolve_x0_keys(
    x0_file: str | None,
    x0_keys: list[str] | None,
    all_x0: bool,
) -> list[str]:
    if all_x0:
        if x0_file is None:
            raise ValueError("--all-x0 requires --x0-file.")
        return load_initial_point_keys(x0_file)
    if x0_keys:
        return x0_keys
    return ["barycenter"]


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")

    records: list[dict[str, Any]] = []
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError(f"Expected every record in {path} to be a JSON object.")
        records.append(record)
    return records


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("matrix")),
        str(record.get("vector")),
        str(record.get("x0_key", "barycenter")),
    )


if __name__ == "__main__":
    main()
