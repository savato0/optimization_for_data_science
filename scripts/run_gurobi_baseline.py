from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simplex_qp import load_problem, solve_gurobi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the Gurobi baseline on saved QP cases.")
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
    parser.add_argument(
        "--include-solution",
        action="store_true",
        help="Include the full Gurobi solution vector in the JSON payload.",
    )
    parser.add_argument(
        "--gurobi-log-dir",
        help="Optional directory for Gurobi log files.",
    )
    parser.add_argument(
        "--gurobi-method",
        type=int,
        default=0,
        help="Gurobi Method parameter. Defaults to 0.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []
    log_dir = Path(args.gurobi_log_dir) if args.gurobi_log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    for raw_case in args.case:
        matrix_name, vector_name = parse_case(raw_case)
        print(f"[GUROBI] Loading case {matrix_name} + {vector_name}...", flush=True)
        problem = load_problem(
            args.data_folder,
            matrix_name,
            vector_name,
            metadata_path=args.metadata,
            num_blocks=args.num_blocks,
            write_inferred_metadata=args.write_metadata,
        )

        log_file = None
        if log_dir is not None:
            log_file = str(log_dir / f"{matrix_name}_{vector_name}_gurobi.log")
        result = solve_gurobi(problem, log_file=log_file, method=args.gurobi_method)
        record = {
            "solver": "gurobi",
            "matrix": matrix_name,
            "vector": vector_name,
            **result.to_dict(include_solution=args.include_solution),
        }
        records.append(record)

        objective = f"{result.objective:.6e}" if result.objective is not None else "nan"
        print(
            f"[GUROBI] {matrix_name} + {vector_name}: "
            f"status={result.status}, objective={objective}, "
            f"runtime={result.runtime_seconds:.4f}s",
            flush=True,
        )

    output_path = (
        Path(args.output) if args.output else Path(args.data_folder) / "gurobi_baseline.json"
    )
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} Gurobi baseline result(s) to {output_path}", flush=True)


def parse_case(raw_case: str) -> tuple[str, str]:
    if ":" not in raw_case:
        raise ValueError(
            f"Invalid case '{raw_case}'. Expected MATRIX_NAME:VECTOR_NAME."
        )
    matrix_name, vector_name = raw_case.split(":", maxsplit=1)
    return matrix_name, vector_name


if __name__ == "__main__":
    main()
