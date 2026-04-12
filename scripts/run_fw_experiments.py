from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simplex_qp import FrankWolfeConfig, load_problem, solve_frank_wolfe


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
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument(
        "--include-solution",
        action="store_true",
        help="Include the full solution vector in the JSON payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FrankWolfeConfig(
        max_iter=args.max_iter,
        tol_gap=args.tol_gap,
        x0="barycenter",
        line_search="exact",
        store_history=True,
    )

    records: list[dict[str, object]] = []
    for raw_case in args.case:
        matrix_name, vector_name = parse_case(raw_case)
        problem = load_problem(
            args.data_folder,
            matrix_name,
            vector_name,
            metadata_path=args.metadata,
            num_blocks=args.num_blocks,
            write_inferred_metadata=args.write_metadata,
        )
        result = solve_frank_wolfe(problem, config)
        record = {
            "solver": "frank_wolfe",
            "matrix": matrix_name,
            "vector": vector_name,
            **result.to_dict(include_solution=args.include_solution),
        }
        records.append(record)
        print(
            f"[FW] {matrix_name} + {vector_name}: status={result.status}, "
            f"objective={result.objective:.6e}, gap={result.gap:.3e}, "
            f"iterations={result.iterations}, runtime={result.runtime_seconds:.4f}s"
        )

    output_path = Path(args.output) if args.output else Path(args.data_folder) / "fw_results.json"
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} Frank-Wolfe result(s) to {output_path}")


def parse_case(raw_case: str) -> tuple[str, str]:
    if ":" not in raw_case:
        raise ValueError(
            f"Invalid case '{raw_case}'. Expected MATRIX_NAME:VECTOR_NAME."
        )
    matrix_name, vector_name = raw_case.split(":", maxsplit=1)
    return matrix_name, vector_name


if __name__ == "__main__":
    main()
