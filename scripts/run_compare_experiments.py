from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simplex_qp import (
    FrankWolfeConfig,
    default_results_folder,
    load_initial_point,
    load_initial_point_keys,
    load_problem,
    solve_frank_wolfe,
    solve_gurobi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Frank-Wolfe against the Gurobi baseline.")
    parser.add_argument(
        "data_folder",
        help="Problem folder containing data/ or the data folder containing matrices.npz and vectors.npz.",
    )
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
        "--tol-rel-gap",
        type=float,
        help=(
            "Optional relative Frank-Wolfe gap tolerance. "
            "Converges when fw_gap/max(1, abs(objective)) is below this value."
        ),
    )
    parser.add_argument(
        "--x0-file",
        help="Optional Frank-Wolfe initial_points.npz file.",
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
        "--include-solution",
        action="store_true",
        help="Include the full Frank-Wolfe solution vector in the JSON payload.",
    )
    parser.add_argument("--gurobi-log-dir", help="Optional directory for Gurobi log files.")
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

    records: list[dict[str, object]] = []
    log_dir = Path(args.gurobi_log_dir) if args.gurobi_log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    for raw_case in args.case:
        matrix_name, vector_name = parse_case(raw_case)
        print(f"[COMPARE] Loading case {matrix_name} + {vector_name}...", flush=True)
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
        baseline_result = solve_gurobi(problem, log_file=log_file)

        gurobi_objective = (
            f"{baseline_result.objective:.6e}"
            if baseline_result.objective is not None
            else "nan"
        )
        for x0_key in x0_keys:
            x0 = (
                load_initial_point(args.x0_file, key=x0_key)
                if args.x0_file
                else "barycenter"
            )
            config = FrankWolfeConfig(
                max_iter=args.max_iter,
                tol_gap=args.tol_gap,
                tol_rel_gap=args.tol_rel_gap,
                x0=x0,
                line_search="exact",
                store_history=True,
                verbose=not args.quiet,
                progress_every=args.progress_every,
            )
            fw_result = solve_frank_wolfe(problem, config)
            fw_payload = fw_result.to_dict(include_solution=args.include_solution)
            fw_payload["x0_key"] = x0_key

            record = {
                "matrix": matrix_name,
                "vector": vector_name,
                "x0_key": x0_key,
                "frank_wolfe": fw_payload,
                "gurobi": baseline_result.to_dict(),
            }
            if baseline_result.objective is not None:
                record["objective_difference"] = fw_result.objective - baseline_result.objective
            records.append(record)

            print(
                f"[COMPARE] {matrix_name} + {vector_name} | x0={x0_key}: "
                f"FW={fw_result.objective:.6e}, "
                f"Gurobi={gurobi_objective}",
                flush=True,
            )

    output_path = (
        Path(args.output)
        if args.output
        else default_results_folder(args.data_folder) / "comparison_results.json"
    )
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} comparison result(s) to {output_path}", flush=True)


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


if __name__ == "__main__":
    main()
