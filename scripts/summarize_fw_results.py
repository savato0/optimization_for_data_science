from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

BASE_TABLE_COLUMNS = [
    "matrix",
    "vector",
    "x0_key",
    "status",
    "f_iter1",
    "alpha_iter0",
    "objective",
    "fw_gap",
    "iterations",
    "runtime_seconds",
]
TARGET_TABLE_COLUMNS = [
    "target_key",
    "distance_to_xu",
]
GUROBI_TABLE_COLUMNS = [
    "gurobi_status",
    "gurobi_objective",
    "objective_difference",
]
SORTABLE_COLUMNS = BASE_TABLE_COLUMNS + TARGET_TABLE_COLUMNS + GUROBI_TABLE_COLUMNS + ["lower_bound"]
HISTORY_COLUMNS = [
    "matrix",
    "vector",
    "x0_key",
    "iteration",
    "objective",
    "fw_gap",
    "lower_bound",
    "alpha",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Frank-Wolfe result JSON files.")
    parser.add_argument("results_json", help="Path to a fw_results.json-style file.")
    parser.add_argument(
        "--sort-by",
        choices=SORTABLE_COLUMNS,
        default="iterations",
        help="Column used to sort the table.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort from largest to smallest.",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV output path for the flattened summary table.",
    )
    parser.add_argument(
        "--output",
        help="Optional text output path for the rendered summary or history table.",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Only include this case in the form MATRIX_NAME:VECTOR_NAME. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--x0-key",
        action="append",
        help="Only include this initial point key. Repeat for multiple keys.",
    )
    parser.add_argument(
        "--targets-file",
        help=(
            "Optional targets.npz file containing x_u_sc1, x_u_sc2, x_u_sc3. "
            "When result records include x, the summary reports ||x - x_u||."
        ),
    )
    parser.add_argument(
        "--gurobi-file",
        help=(
            "Optional gurobi_baseline.json file. The summary reports the matching Gurobi "
            "objective and objective_FW - objective_Gurobi."
        ),
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print per-iteration history rows instead of final summary rows.",
    )
    parser.add_argument(
        "--history-for",
        action="append",
        help="Only print history for this x0_key. Repeat for multiple keys.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(
        args.results_json,
        targets_file=args.targets_file,
        gurobi_file=args.gurobi_file,
    )
    records = filter_records(records, cases=args.case, x0_keys=args.x0_key)

    if args.history or args.history_for:
        history_rows = flatten_history(records, x0_keys=args.history_for)
        output = format_history_table(history_rows)
        emit_output(output, output_path=args.output)
        if args.csv:
            write_history_csv(args.csv, history_rows)
            print(f"\nSaved CSV history to {args.csv}", flush=True)
        return

    records = sort_records(records, sort_by=args.sort_by, descending=args.descending)

    output = format_summary(records) + format_table(records)
    emit_output(output, output_path=args.output)

    if args.csv:
        write_csv(args.csv, records)
        print(f"\nSaved CSV summary to {args.csv}", flush=True)


def load_records(
    path: str | Path,
    *,
    targets_file: str | Path | None = None,
    gurobi_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list of result records.")
    targets = load_targets(targets_file) if targets_file is not None else None
    gurobi_baselines = (
        load_gurobi_baselines(gurobi_file) if gurobi_file is not None else None
    )
    return [
        _add_gurobi_fields(
            _add_target_fields(_add_derived_fields(_flatten_record(record)), targets),
            gurobi_baselines,
        )
        for record in payload
    ]


def load_targets(path: str | Path) -> dict[str, np.ndarray]:
    archive = np.load(Path(path))
    try:
        return {
            key: np.asarray(archive[key], dtype=float).reshape(-1)
            for key in archive.files
        }
    finally:
        archive.close()


def load_gurobi_baselines(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list of Gurobi baseline records.")

    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_record in payload:
        if not isinstance(raw_record, dict):
            raise ValueError("Each Gurobi baseline record must be a JSON object.")
        matrix = raw_record.get("matrix")
        vector = raw_record.get("vector")
        if not isinstance(matrix, str) or not isinstance(vector, str):
            raise ValueError("Each Gurobi baseline record must contain matrix and vector.")

        baseline = raw_record.get("gurobi", raw_record)
        if not isinstance(baseline, dict):
            raise ValueError("Expected 'gurobi' to contain a JSON object.")
        baselines[(matrix, vector)] = baseline
    return baselines


def sort_records(
    records: list[dict[str, Any]],
    *,
    sort_by: str,
    descending: bool,
) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: _sort_value(record.get(sort_by)),
        reverse=descending,
    )


def filter_records(
    records: list[dict[str, Any]],
    *,
    cases: list[str] | None = None,
    x0_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_cases = {_parse_case_filter(raw_case) for raw_case in cases or []}
    selected_x0_keys = set(x0_keys or [])

    filtered: list[dict[str, Any]] = []
    for record in records:
        matrix = record.get("matrix")
        vector = record.get("vector")
        x0_key = str(record.get("x0_key", "barycenter"))

        if selected_cases and (matrix, vector) not in selected_cases:
            continue
        if selected_x0_keys and x0_key not in selected_x0_keys:
            continue
        filtered.append(record)
    return filtered


def format_summary(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No records found."

    converged = sum(1 for record in records if record.get("status") == "converged")
    failed = len(records) - converged
    best_objective = min(records, key=lambda record: _sort_value(record.get("objective")))
    fastest = min(records, key=lambda record: _sort_value(record.get("runtime_seconds")))
    fewest_iterations = min(records, key=lambda record: _sort_value(record.get("iterations")))

    lines = [
        f"Records: {len(records)}",
        f"Converged: {converged}",
        f"Not converged: {failed}",
        (
            "Best objective: "
            f"{_label(best_objective)} -> {_format_float(best_objective.get('objective'))}"
        ),
        (
            "Fewest iterations: "
            f"{_label(fewest_iterations)} -> {fewest_iterations.get('iterations')}"
        ),
        (
            "Fastest runtime: "
            f"{_label(fastest)} -> {_format_float(fastest.get('runtime_seconds'))}s"
        ),
    ]
    return "\n".join(lines)


def format_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""

    columns = _table_columns(records)
    rows = [[_format_value(record.get(column)) for column in columns] for record in records]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    header = "  ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(columns)))
        for row in rows
    ]
    return "\n".join(["", header, separator, *body])


def flatten_history(
    records: list[dict[str, Any]],
    *,
    x0_keys: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_keys = set(x0_keys or [])
    rows: list[dict[str, Any]] = []
    for record in records:
        x0_key = str(record.get("x0_key", "barycenter"))
        if selected_keys and x0_key not in selected_keys:
            continue
        history = record.get("history", [])
        if not isinstance(history, list):
            raise ValueError("Expected 'history' to be a list.")
        for entry in history:
            if not isinstance(entry, dict):
                raise ValueError("Each history entry must be a JSON object.")
            rows.append(
                {
                    "matrix": record.get("matrix"),
                    "vector": record.get("vector"),
                    "x0_key": x0_key,
                    "iteration": entry.get("iteration"),
                    "objective": entry.get("objective"),
                    "fw_gap": entry.get("fw_gap"),
                    "lower_bound": entry.get("lower_bound"),
                    "alpha": entry.get("alpha"),
                }
            )
    return rows


def format_history_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No history rows found."
    return _format_records_as_table(rows, HISTORY_COLUMNS)


def write_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = _table_columns(records)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in columns})


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")


def emit_output(text: str, *, output_path: str | Path | None = None) -> None:
    if output_path is None:
        print(text, flush=True)
        return
    write_text(output_path, text)
    print(f"Saved text output to {output_path}", flush=True)


def write_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HISTORY_COLUMNS})


def _flatten_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Each result record must be a JSON object.")
    if "frank_wolfe" not in record:
        return record

    fw_record = record["frank_wolfe"]
    if not isinstance(fw_record, dict):
        raise ValueError("Expected 'frank_wolfe' to contain a JSON object.")
    flattened = {
        "matrix": record.get("matrix"),
        "vector": record.get("vector"),
        "x0_key": record.get("x0_key"),
        **fw_record,
    }
    return flattened


def _parse_case_filter(raw_case: str) -> tuple[str, str]:
    if ":" not in raw_case:
        raise ValueError(
            f"Invalid case '{raw_case}'. Expected MATRIX_NAME:VECTOR_NAME."
        )
    matrix_name, vector_name = raw_case.split(":", maxsplit=1)
    return matrix_name, vector_name


def _add_derived_fields(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    record["f_iter1"] = _history_value(record, iteration=1, field="objective")
    record["alpha_iter0"] = _history_value(record, iteration=0, field="alpha")
    return record


def _add_target_fields(
    record: dict[str, Any],
    targets: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    if targets is None:
        return record

    record = dict(record)
    target_key = target_key_from_vector(record.get("vector"))
    record["target_key"] = target_key
    record["distance_to_xu"] = None
    if target_key is None:
        return record
    if target_key not in targets:
        available = ", ".join(targets)
        raise KeyError(f"Target '{target_key}' was not found. Available: {available}")

    solution = record.get("x")
    if solution is None:
        return record

    x = np.asarray(solution, dtype=float).reshape(-1)
    x_u = targets[target_key]
    if x.size != x_u.size:
        raise ValueError(
            f"Solution dimension {x.size} does not match target '{target_key}' dimension {x_u.size}."
        )
    record["distance_to_xu"] = float(np.linalg.norm(x - x_u))
    return record


def _add_gurobi_fields(
    record: dict[str, Any],
    baselines: dict[tuple[str, str], dict[str, Any]] | None,
) -> dict[str, Any]:
    if baselines is None:
        return record

    record = dict(record)
    matrix = record.get("matrix")
    vector = record.get("vector")
    if not isinstance(matrix, str) or not isinstance(vector, str):
        record["gurobi_status"] = None
        record["gurobi_objective"] = None
        record["objective_difference"] = None
        return record

    baseline = baselines.get((matrix, vector))
    if baseline is None:
        record["gurobi_status"] = None
        record["gurobi_objective"] = None
        record["objective_difference"] = None
        return record

    gurobi_objective = baseline.get("objective")
    record["gurobi_status"] = baseline.get("status")
    record["gurobi_objective"] = gurobi_objective
    record["objective_difference"] = None
    if isinstance(record.get("objective"), (int, float)) and isinstance(
        gurobi_objective,
        (int, float),
    ):
        record["objective_difference"] = float(record["objective"] - gurobi_objective)
    return record


def target_key_from_vector(vector_name: Any) -> str | None:
    if not isinstance(vector_name, str):
        return None
    for part in reversed(vector_name.split("_")):
        if len(part) > 2 and part.startswith("sc") and part[2:].isdigit():
            return f"x_u_{part}"
    return None


def _history_value(record: dict[str, Any], *, iteration: int, field: str) -> Any:
    history = record.get("history", [])
    if not isinstance(history, list):
        return None
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("iteration") == iteration:
            return entry.get(field)
    return None


def _label(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("matrix", "-")),
        str(record.get("vector", "-")),
        str(record.get("x0_key", "-")),
    ]
    return " / ".join(parts)


def _sort_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return (1, "")
    return (0, value)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _format_float(value: Any) -> str:
    if not isinstance(value, float):
        return _format_value(value)
    return f"{value:.6e}"


def _table_columns(records: list[dict[str, Any]]) -> list[str]:
    has_target_fields = any(
        "target_key" in record or "distance_to_xu" in record
        for record in records
    )
    has_gurobi_fields = any(
        "gurobi_status" in record
        or "gurobi_objective" in record
        or "objective_difference" in record
        for record in records
    )
    columns = list(BASE_TABLE_COLUMNS)
    if has_target_fields:
        columns.extend(TARGET_TABLE_COLUMNS)
    if has_gurobi_fields:
        columns.extend(GUROBI_TABLE_COLUMNS)
    return columns


def _format_records_as_table(records: list[dict[str, Any]], columns: list[str]) -> str:
    rows = [[_format_value(record.get(column)) for column in columns] for record in records]
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    header = "  ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(columns)))
        for row in rows
    ]
    return "\n".join([header, separator, *body])


if __name__ == "__main__":
    main()
