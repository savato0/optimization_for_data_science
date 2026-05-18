from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


TARGET_SUFFIX_PATTERN = re.compile(r"(sc\d+)$")


@dataclass(frozen=True, slots=True)
class TraceProjector:
    """Project high-dimensional FW iterates onto a fixed 2D diagnostic plane."""

    origin: np.ndarray
    axis_1: np.ndarray
    axis_2: np.ndarray
    reference: np.ndarray | None = None
    xu: np.ndarray | None = None
    axis_1_label: str = "axis_1"
    axis_2_label: str = "axis_2"

    def project(self, point: np.ndarray) -> dict[str, float]:
        x = np.asarray(point, dtype=float).reshape(-1)
        _validate_dimension("point", x, self.origin.size)

        delta = x - self.origin
        payload = {
            "coord_1": float(delta @ self.axis_1),
            "coord_2": float(delta @ self.axis_2),
        }
        if self.reference is not None:
            payload["distance_to_reference"] = float(np.linalg.norm(x - self.reference))
        if self.xu is not None:
            payload["distance_to_xu"] = float(np.linalg.norm(x - self.xu))
        return payload

    def metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "axis_1_label": self.axis_1_label,
            "axis_2_label": self.axis_2_label,
            "origin": {"label": "x0", "coord_1": 0.0, "coord_2": 0.0},
        }
        if self.reference is not None:
            coords = self.project(self.reference)
            payload["reference"] = {
                "label": "x_gurobi",
                "coord_1": coords["coord_1"],
                "coord_2": coords["coord_2"],
            }
        if self.xu is not None:
            coords = self.project(self.xu)
            payload["xu"] = {
                "label": "x_u",
                "coord_1": coords["coord_1"],
                "coord_2": coords["coord_2"],
            }
        return payload


def make_trace_projector(
    x0: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    xu: np.ndarray | None = None,
) -> TraceProjector:
    origin = np.asarray(x0, dtype=float).reshape(-1)
    if reference is not None:
        reference = np.asarray(reference, dtype=float).reshape(-1)
        _validate_dimension("reference", reference, origin.size)
    if xu is not None:
        xu = np.asarray(xu, dtype=float).reshape(-1)
        _validate_dimension("xu", xu, origin.size)

    if reference is not None:
        axis_1 = _normalize(reference - origin, "x_gurobi - x0")
        axis_1_label = "x_gurobi - x0"
        axis_2_source = xu - origin if xu is not None else None
        axis_2_label = (
            "orthogonal component of x_u - x0"
            if xu is not None
            else "deterministic orthogonal axis"
        )
    elif xu is not None:
        axis_1 = _normalize(xu - origin, "x_u - x0")
        axis_1_label = "x_u - x0"
        axis_2_source = None
        axis_2_label = "deterministic orthogonal axis"
    else:
        raise ValueError("A trace projector requires x_gurobi and/or x_u as a reference.")

    axis_2 = _orthogonal_axis(axis_1, axis_2_source)
    return TraceProjector(
        origin=origin,
        axis_1=axis_1,
        axis_2=axis_2,
        reference=reference,
        xu=xu,
        axis_1_label=axis_1_label,
        axis_2_label=axis_2_label,
    )


def load_target_for_case(targets_file: str | Path, vector_name: str) -> tuple[str, np.ndarray]:
    target_key = target_key_for_vector(vector_name)
    archive = np.load(Path(targets_file))
    try:
        values = archive[target_key]
    except KeyError as exc:
        available = ", ".join(archive.files)
        raise KeyError(
            f"Target '{target_key}' was not found in {targets_file}. Available: {available}"
        ) from exc
    return target_key, np.asarray(values, dtype=float).reshape(-1)


def target_key_for_vector(vector_name: str) -> str:
    match = TARGET_SUFFIX_PATTERN.search(vector_name)
    if match is None:
        raise ValueError(
            f"Cannot infer target key from vector '{vector_name}'. Expected a suffix like sc1."
        )
    return f"x_u_{match.group(1)}"


def load_gurobi_solution(
    gurobi_file: str | Path,
    matrix_name: str,
    vector_name: str,
) -> np.ndarray | None:
    payload = json.loads(Path(gurobi_file).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {gurobi_file} to contain a JSON list.")

    for record in payload:
        if not isinstance(record, dict):
            continue
        if record.get("matrix") != matrix_name or record.get("vector") != vector_name:
            continue
        solution = _solution_from_record(record)
        if solution is None:
            return None
        return np.asarray(solution, dtype=float).reshape(-1)
    return None


def _solution_from_record(record: dict[str, Any]) -> Any:
    if "x" in record:
        return record["x"]
    gurobi_payload = record.get("gurobi")
    if isinstance(gurobi_payload, dict):
        return gurobi_payload.get("x")
    return None


def _orthogonal_axis(axis_1: np.ndarray, source: np.ndarray | None) -> np.ndarray:
    if source is not None:
        residual = source - float(source @ axis_1) * axis_1
        norm = np.linalg.norm(residual)
        if norm > 1e-12:
            return residual / norm

    basis = np.zeros_like(axis_1)
    basis[int(np.argmin(np.abs(axis_1)))] = 1.0
    residual = basis - float(basis @ axis_1) * axis_1
    return _normalize(residual, "deterministic orthogonal axis")


def _normalize(values: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(values)
    if norm <= 1e-12:
        raise ValueError(f"Cannot build trace axis from near-zero vector: {label}.")
    return values / norm


def _validate_dimension(label: str, values: np.ndarray, expected: int) -> None:
    if values.size != expected:
        raise ValueError(f"{label} has dimension {values.size}; expected {expected}.")
