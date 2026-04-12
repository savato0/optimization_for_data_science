from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .oracle import linear_minimization_oracle
from .problem import SimplexQP

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised via skip paths in tests.
    gp = None
    GRB = None


@dataclass(slots=True)
class BaselineResult:
    x: np.ndarray | None
    objective: float | None
    fw_gap: float | None
    runtime_seconds: float
    status: str
    status_code: int | None
    feasibility: dict[str, float] | None

    def to_dict(self, *, include_solution: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.objective is not None:
            payload["objective"] = self.objective
        if self.fw_gap is not None:
            payload["fw_gap"] = self.fw_gap
        if self.feasibility is not None:
            payload.update(self.feasibility)
        if include_solution and self.x is not None:
            payload["x"] = self.x.tolist()
        return payload


def solve_gurobi(
    problem: SimplexQP,
    *,
    log_file: str | None = None,
    output_flag: bool = False,
) -> BaselineResult:
    if gp is None or GRB is None:
        raise ImportError("gurobipy is not installed in the active Python environment.")

    model = gp.Model(problem.name)
    model.Params.OutputFlag = 1 if output_flag else 0
    if log_file is not None:
        model.setParam("LogFile", log_file)

    x = model.addMVar(shape=problem.dimension, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
    objective = x @ problem.Q @ x + problem.q @ x
    model.setObjective(objective, GRB.MINIMIZE)

    for block_id, block in enumerate(problem.partition.blocks):
        model.addConstr(x[block].sum() == 1.0, name=f"simplex_block_{block_id}")

    start = perf_counter()
    model.optimize()
    runtime = perf_counter() - start

    if model.Status != GRB.OPTIMAL:
        return BaselineResult(
            x=None,
            objective=None,
            fw_gap=None,
            runtime_seconds=runtime,
            status="not_optimal",
            status_code=int(model.Status),
            feasibility=None,
        )

    solution = np.asarray(x.X, dtype=float)
    gradient = problem.gradient(solution)
    fw_vertex = linear_minimization_oracle(gradient, problem.partition)
    fw_gap = float(gradient @ (solution - fw_vertex))

    return BaselineResult(
        x=solution,
        objective=float(model.ObjVal),
        fw_gap=fw_gap,
        runtime_seconds=runtime,
        status="optimal",
        status_code=int(model.Status),
        feasibility=problem.feasibility_metrics(solution),
    )
