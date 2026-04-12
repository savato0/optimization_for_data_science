from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .line_search import exact_line_search
from .oracle import linear_minimization_oracle
from .problem import SimplexQP


@dataclass(slots=True)
class FrankWolfeConfig:
    max_iter: int = 5000
    tol_gap: float = 1e-6
    x0: str | np.ndarray = "barycenter"
    line_search: str = "exact"
    store_history: bool = True


@dataclass(slots=True)
class FrankWolfeResult:
    x: np.ndarray
    objective: float
    gap: float
    iterations: int
    runtime_seconds: float
    status: str
    feasibility: dict[str, float]
    history: list[dict[str, float]] | None = None

    def to_dict(
        self,
        *,
        include_history: bool = True,
        include_solution: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "objective": self.objective,
            "fw_gap": self.gap,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            **self.feasibility,
        }
        if include_history and self.history is not None:
            payload["history"] = self.history
        if include_solution:
            payload["x"] = self.x.tolist()
        return payload


def solve_frank_wolfe(
    problem: SimplexQP,
    config: FrankWolfeConfig | None = None,
) -> FrankWolfeResult:
    cfg = FrankWolfeConfig() if config is None else config
    if cfg.line_search != "exact":
        raise ValueError("Only exact line search is implemented in v1.")

    x = _initial_point(problem, cfg.x0)
    history: list[dict[str, float]] | None = [] if cfg.store_history else None
    updates = 0
    status = "max_iter"

    start = perf_counter()
    for iteration in range(cfg.max_iter):
        gradient = problem.gradient(x)
        s = linear_minimization_oracle(gradient, problem.partition)
        direction = s - x
        gap = float(gradient @ (x - s))
        objective = problem.objective(x)
        step_size = 0.0
        feasibility = problem.feasibility_metrics(x)

        if gap <= cfg.tol_gap:
            status = "converged"
            _append_history(history, iteration, objective, gap, step_size, feasibility)
            break

        step_size = exact_line_search(
            problem,
            x,
            direction,
            gradient=gradient,
        )
        _append_history(history, iteration, objective, gap, step_size, feasibility)
        x = x + step_size * direction
        updates += 1
    else:
        gradient = problem.gradient(x)
        s = linear_minimization_oracle(gradient, problem.partition)
        gap = float(gradient @ (x - s))
        objective = problem.objective(x)
        feasibility = problem.feasibility_metrics(x)
        _append_history(history, cfg.max_iter, objective, gap, 0.0, feasibility)

    runtime = perf_counter() - start
    gradient = problem.gradient(x)
    s = linear_minimization_oracle(gradient, problem.partition)
    gap = float(gradient @ (x - s))
    objective = problem.objective(x)
    feasibility = problem.feasibility_metrics(x)

    return FrankWolfeResult(
        x=x,
        objective=objective,
        gap=gap,
        iterations=updates,
        runtime_seconds=runtime,
        status=status,
        feasibility=feasibility,
        history=history,
    )


def _initial_point(problem: SimplexQP, x0: str | np.ndarray) -> np.ndarray:
    if isinstance(x0, str):
        if x0 != "barycenter":
            raise ValueError("Only x0='barycenter' is implemented in v1.")
        x = problem.barycenter()
    else:
        x = np.asarray(x0, dtype=float).reshape(-1)

    if not problem.is_feasible(x):
        raise ValueError("The initial point must be feasible for the product of simplices.")
    return x.copy()


def _append_history(
    history: list[dict[str, float]] | None,
    iteration: int,
    objective: float,
    gap: float,
    step_size: float,
    feasibility: dict[str, float],
) -> None:
    if history is None:
        return
    history.append(
        {
            "iteration": float(iteration),
            "objective": objective,
            "fw_gap": gap,
            "step_size": step_size,
            **feasibility,
        }
    )
