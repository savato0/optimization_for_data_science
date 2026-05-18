from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .line_search import exact_line_search
from .oracle import linear_minimization_oracle
from .problem import SimplexQP
from .trace import TraceProjector


@dataclass(slots=True)
class FrankWolfeConfig:
    max_iter: int = 5000
    tol_gap: float = 1e-6
    x0: str | np.ndarray = "barycenter"
    line_search: str = "exact"
    store_history: bool = True
    verbose: bool = False
    progress_every: int = 50
    trace_projector: TraceProjector | None = None
    trace_every: int = 1


@dataclass(slots=True)
class FrankWolfeResult:
    x: np.ndarray
    objective: float
    gap: float
    lower_bound: float
    iterations: int
    runtime_seconds: float
    status: str
    feasibility: dict[str, float]
    history: list[dict[str, float]] | None = None
    projected_trace: list[dict[str, float]] | None = None

    def to_dict(
        self,
        *,
        include_history: bool = True,
        include_solution: bool = False,
        include_projected_trace: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "objective": self.objective,
            "fw_gap": self.gap,
            "lower_bound": self.lower_bound,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            **self.feasibility,
        }
        if include_history and self.history is not None:
            payload["history"] = self.history
        if include_projected_trace and self.projected_trace is not None:
            payload["projected_trace"] = self.projected_trace
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
    if cfg.progress_every <= 0:
        raise ValueError("progress_every must be strictly positive.")
    if cfg.trace_every <= 0:
        raise ValueError("trace_every must be strictly positive.")

    x = _initial_point(problem, cfg.x0)
    history: list[dict[str, float]] | None = [] if cfg.store_history else None
    projected_trace: list[dict[str, float]] | None = (
        [] if cfg.trace_projector is not None else None
    )
    updates = 0
    status = "max_iter"

    start = perf_counter()
    if cfg.verbose:
        print(
            f"[FW] Starting {problem.name}: n={problem.dimension}, "
            f"K={problem.partition.K}, max_iter={cfg.max_iter}, tol_gap={cfg.tol_gap:.1e}",
            flush=True,
        )
    for iteration in range(cfg.max_iter):
        gradient = problem.gradient(x)
        s_t = linear_minimization_oracle(gradient, problem.partition)
        d_t = s_t - x
        g_t = float(gradient @ (x - s_t))
        objective = problem.objective(x)
        v_t = objective - g_t
        alpha_t = 0.0
        feasibility = problem.feasibility_metrics(x)

        if g_t <= cfg.tol_gap:
            status = "converged"
            _append_history(history, iteration, objective, g_t, v_t, alpha_t, feasibility)
            _append_projected_trace(
                projected_trace,
                cfg,
                iteration,
                x,
                objective,
                g_t,
                v_t,
                alpha_t,
                force=True,
            )
            break

        alpha_t = exact_line_search(
            problem,
            x,
            d_t,
            gradient=gradient,
        )
        _log_progress(
            problem,
            cfg,
            iteration=iteration,
            objective=objective,
            gap=g_t,
            lower_bound=v_t,
            alpha=alpha_t,
            feasibility=feasibility,
            elapsed_seconds=perf_counter() - start,
            is_terminal=False,
        )
        _append_history(history, iteration, objective, g_t, v_t, alpha_t, feasibility)
        _append_projected_trace(
            projected_trace,
            cfg,
            iteration,
            x,
            objective,
            g_t,
            v_t,
            alpha_t,
        )
        x = x + alpha_t * d_t
        updates += 1
    else:
        gradient = problem.gradient(x)
        s_t = linear_minimization_oracle(gradient, problem.partition)
        g_t = float(gradient @ (x - s_t))
        objective = problem.objective(x)
        v_t = objective - g_t
        feasibility = problem.feasibility_metrics(x)
        _log_progress(
            problem,
            cfg,
            iteration=cfg.max_iter,
            objective=objective,
            gap=g_t,
            lower_bound=v_t,
            alpha=0.0,
            feasibility=feasibility,
            elapsed_seconds=perf_counter() - start,
            is_terminal=True,
        )
        _append_history(history, cfg.max_iter, objective, g_t, v_t, 0.0, feasibility)
        _append_projected_trace(
            projected_trace,
            cfg,
            cfg.max_iter,
            x,
            objective,
            g_t,
            v_t,
            0.0,
            force=True,
        )

    runtime = perf_counter() - start
    gradient = problem.gradient(x)
    s_t = linear_minimization_oracle(gradient, problem.partition)
    g_t = float(gradient @ (x - s_t))
    objective = problem.objective(x)
    v_t = objective - g_t
    feasibility = problem.feasibility_metrics(x)
    if cfg.verbose and status == "converged":
        _log_progress(
            problem,
            cfg,
            iteration=updates,
            objective=objective,
            gap=g_t,
            lower_bound=v_t,
            alpha=0.0,
            feasibility=feasibility,
            elapsed_seconds=runtime,
            is_terminal=True,
        )

    return FrankWolfeResult(
        x=x,
        objective=objective,
        gap=g_t,
        lower_bound=v_t,
        iterations=updates,
        runtime_seconds=runtime,
        status=status,
        feasibility=feasibility,
        history=history,
        projected_trace=projected_trace,
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


def _log_progress(
    problem: SimplexQP,
    cfg: FrankWolfeConfig,
    *,
    iteration: int,
    objective: float,
    gap: float,
    lower_bound: float,
    alpha: float,
    feasibility: dict[str, float],
    elapsed_seconds: float,
    is_terminal: bool,
) -> None:
    if not cfg.verbose:
        return

    should_print = is_terminal or iteration == 0 or (iteration + 1) % cfg.progress_every == 0
    if not should_print:
        return

    label = "final" if is_terminal else f"iter {iteration + 1}"
    print(
        f"[FW] {problem.name} | {label}: "
        f"f={objective:.6e}, gap={gap:.3e}, lower_bound={lower_bound:.6e}, "
        f"alpha={alpha:.3e}, "
        f"block_err={feasibility['max_block_sum_error']:.1e}, "
        f"nonneg_err={feasibility['max_nonnegativity_violation']:.1e}, "
        f"elapsed={elapsed_seconds:.2f}s",
        flush=True,
    )


def _append_history(
    history: list[dict[str, float]] | None,
    iteration: int,
    objective: float,
    gap: float,
    lower_bound: float,
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
            "lower_bound": lower_bound,
            "alpha": step_size,
            **feasibility,
        }
    )


def _append_projected_trace(
    projected_trace: list[dict[str, float]] | None,
    cfg: FrankWolfeConfig,
    iteration: int,
    x: np.ndarray,
    objective: float,
    gap: float,
    lower_bound: float,
    step_size: float,
    *,
    force: bool = False,
) -> None:
    if projected_trace is None or cfg.trace_projector is None:
        return
    if not force and iteration % cfg.trace_every != 0:
        return

    projected_trace.append(
        {
            "iteration": float(iteration),
            "objective": objective,
            "fw_gap": gap,
            "lower_bound": lower_bound,
            "alpha": step_size,
            **cfg.trace_projector.project(x),
        }
    )
