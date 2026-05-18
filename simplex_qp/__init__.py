"""Utilities for convex quadratic programs over products of simplices."""

from .baseline_gurobi import BaselineResult, solve_gurobi
from .experiment_config import (
    DEFAULT_PROBLEM_CONFIG_PATH,
    InitialPointsConfig,
    ProblemConfig,
    load_problem_config,
    partition_from_config,
)
from .fw import FrankWolfeConfig, FrankWolfeResult, solve_frank_wolfe
from .initial_points import (
    canonical_vertex,
    generate_initial_points,
    random_feasible_point,
    random_vertex,
    save_initial_points,
)
from .io import (
    default_results_folder,
    infer_equal_partition,
    load_initial_point,
    load_initial_point_keys,
    linear_term_from_stationary_point,
    load_partition_metadata,
    load_problem,
    resolve_problem_data_folder,
    resolve_problem_root,
    save_partition_metadata,
)
from .oracle import linear_minimization_oracle
from .problem import Partition, SimplexQP
from .trace import (
    TraceProjector,
    load_gurobi_solution,
    load_target_for_case,
    make_trace_projector,
    target_key_for_vector,
)

__all__ = [
    "BaselineResult",
    "DEFAULT_PROBLEM_CONFIG_PATH",
    "FrankWolfeConfig",
    "FrankWolfeResult",
    "InitialPointsConfig",
    "Partition",
    "ProblemConfig",
    "SimplexQP",
    "TraceProjector",
    "canonical_vertex",
    "default_results_folder",
    "generate_initial_points",
    "infer_equal_partition",
    "load_gurobi_solution",
    "load_initial_point",
    "load_initial_point_keys",
    "load_target_for_case",
    "load_problem_config",
    "linear_term_from_stationary_point",
    "linear_minimization_oracle",
    "load_partition_metadata",
    "load_problem",
    "partition_from_config",
    "random_feasible_point",
    "random_vertex",
    "resolve_problem_data_folder",
    "resolve_problem_root",
    "save_initial_points",
    "save_partition_metadata",
    "solve_frank_wolfe",
    "solve_gurobi",
    "make_trace_projector",
    "target_key_for_vector",
]
