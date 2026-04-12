"""Utilities for convex quadratic programs over products of simplices."""

from .baseline_gurobi import BaselineResult, solve_gurobi
from .fw import FrankWolfeConfig, FrankWolfeResult, solve_frank_wolfe
from .io import (
    infer_equal_partition,
    load_partition_metadata,
    load_problem,
    save_partition_metadata,
)
from .oracle import linear_minimization_oracle
from .problem import Partition, SimplexQP

__all__ = [
    "BaselineResult",
    "FrankWolfeConfig",
    "FrankWolfeResult",
    "Partition",
    "SimplexQP",
    "infer_equal_partition",
    "linear_minimization_oracle",
    "load_partition_metadata",
    "load_problem",
    "save_partition_metadata",
    "solve_frank_wolfe",
    "solve_gurobi",
]
