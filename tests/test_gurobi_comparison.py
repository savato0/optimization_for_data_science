import unittest

import numpy as np

from simplex_qp.baseline_gurobi import gp, solve_gurobi
from simplex_qp.fw import FrankWolfeConfig, solve_frank_wolfe
from simplex_qp.problem import Partition, SimplexQP


@unittest.skipUnless(gp is not None, "gurobipy is not installed in the active Python environment.")
class GurobiComparisonTests(unittest.TestCase):
    def test_fw_matches_gurobi_on_small_quadratic_instance(self) -> None:
        partition = Partition((np.array([0, 1]), np.array([2, 3])))
        problem = SimplexQP(
            Q=np.diag([1.0, 1.0, 2.0, 2.0]),
            q=np.array([-10.0, 0.0, -9.0, 0.0]),
            partition=partition,
        )

        fw_result = solve_frank_wolfe(
            problem,
            FrankWolfeConfig(max_iter=25, tol_gap=1e-12, store_history=True),
        )
        baseline_result = solve_gurobi(problem)

        self.assertEqual(fw_result.status, "converged")
        self.assertEqual(baseline_result.status, "optimal")
        self.assertIsNotNone(baseline_result.objective)
        self.assertAlmostEqual(fw_result.objective, baseline_result.objective, places=10)
        self.assertLessEqual(baseline_result.fw_gap or 0.0, 1e-10)


if __name__ == "__main__":
    unittest.main()
