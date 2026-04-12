import unittest

import numpy as np

from simplex_qp.fw import FrankWolfeConfig, solve_frank_wolfe
from simplex_qp.io import linear_term_from_stationary_point
from simplex_qp.problem import Partition, SimplexQP


class FrankWolfeSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.partition = Partition((np.array([0, 1]), np.array([2, 3])))

    def test_barycenter_is_feasible(self) -> None:
        problem = SimplexQP(
            Q=np.eye(4),
            q=np.zeros(4),
            partition=self.partition,
        )

        x0 = problem.barycenter()

        self.assertTrue(problem.is_feasible(x0))

    def test_objective_is_nonincreasing_and_iterates_stay_feasible(self) -> None:
        problem = SimplexQP(
            Q=np.diag([1.0, 2.0, 3.0, 4.0]),
            q=np.array([-4.0, -1.0, -3.0, -0.5]),
            partition=self.partition,
        )
        config = FrankWolfeConfig(max_iter=200, tol_gap=1e-8, store_history=True)

        result = solve_frank_wolfe(problem, config)

        self.assertIsNotNone(result.history)
        objectives = [entry["objective"] for entry in result.history or []]
        self.assertGreater(len(objectives), 0)
        for current, following in zip(objectives, objectives[1:]):
            self.assertLessEqual(following, current + 1e-10)
        for entry in result.history or []:
            self.assertLessEqual(entry["max_block_sum_error"], 1e-10)
            self.assertLessEqual(entry["max_nonnegativity_violation"], 1e-10)

    def test_gap_improves_and_solver_converges_on_toy_problem(self) -> None:
        problem = SimplexQP(
            Q=np.diag([1.0, 1.0, 2.0, 2.0]),
            q=np.array([-10.0, 0.0, -9.0, 0.0]),
            partition=self.partition,
        )
        config = FrankWolfeConfig(max_iter=25, tol_gap=1e-12, store_history=True)

        result = solve_frank_wolfe(problem, config)

        self.assertEqual(result.status, "converged")
        self.assertLessEqual(result.gap, config.tol_gap)
        self.assertIsNotNone(result.history)
        history = result.history or []
        self.assertGreater(history[0]["fw_gap"], result.gap)
        self.assertAlmostEqual(result.objective - result.lower_bound, result.gap, places=12)
        np.testing.assert_allclose(result.x, np.array([1.0, 0.0, 1.0, 0.0]))

    def test_linear_term_matches_report_stationary_point_formula(self) -> None:
        Q = np.diag([1.0, 2.0, 3.0, 4.0])
        x_u = np.array([0.5, 0.5, 0.25, 0.75])

        q = linear_term_from_stationary_point(Q, x_u)
        problem = SimplexQP(Q=Q, q=q, partition=self.partition)

        np.testing.assert_allclose(problem.gradient(x_u), np.zeros_like(x_u))


if __name__ == "__main__":
    unittest.main()
