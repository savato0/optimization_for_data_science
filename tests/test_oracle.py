import unittest

import numpy as np

from simplex_qp.oracle import linear_minimization_oracle
from simplex_qp.problem import Partition


class LinearMinimizationOracleTests(unittest.TestCase):
    def test_oracle_selects_one_hot_minimizers_per_block(self) -> None:
        partition = Partition((np.array([0, 1]), np.array([2, 3, 4])))
        gradient = np.array([2.0, -1.0, 0.5, -3.0, 4.0])

        s = linear_minimization_oracle(gradient, partition)

        np.testing.assert_allclose(s, np.array([0.0, 1.0, 0.0, 1.0, 0.0]))

    def test_oracle_handles_noncontiguous_blocks(self) -> None:
        partition = Partition((np.array([0, 2, 4]), np.array([1, 3])))
        gradient = np.array([3.0, -2.0, -5.0, 1.0, 0.0])

        s = linear_minimization_oracle(gradient, partition)

        np.testing.assert_allclose(s, np.array([0.0, 1.0, 1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
