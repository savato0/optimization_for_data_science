import unittest

import numpy as np

from simplex_qp.initial_points import (
    canonical_vertex,
    generate_initial_points,
    random_feasible_point,
)
from simplex_qp.problem import Partition


class InitialPointGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.partition = Partition.from_block_sizes([4, 4, 4])

    def test_canonical_vertex_selects_same_local_index_per_block(self) -> None:
        x0 = canonical_vertex(self.partition, local_index=0, dimension=12)

        np.testing.assert_allclose(
            x0,
            np.array([1.0, 0.0, 0.0, 0.0] * 3),
        )

    def test_generate_initial_points_contains_barycenter_and_vertices(self) -> None:
        points = generate_initial_points(
            self.partition,
            dimension=12,
            num_canonical_vertices=4,
            num_random_vertices=2,
            num_random_feasible=1,
            seed=123,
        )

        self.assertEqual(
            set(points),
            {
                "barycenter",
                "vertex_0",
                "vertex_1",
                "vertex_2",
                "vertex_3",
                "random_vertex_0",
                "random_vertex_1",
                "random_feasible_0",
            },
        )
        np.testing.assert_allclose(points["barycenter"], np.full(12, 0.25))
        np.testing.assert_allclose(points["vertex_0"], np.array([1.0, 0.0, 0.0, 0.0] * 3))
        for point in points.values():
            self.assertTrue(np.all(point >= 0.0))
            np.testing.assert_allclose(self.partition.block_sums(point), np.ones(3))

    def test_random_vertices_are_reproducible_with_seed(self) -> None:
        first = generate_initial_points(
            self.partition,
            dimension=12,
            num_canonical_vertices=0,
            num_random_vertices=3,
            num_random_feasible=3,
            seed=7,
        )
        second = generate_initial_points(
            self.partition,
            dimension=12,
            num_canonical_vertices=0,
            num_random_vertices=3,
            num_random_feasible=3,
            seed=7,
        )

        for key in first:
            np.testing.assert_allclose(first[key], second[key])

    def test_random_feasible_point_can_use_multiple_active_components_per_block(self) -> None:
        rng = np.random.default_rng(0)
        point = random_feasible_point(self.partition, rng=rng, dimension=12)

        active_counts = [
            int(np.count_nonzero(point[block] > 0.0))
            for block in self.partition.blocks
        ]

        self.assertTrue(any(count > 1 for count in active_counts))
        self.assertTrue(np.all(point >= 0.0))
        np.testing.assert_allclose(self.partition.block_sums(point), np.ones(3))


if __name__ == "__main__":
    unittest.main()
