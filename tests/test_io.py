import tempfile
import unittest
from pathlib import Path

import numpy as np

from simplex_qp import Partition, save_partition_metadata
from simplex_qp.io import (
    default_results_folder,
    load_initial_point,
    load_initial_point_keys,
    load_problem,
    resolve_problem_data_folder,
)


class InitialPointLoaderTests(unittest.TestCase):
    def test_load_initial_point_from_npz_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "initial_points.npz"
            np.savez(path, vertex_0=np.array([1.0, 0.0, 0.0]))

            x0 = load_initial_point(path, key="vertex_0")

        np.testing.assert_allclose(x0, np.array([1.0, 0.0, 0.0]))

    def test_load_initial_point_keys_from_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "initial_points.npz"
            np.savez(path, barycenter=np.array([0.5, 0.5]), vertex_0=np.array([1.0, 0.0]))

            keys = load_initial_point_keys(path)

        self.assertEqual(keys, ["barycenter", "vertex_0"])

    def test_load_initial_point_requires_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "x0.txt"
            path.write_text("1 0 0", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_initial_point(path, key="vertex_0")

    def test_load_initial_point_requires_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "initial_points.npz"
            np.savez(path, vertex_0=np.array([1.0, 0.0]))

            with self.assertRaises(ValueError):
                load_initial_point(path, key="")

    def test_load_initial_point_rejects_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "initial_points.npz"
            np.savez(path, vertex_0=np.array([1.0, 0.0]))

            with self.assertRaises(KeyError):
                load_initial_point(path, key="missing")

    def test_load_problem_accepts_problem_root_with_data_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dim_n4_k2"
            data_folder = root / "data"
            data_folder.mkdir(parents=True)
            np.savez(data_folder / "matrices.npz", Q=np.eye(4))
            np.savez(data_folder / "vectors.npz", q=np.zeros(4))
            save_partition_metadata(data_folder, Partition.from_block_sizes([2, 2]))

            problem = load_problem(root, "Q", "q")

        self.assertEqual(problem.dimension, 4)
        self.assertEqual(problem.partition.block_sizes, (2, 2))

    def test_resolve_problem_data_folder_supports_root_or_data_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dim_n4_k2"
            data_folder = root / "data"
            data_folder.mkdir(parents=True)

            self.assertEqual(resolve_problem_data_folder(root), data_folder)
            self.assertEqual(resolve_problem_data_folder(data_folder), data_folder)
            self.assertEqual(default_results_folder(root), root / "results")
            self.assertEqual(default_results_folder(data_folder), root / "results")


if __name__ == "__main__":
    unittest.main()
