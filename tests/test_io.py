import tempfile
import unittest
from pathlib import Path

import numpy as np

from simplex_qp.io import load_initial_point, load_initial_point_keys


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


if __name__ == "__main__":
    unittest.main()
