import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_compare_experiments import _resolve_x0_keys


class RunCompareExperimentsTests(unittest.TestCase):
    def test_resolve_x0_keys_defaults_to_barycenter(self) -> None:
        self.assertEqual(_resolve_x0_keys(None, None, False), ["barycenter"])

    def test_resolve_x0_keys_uses_repeated_keys(self) -> None:
        self.assertEqual(
            _resolve_x0_keys("initial_points.npz", ["vertex_0", "random_vertex_0"], False),
            ["vertex_0", "random_vertex_0"],
        )

    def test_resolve_x0_keys_reads_all_keys_from_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "initial_points.npz"
            np.savez(path, barycenter=np.array([0.5, 0.5]), vertex_0=np.array([1.0, 0.0]))

            keys = _resolve_x0_keys(str(path), None, True)

        self.assertEqual(keys, ["barycenter", "vertex_0"])


if __name__ == "__main__":
    unittest.main()
