import unittest

from scripts.run_fw_experiments import _resolve_x0_keys


class RunFwExperimentsTests(unittest.TestCase):
    def test_resolve_x0_keys_defaults_to_barycenter(self) -> None:
        self.assertEqual(_resolve_x0_keys(None, None, False), ["barycenter"])

    def test_resolve_x0_keys_uses_repeated_keys(self) -> None:
        self.assertEqual(
            _resolve_x0_keys("initial_points.npz", ["vertex_0", "random_vertex_0"], False),
            ["vertex_0", "random_vertex_0"],
        )


if __name__ == "__main__":
    unittest.main()
