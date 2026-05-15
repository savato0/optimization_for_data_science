import json
import tempfile
import unittest
from pathlib import Path

from simplex_qp.experiment_config import load_problem_config, partition_from_config


class ExperimentConfigTests(unittest.TestCase):
    def test_load_problem_config_builds_equal_contiguous_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "problem_config.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "toy",
                        "data_root": tmpdir,
                        "n": 12,
                        "k": 3,
                        "seed": 42,
                            "initial_points": {
                                "num_canonical_vertices": 4,
                                "num_random_vertices": 2,
                                "num_random_feasible": 5,
                                "seed": 7,
                            },
                    }
                ),
                encoding="utf-8",
            )

            config = load_problem_config(path)
            partition = partition_from_config(config)

        self.assertEqual(config.n, 12)
        self.assertEqual(config.k, 3)
        self.assertEqual(config.block_size, 4)
        self.assertEqual(config.initial_points.num_random_vertices, 2)
        self.assertEqual(config.initial_points.num_random_feasible, 5)
        self.assertEqual(partition.block_sizes, (4, 4, 4))

    def test_load_problem_config_uses_explicit_dimension_name_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "problem_config.json"
            path.write_text(
                json.dumps({"data_root": tmpdir, "n": 1000, "k": 100}),
                encoding="utf-8",
            )

            config = load_problem_config(path)

        self.assertEqual(config.name, "dim_n1000_k100")
        self.assertEqual(config.data_folder, Path(tmpdir) / "dim_n1000_k100")

    def test_load_problem_config_requires_divisible_n_and_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "problem_config.json"
            path.write_text(json.dumps({"name": "bad", "n": 10, "k": 3}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_problem_config(path)


if __name__ == "__main__":
    unittest.main()
