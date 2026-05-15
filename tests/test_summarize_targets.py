import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.summarize_targets import format_table, load_target_records, summarize_target
from simplex_qp import Partition, save_partition_metadata


class SummarizeTargetsTests(unittest.TestCase):
    def test_summarize_target_reports_geometry_metrics(self) -> None:
        partition = Partition.from_block_sizes([2, 2])
        record = summarize_target(
            "x_u_sc3",
            np.array([1.1, -0.1, 0.25, 0.75]),
            partition,
        )

        self.assertEqual(record["target"], "x_u_sc3")
        self.assertEqual(record["negative_count"], 1)
        self.assertAlmostEqual(record["min_value"], -0.1)
        self.assertAlmostEqual(record["max_block_sum_error"], 0.0)

    def test_load_target_records_reads_targets_npz_and_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            save_partition_metadata(folder, Partition.from_block_sizes([2, 2]))
            np.savez(
                folder / "targets.npz",
                x_u_sc1=np.array([0.5, 0.5, 0.25, 0.75]),
                x_u_sc2=np.array([2.0, -1.0, 1.5, -0.5]),
            )

            records = load_target_records(folder)

        self.assertEqual([record["target"] for record in records], ["x_u_sc1", "x_u_sc2"])
        self.assertEqual(records[0]["negative_count"], 0)
        self.assertEqual(records[1]["negative_count"], 2)

    def test_format_table_includes_expected_columns(self) -> None:
        table = format_table(
            [
                {
                    "target": "x_u_sc1",
                    "min_value": 0.1,
                    "negative_count": 0,
                    "max_block_sum_error": 0.0,
                    "l2_norm": 1.0,
                }
            ]
        )

        self.assertIn("target", table)
        self.assertIn("negative_count", table)
        self.assertIn("x_u_sc1", table)


if __name__ == "__main__":
    unittest.main()
