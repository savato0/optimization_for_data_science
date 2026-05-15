import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.summarize_fw_results import (
    emit_output,
    filter_records,
    flatten_history,
    format_history_table,
    format_summary,
    format_table,
    load_gurobi_baselines,
    load_records,
    sort_records,
    target_key_from_vector,
    write_text,
)


class SummarizeFwResultsTests(unittest.TestCase):
    def test_load_records_reads_fw_result_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fw_results.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "matrix": "Q_well",
                            "vector": "q_well_sc2",
                            "x0_key": "vertex_0",
                            "status": "converged",
                            "objective": -1.0,
                            "fw_gap": 1e-7,
                            "iterations": 10,
                            "runtime_seconds": 0.1,
                            "history": [
                                {"iteration": 0.0, "objective": 3.0, "alpha": 0.75},
                                {"iteration": 1.0, "objective": -2.0},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = load_records(path)

        self.assertEqual(records[0]["x0_key"], "vertex_0")
        self.assertEqual(records[0]["objective"], -1.0)
        self.assertEqual(records[0]["f_iter1"], -2.0)
        self.assertEqual(records[0]["alpha_iter0"], 0.75)

    def test_load_records_flattens_comparison_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison_results.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "matrix": "Q_well",
                            "vector": "q_well_sc2",
                            "x0_key": "vertex_0",
                            "frank_wolfe": {
                                "status": "converged",
                                "objective": -1.0,
                                "fw_gap": 1e-7,
                                "iterations": 10,
                                "runtime_seconds": 0.1,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = load_records(path)

        self.assertEqual(records[0]["matrix"], "Q_well")
        self.assertEqual(records[0]["status"], "converged")
        self.assertEqual(records[0]["x0_key"], "vertex_0")

    def test_target_key_from_vector_maps_project_scenarios(self) -> None:
        self.assertEqual(target_key_from_vector("q_well_sc1"), "x_u_sc1")
        self.assertEqual(target_key_from_vector("q_ill_sc2"), "x_u_sc2")
        self.assertEqual(target_key_from_vector("q_well_sc3"), "x_u_sc3")
        self.assertIsNone(target_key_from_vector("q_without_scenario"))

    def test_load_records_adds_target_distance_when_solution_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            results_path = folder / "fw_results.json"
            targets_path = folder / "targets.npz"
            results_path.write_text(
                json.dumps(
                    [
                        {
                            "matrix": "Q_well",
                            "vector": "q_well_sc1",
                            "x0_key": "vertex_0",
                            "status": "converged",
                            "objective": -1.0,
                            "fw_gap": 1e-7,
                            "iterations": 10,
                            "runtime_seconds": 0.1,
                            "x": [1.0, 0.0, 0.0, 1.0],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            np.savez(targets_path, x_u_sc1=np.array([0.0, 0.0, 0.0, 1.0]))

            records = load_records(results_path, targets_file=targets_path)
            table = format_table(records)

        self.assertEqual(records[0]["target_key"], "x_u_sc1")
        self.assertAlmostEqual(records[0]["distance_to_xu"], 1.0)
        self.assertIn("target_key", table)
        self.assertIn("distance_to_xu", table)

    def test_load_records_adds_target_key_without_distance_when_solution_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            results_path = folder / "fw_results.json"
            targets_path = folder / "targets.npz"
            results_path.write_text(
                json.dumps(
                    [
                        {
                            "matrix": "Q_ill",
                            "vector": "q_ill_sc3",
                            "x0_key": "barycenter",
                            "status": "max_iter",
                            "objective": -1.0,
                            "fw_gap": 1e-3,
                            "iterations": 100,
                            "runtime_seconds": 0.2,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            np.savez(targets_path, x_u_sc3=np.array([0.5, 0.5]))

            records = load_records(results_path, targets_file=targets_path)

        self.assertEqual(records[0]["target_key"], "x_u_sc3")
        self.assertIsNone(records[0]["distance_to_xu"])

    def test_load_records_adds_gurobi_objective_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            results_path = folder / "fw_results.json"
            gurobi_path = folder / "gurobi_baseline.json"
            results_path.write_text(
                json.dumps(
                    [
                        {
                            "matrix": "Q_well",
                            "vector": "q_well_sc2",
                            "x0_key": "vertex_0",
                            "status": "converged",
                            "objective": -9.5,
                            "fw_gap": 1e-7,
                            "iterations": 10,
                            "runtime_seconds": 0.1,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            gurobi_path.write_text(
                json.dumps(
                    [
                        {
                            "solver": "gurobi",
                            "matrix": "Q_well",
                            "vector": "q_well_sc2",
                            "status": "optimal",
                            "objective": -10.0,
                            "runtime_seconds": 0.2,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = load_records(results_path, gurobi_file=gurobi_path)
            table = format_table(records)

        self.assertEqual(records[0]["gurobi_status"], "optimal")
        self.assertEqual(records[0]["gurobi_objective"], -10.0)
        self.assertAlmostEqual(records[0]["objective_difference"], 0.5)
        self.assertIn("gurobi_objective", table)
        self.assertIn("objective_difference", table)

    def test_load_gurobi_baselines_accepts_nested_compare_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison_results.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "matrix": "Q_well",
                            "vector": "q_well_sc1",
                            "gurobi": {
                                "status": "optimal",
                                "objective": -8.0,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            baselines = load_gurobi_baselines(path)

        self.assertEqual(baselines[("Q_well", "q_well_sc1")]["status"], "optimal")
        self.assertEqual(baselines[("Q_well", "q_well_sc1")]["objective"], -8.0)

    def test_format_summary_and_table_include_key_fields(self) -> None:
        records = [
            {
                "matrix": "Q_well",
                "vector": "q_well_sc2",
                "x0_key": "vertex_0",
                "status": "converged",
                "objective": -2.0,
                "fw_gap": 1e-7,
                "iterations": 5,
                "runtime_seconds": 0.1,
            },
            {
                "matrix": "Q_well",
                "vector": "q_well_sc2",
                "x0_key": "vertex_1",
                "status": "max_iter",
                "objective": -1.0,
                "fw_gap": 1e-3,
                "iterations": 100,
                "runtime_seconds": 0.2,
            },
        ]

        summary = format_summary(records)
        table = format_table(sort_records(records, sort_by="iterations", descending=False))

        self.assertIn("Records: 2", summary)
        self.assertIn("Converged: 1", summary)
        self.assertIn("vertex_0", table)
        self.assertIn("max_iter", table)
        self.assertIn("f_iter1", table)
        self.assertIn("alpha_iter0", table)

    def test_filter_records_by_case_and_x0_key(self) -> None:
        records = [
            {
                "matrix": "Q_well",
                "vector": "q_well_sc1",
                "x0_key": "vertex_0",
            },
            {
                "matrix": "Q_well",
                "vector": "q_well_sc2",
                "x0_key": "vertex_0",
            },
            {
                "matrix": "Q_ill",
                "vector": "q_ill_sc1",
                "x0_key": "vertex_1",
            },
        ]

        by_case = filter_records(records, cases=["Q_well:q_well_sc1"])
        by_x0 = filter_records(records, x0_keys=["vertex_0"])
        by_both = filter_records(
            records,
            cases=["Q_well:q_well_sc2"],
            x0_keys=["vertex_0"],
        )

        self.assertEqual(len(by_case), 1)
        self.assertEqual(by_case[0]["vector"], "q_well_sc1")
        self.assertEqual(len(by_x0), 2)
        self.assertEqual(len(by_both), 1)
        self.assertEqual(by_both[0]["vector"], "q_well_sc2")

    def test_filter_records_rejects_invalid_case(self) -> None:
        with self.assertRaises(ValueError):
            filter_records([], cases=["Q_well"])

    def test_write_text_creates_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.txt"

            write_text(path, "hello")

            self.assertEqual(path.read_text(encoding="utf-8"), "hello\n")

    def test_emit_output_prints_only_completion_message_when_output_file_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.txt"

            with patch("builtins.print") as print_mock:
                emit_output("large table", output_path=path)

            self.assertEqual(path.read_text(encoding="utf-8"), "large table\n")
            print_mock.assert_called_once_with(f"Saved text output to {path}", flush=True)

    def test_emit_output_prints_content_without_output_file(self) -> None:
        with patch("builtins.print") as print_mock:
            emit_output("small table")

        print_mock.assert_called_once_with("small table", flush=True)

    def test_flatten_history_filters_by_x0_key(self) -> None:
        records = [
            {
                "matrix": "Q_well",
                "vector": "q_well_sc2",
                "x0_key": "vertex_0",
                "history": [
                    {
                        "iteration": 0.0,
                        "objective": 1.0,
                        "fw_gap": 2.0,
                        "lower_bound": -1.0,
                        "alpha": 1.0,
                    }
                ],
            },
            {
                "matrix": "Q_well",
                "vector": "q_well_sc2",
                "x0_key": "vertex_1",
                "history": [
                    {
                        "iteration": 0.0,
                        "objective": 3.0,
                        "fw_gap": 4.0,
                        "lower_bound": -1.0,
                        "alpha": 1.0,
                    }
                ],
            },
        ]

        rows = flatten_history(records, x0_keys=["vertex_0"])
        table = format_history_table(rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["x0_key"], "vertex_0")
        self.assertIn("iteration", table)
        self.assertIn("vertex_0", table)
        self.assertIn("0  ", table)
        self.assertNotIn("0.000000e+00", table)


if __name__ == "__main__":
    unittest.main()
