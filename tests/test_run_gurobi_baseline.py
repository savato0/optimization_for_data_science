import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_gurobi_baseline import (
    _load_existing_records,
    _record_key,
    _write_records,
    parse_case,
)


class RunGurobiBaselineTests(unittest.TestCase):
    def test_parse_case_splits_matrix_and_vector(self) -> None:
        self.assertEqual(parse_case("Q_well:q_well_sc1"), ("Q_well", "q_well_sc1"))

    def test_parse_case_rejects_missing_separator(self) -> None:
        with self.assertRaises(ValueError):
            parse_case("Q_well")

    def test_load_existing_records_returns_empty_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gurobi_baseline.json"

            self.assertEqual(_load_existing_records(path), [])

    def test_write_and_load_existing_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "gurobi_baseline.json"
            records = [
                {
                    "matrix": "Q_well",
                    "vector": "q_well_sc2",
                    "objective": -1.0,
                }
            ]

            _write_records(path, records)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), records)
            self.assertEqual(_load_existing_records(path), records)

    def test_record_key_identifies_matrix_and_vector(self) -> None:
        self.assertEqual(
            _record_key(
                {
                    "matrix": "Q_well",
                    "vector": "q_well_sc2",
                }
            ),
            ("Q_well", "q_well_sc2"),
        )


if __name__ == "__main__":
    unittest.main()
