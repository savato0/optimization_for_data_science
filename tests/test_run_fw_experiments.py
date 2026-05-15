import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_fw_experiments import (
    _load_existing_records,
    _record_key,
    _resolve_x0_keys,
    _write_records,
)


class RunFwExperimentsTests(unittest.TestCase):
    def test_resolve_x0_keys_defaults_to_barycenter(self) -> None:
        self.assertEqual(_resolve_x0_keys(None, None, False), ["barycenter"])

    def test_resolve_x0_keys_uses_repeated_keys(self) -> None:
        self.assertEqual(
            _resolve_x0_keys("initial_points.npz", ["vertex_0", "random_vertex_0"], False),
            ["vertex_0", "random_vertex_0"],
        )

    def test_load_existing_records_returns_empty_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fw_results.json"

            self.assertEqual(_load_existing_records(path), [])

    def test_write_and_load_existing_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "fw_results.json"
            records = [
                {
                    "matrix": "Q_well",
                    "vector": "q_well_sc2",
                    "x0_key": "vertex_0",
                    "objective": -1.0,
                }
            ]

            _write_records(path, records)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), records)
            self.assertEqual(_load_existing_records(path), records)

    def test_record_key_identifies_matrix_vector_and_x0(self) -> None:
        self.assertEqual(
            _record_key(
                {
                    "matrix": "Q_well",
                    "vector": "q_well_sc2",
                    "x0_key": "vertex_0",
                }
            ),
            ("Q_well", "q_well_sc2", "vertex_0"),
        )


if __name__ == "__main__":
    unittest.main()
