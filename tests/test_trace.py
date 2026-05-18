import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simplex_qp.trace import (
    load_gurobi_solution,
    load_target_for_case,
    make_trace_projector,
    target_key_for_vector,
)


class TraceProjectorTests(unittest.TestCase):
    def test_projector_places_reference_on_first_axis(self) -> None:
        x0 = np.array([0.0, 0.0, 0.0])
        reference = np.array([2.0, 0.0, 0.0])
        xu = np.array([0.0, 3.0, 0.0])

        projector = make_trace_projector(x0, reference=reference, xu=xu)

        np.testing.assert_allclose(projector.axis_1 @ projector.axis_2, 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(projector.axis_1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(projector.axis_2), 1.0)
        self.assertEqual(projector.project(x0)["coord_1"], 0.0)
        self.assertEqual(projector.project(x0)["coord_2"], 0.0)
        np.testing.assert_allclose(projector.project(reference)["coord_1"], 2.0)
        np.testing.assert_allclose(projector.project(reference)["coord_2"], 0.0)
        np.testing.assert_allclose(projector.project(xu)["coord_1"], 0.0)
        np.testing.assert_allclose(projector.project(xu)["coord_2"], 3.0)

    def test_projector_can_use_only_xu(self) -> None:
        x0 = np.array([1.0, 0.0, 0.0])
        xu = np.array([1.0, 4.0, 0.0])

        projector = make_trace_projector(x0, xu=xu)

        projected = projector.project(xu)
        np.testing.assert_allclose(projected["coord_1"], 4.0)
        np.testing.assert_allclose(projected["coord_2"], 0.0, atol=1e-12)

    def test_target_key_for_vector_uses_scenario_suffix(self) -> None:
        self.assertEqual(target_key_for_vector("q_well_sc2"), "x_u_sc2")
        self.assertEqual(target_key_for_vector("q_ill_sc3"), "x_u_sc3")

    def test_load_target_for_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "targets.npz"
            np.savez(path, x_u_sc2=np.array([1.0, 2.0]))

            key, values = load_target_for_case(path, "q_well_sc2")

            self.assertEqual(key, "x_u_sc2")
            np.testing.assert_allclose(values, np.array([1.0, 2.0]))

    def test_load_gurobi_solution_supports_baseline_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gurobi.json"
            records = [
                {
                    "matrix": "Q_well",
                    "vector": "q_well_sc2",
                    "x": [0.25, 0.75],
                }
            ]
            path.write_text(json.dumps(records), encoding="utf-8")

            values = load_gurobi_solution(path, "Q_well", "q_well_sc2")

            np.testing.assert_allclose(values, np.array([0.25, 0.75]))


if __name__ == "__main__":
    unittest.main()
