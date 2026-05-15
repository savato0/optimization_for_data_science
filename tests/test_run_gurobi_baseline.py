import unittest

from scripts.run_gurobi_baseline import parse_case


class RunGurobiBaselineTests(unittest.TestCase):
    def test_parse_case_splits_matrix_and_vector(self) -> None:
        self.assertEqual(parse_case("Q_well:q_well_sc1"), ("Q_well", "q_well_sc1"))

    def test_parse_case_rejects_missing_separator(self) -> None:
        with self.assertRaises(ValueError):
            parse_case("Q_well")


if __name__ == "__main__":
    unittest.main()
