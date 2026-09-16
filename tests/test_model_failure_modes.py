import unittest

from run import run_engine_end_to_end


class ModelFailureModeTests(unittest.TestCase):
    def test_adverse_ecl_is_monotonic_in_severity(self):
        ecl_values = []
        for severity in (0.5, 1.0, 1.5, 2.5):
            result = run_engine_end_to_end(
                scenario="Adverse", severity_multiplier=severity,
                n_accounts=30, n_mc_sims=20, seed=2024,
                as_of_date="2026-09-16",
            )
            ecl_values.append(result["ifrs9"]["ecl_total"])
        self.assertEqual(ecl_values, sorted(ecl_values))


if __name__ == "__main__":
    unittest.main()