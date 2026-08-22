import unittest

import numpy as np
import pandas as pd

from ecap.copula_mc import simulate_copula_defaults
from regcap.rwa_engine import compute_total_rwa


class RiskSanityTests(unittest.TestCase):
    def setUp(self):
        self.portfolio = pd.DataFrame({
            "segment": ["Retail_Mortgage", "SME_Corporate", "Corporate_Large"],
            "ead": [1_000_000.0, 2_000_000.0, 3_000_000.0],
            "lgd": [0.30, 0.45, 0.40],
            "pit_pd_12m": [0.01, 0.04, 0.02],
            "ttc_pd": [0.015, 0.045, 0.02],
            "downturn_lgd": [0.35, 0.50, 0.45],
            "base_segment_corr": [0.15, 0.18, 0.24],
            "tenure_years": [10, 5, 8],
        })

    def test_copula_is_reproducible_and_non_negative(self):
        first = simulate_copula_defaults(self.portfolio, n_sims=1000, seed=7, copula_type="t")
        second = simulate_copula_defaults(self.portfolio, n_sims=1000, seed=7, copula_type="t")

        np.testing.assert_array_equal(first["simulated_losses"], second["simulated_losses"])
        self.assertTrue(np.all(first["simulated_losses"] >= 0))
        self.assertEqual(first["copula_type"], "t")
        self.assertTrue(first["tail_estimate_warning"])

    def test_tail_warning_is_exposed_for_small_samples(self):
        result = simulate_copula_defaults(self.portfolio, n_sims=100, seed=7, copula_type="Gaussian")
        self.assertTrue(result["tail_estimate_warning"])
        self.assertEqual(result["tail_observations_999"], 1)

    def test_copula_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            simulate_copula_defaults(self.portfolio, copula_type="unknown")
        with self.assertRaises(ValueError):
            simulate_copula_defaults(self.portfolio, n_sims=0)
        with self.assertRaises(ValueError):
            simulate_copula_defaults(self.portfolio, t_df=2)

    def test_rwa_reports_its_methodology(self):
        result = compute_total_rwa(self.portfolio, {"load_shedding_stage": 2})
        self.assertEqual(result["methodology"]["credit"], "IRB-style Vasicek ASRF")
        self.assertGreaterEqual(result["total_rwa"], 0)


if __name__ == "__main__":
    unittest.main()
