import unittest

import numpy as np
import pandas as pd

from ecap.copula_mc import compute_convergence_diagnostics, simulate_copula_defaults
from ifrs9.pd_model import convert_ttc_to_pit
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

    def test_convergence_diagnostics_reports_checkpoint_changes(self):
        result = compute_convergence_diagnostics(np.arange(100, dtype=float), checkpoints=[0.5, 1.0])
        self.assertEqual(result["simulation_counts"], [50, 100])
        self.assertEqual(len(result["relative_changes"]), 1)

    def test_lower_gdp_increases_sme_point_in_time_pd(self):
        portfolio = pd.DataFrame({
            "segment": ["SME_Corporate"],
            "loadshedding_vulnerability_score": [3],
            "months_on_book": [24],
            "internal_rating": ["BBB"],
            "base_segment_ttc_pd": [0.045],
            "dpd": [0],
            "debt_review_flag": [False],
            "judgement_flag": [False],
        })
        base = convert_ttc_to_pit(portfolio, {
            "gdp_yoy": 0.015, "unemployment_rate": 0.32,
            "load_shedding_stage": 2, "cpi_yoy": 0.05, "repo_rate": 0.0775,
        })
        stress = convert_ttc_to_pit(portfolio, {
            "gdp_yoy": -0.03, "unemployment_rate": 0.32,
            "load_shedding_stage": 2, "cpi_yoy": 0.05, "repo_rate": 0.0775,
        })
        self.assertGreater(stress.loc[0, "pit_pd_12m"], base.loc[0, "pit_pd_12m"])


if __name__ == "__main__":
    unittest.main()
