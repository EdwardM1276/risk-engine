"""Regression tests for explicit proxy and uncertainty contracts."""

import unittest

import numpy as np
import pandas as pd

from config.params import NEDBANK_ECAP_BENCHMARK_2024
from ecap.allocation import allocate_nedbank_ecap_benchmark
from ecap.copula_mc import (
    compute_convergence_diagnostics,
    compute_market_risk_ecap,
    compute_operational_ecap,
    repair_correlation_matrix,
)
from ecap.coverage import compute_afr_coverage_ratio
from data.acquisition import generate_sa_loan_portfolio
from data.calibration import apply_pd_calibration, calibrate_ecl_to_target
from ifrs9.lgd_model import calculate_sa_adjusted_lgd
from ifrs9.pd_model import convert_ttc_to_pit
from regcap.rwa_engine import current_output_floor_pct
from run import _compute_uncertainty_bands
from scenarios.stress_engine import (
    create_idiosyncratic_scenario,
    scenario_to_macro_conditions,
    scenario_to_market_data,
)


class RemediationContractTests(unittest.TestCase):
    def test_south_african_output_floor_effective_date_boundaries(self):
        cases = {
            "2025-06-30": 0.0,
            "2025-07-01": 0.60,
            "2025-12-31": 0.60,
            "2026-01-01": 0.65,
            "2027-01-01": 0.70,
            "2028-01-01": 0.725,
            "2031-01-01": 0.725,
        }
        for as_of_date, expected in cases.items():
            with self.subTest(as_of_date=as_of_date):
                self.assertEqual(current_output_floor_pct(as_of_date), expected)

    def test_nested_prefix_stability_never_claims_tail_convergence(self):
        result = compute_convergence_diagnostics(
            np.linspace(0, 1000, 10000), checkpoints=[0.5, 1.0]
        )
        self.assertIn("checkpoint_stable", result)
        self.assertFalse(result["converged"])
        self.assertFalse(result["formal_interval_available"])
        self.assertEqual(result["independent_replicates"], 1)

    def test_correlation_repair_is_reported_and_remains_factorable(self):
        non_positive_definite = np.array([
            [1.0, 0.95, 0.95],
            [0.95, 1.0, -0.95],
            [0.95, -0.95, 1.0],
        ])
        repaired, diagnostics = repair_correlation_matrix(non_positive_definite)
        self.assertTrue(diagnostics["matrix_repaired"])
        self.assertGreater(diagnostics["frobenius_adjustment"], 0.0)
        self.assertGreater(diagnostics["minimum_eigenvalue_after"], 0.0)
        np.linalg.cholesky(repaired)

    def test_benchmark_mix_reconciles_and_stress_is_separate(self):
        result = allocate_nedbank_ecap_benchmark(
            total_rwa=1_000_000_000,
            ifrs9_ecl_total=10_000_000,
        )
        weights = result["minimum_risk_mix_weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(result["minimum_risk_mix_weight_sum"], 1.0)
        self.assertNotIn("Stress Buffer ECap", weights)
        self.assertIn("Other Risk ECap (benchmark residual)", result["components"])
        self.assertIn("deterministic overlay", result["aggregation_policy"])
        self.assertAlmostEqual(
            NEDBANK_ECAP_BENCHMARK_2024["stress_buffer_pct"],
            result["stress_buffer_pct_of_benchmark_total"],
        )

    def test_coverage_labels_illustrative_resources(self):
        result = compute_afr_coverage_ratio(120.0, 100.0, {"Credit": 100.0})
        self.assertFalse(result["is_actual_bank_afr"])
        self.assertIn("not actual bank AFR", result["AFR basis"])

    def test_uncertainty_bands_are_not_called_confidence_intervals(self):
        bands = _compute_uncertainty_bands(
            pd.DataFrame({"ecl": [100.0]}),
            {"capital_ratios": {"Total CAR": 0.15}},
            {"total_ecap": 200.0},
            {
                "simulated_losses": np.arange(10, dtype=float),
                "credit_ecap_999": 9.0,
                "tail_observations_999": 1,
                "expected_tail_observations_999": 0.01,
                "tail_estimate_warning": True,
            },
            {"Coverage Ratio": 1.2},
        )
        self.assertNotIn("lower_95", bands["ECL"])
        self.assertIn("illustrative_lower", bands["ECL"])
        self.assertNotIn("illustrative_lower", bands["Credit ECap (99.9)"])
        self.assertTrue(bands["Credit ECap (99.9)"]["tail_estimate_warning"])
        self.assertIn("not a confidence interval", bands["ECL"]["range_type"])

    def test_pd_calibration_restages_and_reports_target_attainability(self):
        account = pd.DataFrame({
            "segment": ["SME_Corporate"],
            "ttc_pd": [0.01], "pit_pd_12m": [0.01], "lifetime_pd": [0.05],
            "ead": [100.0], "lgd": [0.5], "downturn_lgd": [0.6],
            "ifrs9_stage": [1], "dpd": [0], "debt_review_flag": [False],
            "judgement_flag": [False], "administration_order": [False],
            "loadshedding_vulnerability_score": [5], "internal_rating": ["BBB"],
        })
        severe_macro = {
            "gdp_yoy": -0.10, "unemployment_rate": 0.60,
            "load_shedding_stage": 8,
        }
        recalculated = apply_pd_calibration(account, 4.0, severe_macro)
        self.assertEqual(int(recalculated["ifrs9_stage"].iloc[0]), 2)
        _, diagnostics = calibrate_ecl_to_target(recalculated, 0.99)
        self.assertFalse(diagnostics["ecl_target_attainable"])
        self.assertFalse(diagnostics["ecl_calibration_converged"])
        self.assertGreater(abs(diagnostics["ecl_ead_ratio_residual"]), 0.0)


class ScenarioShockPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.portfolio = generate_sa_loan_portfolio(
            total_exposure=100_000_000.0, n_accounts=100, seed=19,
            as_of_date="2026-10-20",
        )
        cls.neutral_macro = {
            "gdp_yoy": 0.015, "unemployment_rate": 0.32,
            "load_shedding_stage": 2, "cpi_yoy": 0.05, "repo_rate": 0.0775,
        }

    def test_sovereign_shock_reaches_market_and_sovereign_lgd(self):
        params = create_idiosyncratic_scenario(sovereign_downgrade=True, seed=7)
        macro = scenario_to_macro_conditions(params)
        market = scenario_to_market_data(params)
        self.assertGreater(macro["sovereign_cds_change_bps"], 0)
        self.assertGreater(market["sovereign_cds_change_bps"], 0)
        sovereign = self.portfolio[self.portfolio["segment"] == "Sovereign_Bank"].head(1)
        base = calculate_sa_adjusted_lgd(sovereign, self.neutral_macro)
        shocked = calculate_sa_adjusted_lgd(sovereign, self.neutral_macro, market)
        self.assertGreater(shocked["lgd"].iloc[0], base["lgd"].iloc[0])

    def test_commodity_shock_reaches_sme_lgd_and_market_losses(self):
        params = create_idiosyncratic_scenario(commodity_collapse=True, seed=7)
        market = scenario_to_market_data(params)
        sme = self.portfolio[self.portfolio["segment"] == "SME_Corporate"]
        base = calculate_sa_adjusted_lgd(sme, self.neutral_macro)
        shocked = calculate_sa_adjusted_lgd(sme, self.neutral_macro, market)
        self.assertLess(market["coal_price_change"], 0)
        self.assertTrue((shocked["lgd"].values > base["lgd"].values).any())
        plain_market = compute_market_risk_ecap(10_000_000, n_sims=500, seed=11)
        shock_market = compute_market_risk_ecap(
            10_000_000, n_sims=500, seed=11, scenario_shocks=market
        )
        self.assertGreater(shock_market["market_ecap"], plain_market["market_ecap"])

    def test_cyber_shock_reaches_operational_simulation(self):
        params = create_idiosyncratic_scenario(cyber_incident=True, seed=7)
        macro = scenario_to_macro_conditions(params)
        plain = compute_operational_ecap(10_000_000, n_scenarios=100, seed=11)
        shocked = compute_operational_ecap(
            10_000_000, n_scenarios=100, seed=11,
            operational_shock_multiplier=macro["operational_shock_ecap_mult"],
        )
        self.assertEqual(macro["operational_shock_ecap_mult"], 2.5)
        self.assertGreater(shocked["operational_ecap"], plain["operational_ecap"])

    def test_housing_shock_reaches_mortgage_collateral_lgd(self):
        params = create_idiosyncratic_scenario(housing_crash=True, seed=7)
        market = scenario_to_market_data(params)
        mortgage = self.portfolio[
            self.portfolio["segment"] == "Retail_Mortgage"
        ].head(1)
        base = calculate_sa_adjusted_lgd(mortgage, self.neutral_macro)
        shocked = calculate_sa_adjusted_lgd(mortgage, self.neutral_macro, market)
        self.assertEqual(market["jse_property_change"], -0.30)
        self.assertGreater(shocked["lgd"].iloc[0], base["lgd"].iloc[0])

    def test_sme_failure_wave_reaches_sme_pit_pd_only(self):
        params = create_idiosyncratic_scenario(smes_failure_wave=True, seed=7)
        macro = scenario_to_macro_conditions(params)
        sme = self.portfolio[self.portfolio["segment"] == "SME_Corporate"].head(1)
        same_scenario_without_specific_multiplier = {**macro, "sme_pd_multiplier": 1.0}
        base = convert_ttc_to_pit(sme, same_scenario_without_specific_multiplier)
        shocked = convert_ttc_to_pit(sme, macro)
        self.assertEqual(macro["sme_pd_multiplier"], 2.5)
        self.assertGreater(shocked["pit_pd_12m"].iloc[0], base["pit_pd_12m"].iloc[0])
        retail = self.portfolio[self.portfolio["segment"] == "Retail_Mortgage"].head(1)
        retail_base = convert_ttc_to_pit(retail, same_scenario_without_specific_multiplier)
        retail_shocked = convert_ttc_to_pit(retail, macro)
        self.assertAlmostEqual(
            retail_shocked["pit_pd_12m"].iloc[0], retail_base["pit_pd_12m"].iloc[0]
        )


if __name__ == "__main__":
    unittest.main()
