"""Sanity tests asserting the engine matches the averaged SA D-SIB benchmarks.

Targets (D_SIB_AVERAGE profile, base scenario):
    - EAD equals the requested target portfolio size
    - RWA density ~48% (benchmark range 40-57%)
    - ECL/EAD ~1.5% (benchmark range 1.2-2.0%)
    - CET1 ~11.6% (range 11-13%), Tier 1 ~13.6%, Total CAR ~15.5%
    - Leverage ratio ~6.4% (range 5-8%)
    - Output floor mechanics and D-SIB buffer tiers
"""

import unittest

import numpy as np
import pandas as pd

from config.params import BANK_BENCHMARK_PROFILES, D_SIB_BUFFER_TIERS
from data.acquisition import scale_portfolio_to_target
from ecap.copula_mc import compute_convergence_diagnostics, simulate_copula_defaults
from ifrs9.pd_model import convert_ttc_to_pit, logit_pit_transform
from ifrs9.staging import calculate_ecl
from regcap.capital_stack import d_sib_buffer_for_bucket
from regcap.rwa_engine import apply_output_floor, compute_total_rwa, vasicek_asrf
from run import run_engine_end_to_end


class EngineBenchmarkTests(unittest.TestCase):
    """End-to-end run must reproduce the averaged D-SIB benchmark profile."""

    @classmethod
    def setUpClass(cls):
        cls.profile = BANK_BENCHMARK_PROFILES["D_SIB_AVERAGE"]
        cls.result = run_engine_end_to_end(
            scenario="Base",
            n_accounts=1200,
            n_mc_sims=400,
            seed=2024,
        )

    def test_ead_matches_target_portfolio_size(self):
        target = self.profile["total_exposure_bn"] * 1e9
        ead = self.result["ifrs9"]["ead_total"]
        self.assertAlmostEqual(ead / target, 1.0, places=4)

    def test_rwa_density_in_benchmark_range(self):
        density = self.result["rwa_density"]
        self.assertGreaterEqual(density, 0.40)
        self.assertLessEqual(density, 0.57)
        self.assertAlmostEqual(density, self.profile["rwa_density"], delta=0.03)

    def test_ecl_ead_ratio_in_benchmark_range(self):
        ratio = self.result["ifrs9"]["ecl_ead_ratio"]
        self.assertGreaterEqual(ratio, 0.012)
        self.assertLessEqual(ratio, 0.020)
        self.assertAlmostEqual(ratio, self.profile["ecl_ead_target"], delta=0.002)

    def test_capital_ratios_match_dsib_average(self):
        ratios = self.result["regcap"]["capital_ratios"]
        self.assertGreaterEqual(ratios["CET1 Ratio"], 0.11)
        self.assertLessEqual(ratios["CET1 Ratio"], 0.13)
        self.assertAlmostEqual(ratios["CET1 Ratio"], self.profile["cet1"], delta=0.005)
        self.assertAlmostEqual(ratios["Tier1 Ratio"], self.profile["tier1"], delta=0.005)
        self.assertAlmostEqual(ratios["Total CAR"], self.profile["total_capital"], delta=0.005)
        self.assertGreater(ratios["Tier1 Ratio"], ratios["CET1 Ratio"])
        self.assertGreater(ratios["Total CAR"], ratios["Tier1 Ratio"])

    def test_leverage_ratio_in_benchmark_range(self):
        ratios = self.result["regcap"]["capital_ratios"]
        self.assertGreaterEqual(ratios["Leverage Ratio"], 0.05)
        self.assertLessEqual(ratios["Leverage Ratio"], 0.08)
        self.assertAlmostEqual(ratios["Leverage Ratio"], self.profile["leverage"], delta=0.008)
        self.assertGreater(ratios["Leverage Ratio"], ratios["Leverage Requirement"])

    def test_output_floor_reported_and_consistent(self):
        floor = self.result["output_floor"]
        self.assertGreater(floor["standardised_rwa"], 0.0)
        self.assertGreaterEqual(floor["final_rwa"], floor["modelled_rwa"])
        self.assertGreaterEqual(floor["final_rwa"], floor["floor_rwa"])
        expected_applied = floor["floor_rwa"] > floor["modelled_rwa"]
        self.assertEqual(floor["floor_applied"], expected_applied)

    def test_rwa_composition_matches_benchmark_shares(self):
        bd = self.result["rwa_breakdown"]
        total = sum(bd.values())
        credit_share = bd["Credit RWA"] / total
        op_share = bd["Operational RWA"] / total
        market_share = bd["Market RWA"] / total
        self.assertAlmostEqual(credit_share, self.profile["credit_rwa_share"], delta=0.05)
        self.assertAlmostEqual(op_share, self.profile["oprisk_rwa_share"], delta=0.03)
        self.assertAlmostEqual(market_share, self.profile["market_rwa_share"], delta=0.03)

    def test_dsib_buffer_uses_tier_schedule(self):
        d_sib = self.result["regcap"]["capital_stack"]["d_sib"]
        self.assertIn(d_sib["bucket"], D_SIB_BUFFER_TIERS)
        self.assertEqual(d_sib["buffer_rate"], D_SIB_BUFFER_TIERS[d_sib["bucket"]])

    def test_calibration_metadata_is_audit_ready(self):
        calib = self.result["calibration"]
        self.assertGreater(calib["ead_scaling_factor"], 0.0)
        self.assertGreater(calib["pd_calibration_factor"], 0.0)
        self.assertGreater(calib["lgd_calibration_factor"], 0.0)
        self.assertAlmostEqual(
            calib["achieved_ecl_ead_ratio"], calib["target_ecl_ead_ratio"], delta=0.002
        )
        self.assertAlmostEqual(
            calib["achieved_credit_rwa_density"], calib["target_credit_rwa_density"], delta=0.02
        )

    def test_benchmark_comparison_includes_engine_and_all_banks(self):
        bench = self.result["benchmark_comparison"]
        entities = list(bench["Entity"])
        self.assertTrue(entities[0].startswith("Engine"))
        self.assertIn("D SIB AVERAGE", entities)
        self.assertIn("Nedbank", entities)
        self.assertEqual(len(entities), 1 + len(BANK_BENCHMARK_PROFILES))

    def test_severe_scenario_erodes_ratios_relative_to_base(self):
        severe = run_engine_end_to_end(
            scenario="Severe",
            n_accounts=1200,
            n_mc_sims=400,
            seed=2024,
        )
        base_ratios = self.result["regcap"]["capital_ratios"]
        sev_ratios = severe["regcap"]["capital_ratios"]
        self.assertGreater(
            severe["ifrs9"]["ecl_ead_ratio"], self.result["ifrs9"]["ecl_ead_ratio"]
        )
        self.assertGreater(severe["rwa_density"], self.result["rwa_density"])
        self.assertLess(sev_ratios["CET1 Ratio"], base_ratios["CET1 Ratio"])
        self.assertLess(sev_ratios["Tier1 Ratio"], base_ratios["Tier1 Ratio"])
        self.assertLess(sev_ratios["Total CAR"], base_ratios["Total CAR"])


class UnitMechanicsTests(unittest.TestCase):
    def test_vasicek_asrf_is_unexpected_loss_only(self):
        pd_val, lgd, corr = 0.01, 0.40, 0.15
        k = float(vasicek_asrf(
            np.array([pd_val]), np.array([lgd]), np.array([corr]),
            maturity_adj=False,
        )[0])
        from scipy.stats import norm
        cond_pd = norm.cdf((norm.ppf(pd_val) + np.sqrt(corr) * norm.ppf(0.999))
                           / np.sqrt(1.0 - corr))
        self.assertAlmostEqual(k, (cond_pd - pd_val) * lgd, places=10)
        self.assertLess(k, lgd)

    def test_output_floor_binds_when_modelled_below_floor(self):
        res = apply_output_floor(50.0, 100.0, floor_pct=0.725)
        self.assertTrue(res["floor_applied"])
        self.assertEqual(res["final_rwa"], 72.5)
        res2 = apply_output_floor(90.0, 100.0, floor_pct=0.725)
        self.assertFalse(res2["floor_applied"])
        self.assertEqual(res2["final_rwa"], 90.0)

    def test_dsib_tiers_cover_half_to_two_and_half_percent(self):
        self.assertEqual(d_sib_buffer_for_bucket(0), 0.0)
        self.assertEqual(d_sib_buffer_for_bucket(1), 0.005)
        self.assertEqual(d_sib_buffer_for_bucket(3), 0.015)
        self.assertEqual(d_sib_buffer_for_bucket(5), 0.025)
        self.assertEqual(d_sib_buffer_for_bucket(9), 0.025)

    def test_logit_pit_transform_is_bounded_and_monotone(self):
        pit_up = float(logit_pit_transform(0.02, 2.0))
        pit_down = float(logit_pit_transform(0.02, 0.5))
        self.assertGreater(pit_up, 0.02)
        self.assertLess(pit_down, 0.02)
        extreme = float(logit_pit_transform(0.5, 1e9))
        self.assertLessEqual(extreme, 0.99)

    def test_stage2_uses_lifetime_ecl(self):
        df = pd.DataFrame({
            "ead": [100.0, 100.0, 100.0],
            "pit_pd_12m": [0.02, 0.02, 0.02],
            "lifetime_pd": [0.10, 0.10, 0.10],
            "lgd": [0.5, 0.5, 0.5],
            "downturn_lgd": [0.6, 0.6, 0.6],
            "ifrs9_stage": [1, 2, 3],
        })
        out = calculate_ecl(df)
        self.assertAlmostEqual(out.loc[0, "ecl"], 100.0 * 0.02 * 0.5)
        self.assertAlmostEqual(out.loc[1, "ecl"], 100.0 * 0.10 * 0.5)
        self.assertAlmostEqual(out.loc[2, "ecl"], 100.0 * 0.10 * 0.5)

    def test_scale_portfolio_to_target_hits_exact_ead(self):
        df = pd.DataFrame({
            "ead": [100.0, 300.0],
            "principal_outstanding": [90.0, 250.0],
            "undrawn_limit": [20.0, 100.0],
        })
        scaled = scale_portfolio_to_target(df, 1000.0)
        self.assertAlmostEqual(float(scaled["ead"].sum()), 1000.0)
        self.assertAlmostEqual(
            float(scaled["principal_outstanding"].sum()), 340.0 * 2.5
        )
        self.assertAlmostEqual(scaled.attrs["exposure_scaling_factor"], 2.5)


class CopulaSanityTests(unittest.TestCase):
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
        self.assertIn("Vasicek ASRF", result["methodology"]["credit"])
        self.assertIn("output_floor", result["methodology"])
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
