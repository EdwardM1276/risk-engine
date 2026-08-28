import unittest
from io import BytesIO
from zipfile import ZipFile
from unittest.mock import patch

import pandas as pd

from data.acquisition import (
    acquire_all_data,
    fetch_uci_credit_card_defaults,
    fetch_world_bank_macro_data,
    normalize_fdic_portfolio,
    summarize_uci_default_benchmark,
)


class AcquisitionTests(unittest.TestCase):
    def test_synthetic_mode_is_explicitly_labelled(self):
        result = acquire_all_data(periods=3, n_accounts=20, data_source="synthetic")
        self.assertEqual(result.data_quality["data_source"], "synthetic")
        self.assertFalse(result.data_quality["synthetic_fallback_used"])

    def test_invalid_source_is_rejected(self):
        with self.assertRaises(ValueError):
            acquire_all_data(data_source="unknown")
        with self.assertRaises(ValueError):
            acquire_all_data(data_source="institutional")

    def test_public_failure_does_not_silently_use_synthetic_data(self):
        with patch(
            "data.acquisition.fetch_world_bank_macro_data",
            side_effect=RuntimeError("offline"),
        ):
            with self.assertRaises(RuntimeError):
                acquire_all_data(data_source="public")

    def test_public_aggregate_data_is_not_marked_validation_ready(self):
        macro = pd.DataFrame({
            "date": [pd.Timestamp("2025-12-31")], "repo_rate": [None],
            "prime_rate": [None], "cpi_yoy": [0.05], "gdp_yoy": [0.01],
            "unemployment_rate": [0.30], "zar_usd": [18.0],
        })
        financials = pd.DataFrame([{
            "date": pd.Timestamp("2025-12-31"), "bank_name": "Example",
            "bank_cert": 1, "total_assets_usd": 100.0,
            "loan_balance_usd": 50.0, "deposits_usd": 80.0, "net_income_usd": 1.0,
        }])
        markets = pd.DataFrame({
            "date": [pd.Timestamp("2025-12-31")], "zar_usd": [18.0],
            "jse_alsi": [None], "jse_property": [None], "sovereign_cds_bps": [None],
            "gold_price_zar": [None], "platinum_price_zar": [None], "coal_price_zar": [None],
        })
        with patch("data.acquisition.fetch_world_bank_macro_data", return_value=macro), \
             patch("data.acquisition.fetch_sarb_current_rates", return_value={"repo_rate": 0.07, "prime_rate": 0.105}), \
             patch("data.acquisition.fetch_fdic_bank_financials", return_value=financials), \
             patch("data.acquisition.fetch_fred_exchange_rate", return_value=markets):
            result = acquire_all_data(periods=2, n_accounts=2, data_source="public")
        self.assertFalse(result.data_quality["validation_ready"])

    def test_fdic_rows_are_normalized_to_engine_contract(self):
        source = pd.DataFrame([{
            "date": pd.Timestamp("2025-12-31"),
            "bank_name": "Example Bank",
            "bank_cert": 12345,
            "total_assets_usd": 100_000_000,
            "loan_balance_usd": 60_000_000,
            "deposits_usd": 80_000_000,
            "net_income_usd": 1_000_000,
        }])
        result = normalize_fdic_portfolio(source)
        required = {
            "account_id", "segment", "principal_outstanding", "undrawn_limit",
            "collateral_value", "base_segment_ttc_pd", "base_segment_lgd",
        }
        self.assertTrue(required.issubset(result.columns))
        self.assertEqual(result.loc[0, "principal_outstanding"], 60_000_000)
        self.assertEqual(result.loc[0, "source_bank"], "Example Bank")

    def test_world_bank_percentages_are_converted_to_model_fractions(self):
        def response(url, params):
            values = {
                "NY.GDP.MKTP.KD.ZG": 2.0,
                "FP.CPI.TOTL.ZG": 5.0,
                "SL.UEM.TOTL.ZS": 30.0,
                "PA.NUS.FCRF": 18.0,
            }
            indicator = url.rsplit("/", 1)[-1]
            return [{}, [{"date": "2024", "value": values[indicator]}]]

        with patch("data.acquisition._get_json", side_effect=response):
            result = fetch_world_bank_macro_data(periods=1)

        self.assertAlmostEqual(result.loc[0, "gdp_yoy"], 0.02)
        self.assertAlmostEqual(result.loc[0, "cpi_yoy"], 0.05)
        self.assertAlmostEqual(result.loc[0, "unemployment_rate"], 0.30)

    def test_uci_credit_defaults_parser_preserves_observed_outcomes(self):
        source = pd.DataFrame([{
            "ID": 1,
            "LIMIT_BAL": 100000,
            "PAY_0": 2, "PAY_2": 1, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 50000, "BILL_AMT2": 50000, "BILL_AMT3": 50000,
            "BILL_AMT4": 50000, "BILL_AMT5": 50000, "BILL_AMT6": 50000,
            "default payment next month": 1,
        }])
        workbook = BytesIO()
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            source.to_excel(writer, index=False, startrow=1)
        archive = BytesIO()
        with ZipFile(archive, "w") as zipped:
            zipped.writestr("default of credit card clients.xlsx", workbook.getvalue())

        with patch("data.acquisition.fetch_binary_url", return_value=archive.getvalue()):
            result = fetch_uci_credit_card_defaults()

        self.assertEqual(len(result), 1)
        self.assertTrue(result.loc[0, "default_flag"])
        self.assertEqual(result.loc[0, "max_dpd_bucket"], 2)
        self.assertAlmostEqual(result.loc[0, "utilisation_proxy"], 0.5)

    def test_uci_benchmark_summary_uses_observed_outcomes(self):
        frame = pd.DataFrame({
            "default_flag": [False, True],
            "credit_limit": [100.0, 200.0],
            "max_dpd_bucket": [0, 2],
            "utilisation_proxy": [0.2, 0.8],
        })
        summary = summarize_uci_default_benchmark(frame)
        self.assertEqual(summary["observations"], 2)
        self.assertAlmostEqual(summary["overall_default_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()