import unittest
from datetime import date
from decimal import Decimal

import pandas as pd

from config.reference_data import RateRecord, RateSeries, RateUnavailable, load_rates
from ifrs9.pd_model import convert_ttc_to_pit


class ReferenceDataTests(unittest.TestCase):
    def test_effective_dated_lookup_and_boundaries(self):
        records = [
            RateRecord("X", Decimal("1"), date(2026, 1, 1), "source", date(2026, 1, 2), "A", 1),
            RateRecord("X", Decimal("2"), date(2026, 6, 1), "source", date(2026, 6, 2), "B", 2),
        ]
        series = RateSeries("X", records)
        self.assertEqual(series.at(date(2026, 1, 1)).value, Decimal("1"))
        self.assertEqual(series.at(date(2026, 5, 31)).value, Decimal("1"))
        self.assertEqual(series.at(date(2026, 6, 1)).value, Decimal("2"))
        self.assertEqual(series.latest().version, 2)
        with self.assertRaises(RateUnavailable):
            series.at(date(2025, 12, 31))

    def test_bad_status_and_duplicate_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            RateRecord("X", Decimal("1"), date.today(), "source", date.today(), "A", 1, "INVALID")
        record = RateRecord("X", Decimal("1"), date(2026, 1, 1), "source", date.today(), "A", 1)
        with self.assertRaises(ValueError):
            RateSeries("X", [record, record])

    def test_benchmark_transition_uplifts_only_non_stage_three_retail_pd(self):
        portfolio = pd.DataFrame({
            "segment": ["Retail_Mortgage", "Retail_Mortgage"],
            "loadshedding_vulnerability_score": [3, 3], "months_on_book": [24, 24],
            "internal_rating": ["BBB", "BBB"], "base_segment_ttc_pd": [0.015, 0.015],
            "dpd": [0, 120], "debt_review_flag": [False, False],
            "judgement_flag": [False, False], "administration_order": [False, False],
        })
        conditions = {
            "gdp_yoy": 0.015, "unemployment_rate": 0.32, "load_shedding_stage": 2,
            "cpi_yoy": 0.05, "repo_rate": 0.0775,
            "benchmark_transition": True, "as_of_date": date(2026, 9, 16),
            "benchmark_transition_effective_from": date(2026, 5, 1),
        }
        result = convert_ttc_to_pit(portfolio, conditions)
        base = convert_ttc_to_pit(portfolio, {**conditions, "benchmark_transition": False})
        self.assertGreater(result.loc[0, "pit_pd_12m"], base.loc[0, "pit_pd_12m"])
        self.assertEqual(result.loc[1, "pit_pd_12m"], base.loc[1, "pit_pd_12m"])

    def test_rates_file_loads(self):
        rates = load_rates()
        self.assertIn("SA_POLICY", rates)


if __name__ == "__main__":
    unittest.main()
