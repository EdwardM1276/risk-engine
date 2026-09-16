import unittest
from datetime import date
from decimal import Decimal

import pandas as pd

from config import params
from engine.money import post_sum
from ifrs9.staging import calculate_ecl
from nca.in_duplum import (
    CONTROLLED_CATEGORIES,
    DefaultEpisode,
    apply_in_duplum_cap,
    accrue_with_episode,
)


class InDuplumTests(unittest.TestCase):
    def test_cap_open_partial_and_exhausted(self):
        partial = apply_in_duplum_cap(300, 10_000, 9_850)
        self.assertEqual(partial.recognised, Decimal("150.00"))
        self.assertEqual(partial.suppressed, Decimal("150.00"))
        exhausted = apply_in_duplum_cap(1, 10_000, 10_000)
        self.assertEqual(exhausted.recognised, Decimal("0.00"))
        self.assertEqual(exhausted.suppressed, Decimal("1.00"))
        open_cap = apply_in_duplum_cap(300, 10_000, 0)
        self.assertEqual(open_cap.recognised, Decimal("300.00"))

    def test_invalid_values_and_categories_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_in_duplum_cap(-1, 10_000, 0)
        with self.assertRaises(ValueError):
            apply_in_duplum_cap(1, 10_000, 0, "unknown")
        with self.assertRaises(ValueError):
            DefaultEpisode("A", date.today(), 100, "policy", {"unknown": 1})

    def test_categories_share_one_aggregate_pool(self):
        episode = DefaultEpisode("A", date.today(), 10_000, "nca-s1035-2026.1", {"fees": 9_900})
        result = accrue_with_episode(episode, "collection_costs", 200)
        self.assertEqual(result.recognised, Decimal("100.00"))
        self.assertEqual(set(CONTROLLED_CATEGORIES), {
            "arrears_interest", "fees", "credit_insurance", "default_admin_charges", "collection_costs"
        })

    def test_post_sum_rounds_once(self):
        self.assertEqual(post_sum([0.145, 0.145, 0.145]), Decimal("0.44"))

    def test_enabled_cap_cannot_reduce_stage_three_ecl(self):
        episode = DefaultEpisode("A", date.today(), 1_000, "nca-s1035-2026.1")
        frame = pd.DataFrame([{
            "ead": 1_000.0, "pit_pd_12m": 0.5, "lifetime_pd": 0.9,
            "lgd": 0.5, "downturn_lgd": 0.6, "ifrs9_stage": 3,
            "dpd": 3_600, "default_episode": episode,
        }])
        original = params.NCA_IN_DUPLUM_ENABLED
        try:
            params.NCA_IN_DUPLUM_ENABLED = False
            uncapped = calculate_ecl(frame).loc[0, "ecl"]
            params.NCA_IN_DUPLUM_ENABLED = True
            capped = calculate_ecl(frame).loc[0, "ecl"]
        finally:
            params.NCA_IN_DUPLUM_ENABLED = original
        self.assertGreaterEqual(capped, uncapped)


if __name__ == "__main__":
    unittest.main()
