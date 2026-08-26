import unittest
import pandas as pd
import numpy as np
from regcap.rwa_engine import compute_total_rwa

class RiskSanityTests(unittest.TestCase):
    def test_calibrated_book_matches_nedbank_pillar3(self):
        # Verify synthetic book reproduces published Nedbank 2024 aggregates
        from data.calibration import calibrate_synthetic_book
        
        target = {
            "total_exposure_bn": 1100,
            "rwa_density_pct": 59,
            "ecl_ratio_pct": 1.6,
            "segment_weights": None
        }

        df = calibrate_synthetic_book(target, n_accounts=1000)
        
        from data.calibration_helpers import compute_rwa_density
        actual_density = compute_rwa_density(df) * 100
        actual_ead_bn = (df['principal'] + df['undrawn']*0.75).sum() / 1e9

        # Assertions based on Nedbank 2024 Pillar 3 report targets
        assert abs(actual_density - 59) <= 5
        assert abs(actual_ead_bn - 1100) <= 25