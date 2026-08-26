import unittest
import pandas as pd
from data.calibration import calibrate_synthetic_book, compute_rwa_density

class CalibrationTests(unittest.TestCase):
    def test_calibrated_book_matches_pillar3_aggregates(self):
        target = {
            "total_exposure_bn": 1100,
            "rwa_density_pct": 59,
            "ecl_ratio_pct": 1.6,
            "segment_weights": None
        }
        df = calibrate_synthetic_book(target, n_accounts=2000)
        density = compute_rwa_density(df) * 100
        print(f"\nActual RWA Density: {density:.2f}%")
        self.assertTrue(55 <= density <= 65)