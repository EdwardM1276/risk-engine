"""Benchmark calibration methodology.

Two documented central-tendency calibration scalars align the synthetic book
with the averaged SA D-SIB benchmark profile:

1. ``solve_lgd_calibration_factor`` -- a single multiplicative scalar on
   regulatory LGD, solved so portfolio IRB credit RWA density matches the
   benchmark credit-RWA density (profile RWA density x credit RWA share).
   IRB K is linear in LGD, so the solve is a fixed-point iteration that
   converges in a few steps (bounds guard the LGD clip at 1.0).

2. ``calibrate_ecl_to_target`` -- a single multiplicative scalar on PIT and
   lifetime PDs, solved so total ECL / total EAD matches the benchmark
   ECL/EAD target (~1.5%). ECL is linear in PD, so the scalar is the ratio
   of target to modelled ECL, re-applied through the ECL engine.

Both scalars are bounded, reported in the run metadata, and asserted in
tests/test_risk_sanity.py for auditability.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config.params import BANK_BENCHMARK_PROFILES, DEFAULT_BANK_PROFILE
from ifrs9.staging import calculate_ecl
from regcap.rwa_engine import compute_credit_rwa

LGD_FACTOR_BOUNDS: Tuple[float, float] = (0.25, 4.0)
PD_FACTOR_BOUNDS: Tuple[float, float] = (0.25, 4.0)


def solve_lgd_calibration_factor(
    ifrs9_df: pd.DataFrame,
    target_credit_density: float,
    max_iter: int = 6,
    tol: float = 1e-4,
) -> Dict[str, float]:
    """Solve the LGD scalar so IRB credit RWA / EAD hits the target density."""
    ead_total = float(ifrs9_df["ead"].sum())
    factor = 1.0
    achieved = float("nan")
    for _ in range(max_iter):
        credit_df = compute_credit_rwa(ifrs9_df, lgd_calibration_factor=factor)
        achieved = float(credit_df["credit_rwa"].sum()) / max(ead_total, 1.0)
        if abs(achieved - target_credit_density) < tol:
            break
        factor = float(np.clip(factor * target_credit_density / max(achieved, 1e-9),
                               *LGD_FACTOR_BOUNDS))
    return {
        "lgd_calibration_factor": float(factor),
        "target_credit_rwa_density": float(target_credit_density),
        "achieved_credit_rwa_density": float(achieved),
    }


def apply_pd_calibration(ifrs9_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Apply a PD calibration scalar to PIT/lifetime PDs and recompute ECL.

    Staging is left unchanged (assigned from uncalibrated PDs); only the
    provision level is re-anchored.
    """
    df = ifrs9_df.copy()
    df["pit_pd_12m"] = np.clip(df["pit_pd_12m"].astype(float) * factor, 5e-5, 0.99)
    df["lifetime_pd"] = np.clip(df["lifetime_pd"].astype(float) * factor, 5e-5, 0.99)
    df["lifetime_pd"] = np.maximum(df["lifetime_pd"], df["pit_pd_12m"])
    return calculate_ecl(df)


def calibrate_ecl_to_target(
    ifrs9_df: pd.DataFrame,
    target_ecl_ead_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Solve and apply the PD scalar so total ECL / EAD hits the benchmark
    target (ECL is linear in PD)."""
    ead_total = float(ifrs9_df["ead"].sum())
    raw_ratio = float(ifrs9_df["ecl"].sum()) / max(ead_total, 1.0)
    factor = float(np.clip(target_ecl_ead_ratio / max(raw_ratio, 1e-9), *PD_FACTOR_BOUNDS))

    df = apply_pd_calibration(ifrs9_df, factor)
    achieved = float(df["ecl"].sum()) / max(ead_total, 1.0)
    return df, {
        "pd_calibration_factor": factor,
        "raw_ecl_ead_ratio": raw_ratio,
        "target_ecl_ead_ratio": float(target_ecl_ead_ratio),
        "achieved_ecl_ead_ratio": achieved,
    }


def calibration_targets_for_profile(bank_profile: str = DEFAULT_BANK_PROFILE) -> Dict[str, float]:
    """Benchmark calibration targets implied by a bank profile."""
    profile = BANK_BENCHMARK_PROFILES.get(bank_profile, BANK_BENCHMARK_PROFILES[DEFAULT_BANK_PROFILE])
    return {
        "rwa_density": float(profile["rwa_density"]),
        "credit_rwa_density": float(profile["rwa_density"]) * float(profile["credit_rwa_share"]),
        "ecl_ead_ratio": float(profile["ecl_ead_target"]),
    }
