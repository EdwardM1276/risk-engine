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
from ifrs9.staging import assign_ifrs9_staging, calculate_ecl
from regcap.rwa_engine import compute_credit_rwa

LGD_FACTOR_BOUNDS: Tuple[float, float] = (0.25, 4.0)
PD_FACTOR_BOUNDS: Tuple[float, float] = (0.25, 4.0)


def solve_lgd_calibration_factor(
    ifrs9_df: pd.DataFrame,
    target_credit_density: float,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Dict[str, float]:
    """Solve the LGD scalar so IRB credit RWA / EAD hits the target density."""
    ead_total = float(ifrs9_df["ead"].sum())
    def density(factor_value: float) -> float:
        credit_df = compute_credit_rwa(ifrs9_df, lgd_calibration_factor=factor_value)
        return float(credit_df["credit_rwa"].sum()) / max(ead_total, 1.0)

    lower, upper = LGD_FACTOR_BOUNDS
    low_value, high_value = density(lower), density(upper)
    attainable = min(low_value, high_value) - tol <= target_credit_density <= max(low_value, high_value) + tol
    iterations = 0
    if attainable:
        lo, hi = lower, upper
        for iterations in range(1, max_iter + 1):
            factor = (lo + hi) / 2.0
            achieved = density(factor)
            if abs(achieved - target_credit_density) <= tol:
                break
            if (achieved < target_credit_density) == (low_value < high_value):
                lo = factor
            else:
                hi = factor
        factor = (lo + hi) / 2.0
    else:
        factor = lower if abs(low_value - target_credit_density) <= abs(high_value - target_credit_density) else upper
    achieved = density(factor)
    return {
        "lgd_calibration_factor": float(factor),
        "target_credit_rwa_density": float(target_credit_density),
        "achieved_credit_rwa_density": float(achieved),
        "initial_credit_rwa_density": float(density(1.0)),
        "credit_rwa_density_residual": float(achieved - target_credit_density),
        "lgd_factor_bounds": list(LGD_FACTOR_BOUNDS),
        "credit_rwa_target_attainable": bool(attainable),
        "credit_rwa_calibration_converged": bool(abs(achieved - target_credit_density) <= tol),
        "credit_rwa_calibration_iterations": int(iterations),
        "credit_rwa_calibration_adjusted_parameter": "ordinary LGD calibration scalar used by prototype RWA engine",
    }


def apply_pd_calibration(
    ifrs9_df: pd.DataFrame,
    factor: float,
    macro_conditions: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply a PD calibration scalar to PIT/lifetime PDs and recompute ECL.

    When macro conditions are supplied, recompute SICR stages after applying
    the calibrated PDs and then recompute all dependent ECL columns.
    """
    df = ifrs9_df.copy()
    df["pit_pd_12m"] = np.clip(df["pit_pd_12m"].astype(float) * factor, 5e-5, 0.99)
    df["lifetime_pd"] = np.clip(df["lifetime_pd"].astype(float) * factor, 5e-5, 0.99)
    df["lifetime_pd"] = np.maximum(df["lifetime_pd"], df["pit_pd_12m"])
    if macro_conditions is not None:
        df = assign_ifrs9_staging(df, macro_conditions)
    return calculate_ecl(df)


def calibrate_ecl_to_target(
    ifrs9_df: pd.DataFrame,
    target_ecl_ead_ratio: float,
    macro_conditions: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Solve and apply a bounded PD scalar against the configured ECL/EAD target.

    With macro conditions, each trial recalculates staging; because stage
    changes are discrete, the exact aggregate target may be unreachable.
    """
    ead_total = float(ifrs9_df["ead"].sum())
    raw_ratio = float(ifrs9_df["ecl"].sum()) / max(ead_total, 1.0)
    lower, upper = PD_FACTOR_BOUNDS

    def achieved_at(candidate: float) -> float:
        candidate_df = apply_pd_calibration(ifrs9_df, candidate, macro_conditions)
        return float(candidate_df["ecl"].sum()) / max(ead_total, 1.0)

    low_ratio, high_ratio = achieved_at(lower), achieved_at(upper)
    attainable = min(low_ratio, high_ratio) <= target_ecl_ead_ratio <= max(low_ratio, high_ratio)
    raw_factor = target_ecl_ead_ratio / max(raw_ratio, 1e-9)
    candidates = [(lower, low_ratio), (upper, high_ratio)]
    if attainable:
        lo, hi = lower, upper
        for _ in range(40):
            midpoint = (lo + hi) / 2.0
            ratio = achieved_at(midpoint)
            candidates.append((midpoint, ratio))
            if abs(ratio - target_ecl_ead_ratio) <= 1e-7:
                break
            if ratio < target_ecl_ead_ratio:
                lo = midpoint
            else:
                hi = midpoint
    factor, achieved = min(
        candidates, key=lambda pair: abs(pair[1] - target_ecl_ead_ratio)
    )
    df = apply_pd_calibration(ifrs9_df, factor, macro_conditions)
    achieved = float(df["ecl"].sum()) / max(ead_total, 1.0)
    return df, {
        "pd_calibration_factor": factor,
        "raw_pd_calibration_factor": float(raw_factor),
        "raw_ecl_ead_ratio": raw_ratio,
        "target_ecl_ead_ratio": float(target_ecl_ead_ratio),
        "achieved_ecl_ead_ratio": achieved,
        "ecl_ead_ratio_residual": float(achieved - target_ecl_ead_ratio),
        "pd_factor_bounds": list(PD_FACTOR_BOUNDS),
        "ecl_target_attainable": bool(attainable),
        "ecl_calibration_converged": bool(abs(achieved - target_ecl_ead_ratio) <= 1e-6),
        "pd_calibration_adjusted_parameter": "PIT/lifetime PD scalar; staging recomputed when macro conditions are supplied",
    }


def calibration_targets_for_profile(bank_profile: str = DEFAULT_BANK_PROFILE) -> Dict[str, float]:
    """Benchmark calibration targets implied by a bank profile."""
    profile = BANK_BENCHMARK_PROFILES.get(bank_profile, BANK_BENCHMARK_PROFILES[DEFAULT_BANK_PROFILE])
    return {
        "rwa_density": float(profile["rwa_density"]),
        "credit_rwa_density": float(profile["rwa_density"]) * float(profile["credit_rwa_share"]),
        "ecl_ead_ratio": float(profile["ecl_ead_target"]),
    }
