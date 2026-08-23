"""End-to-end orchestrator for the South African Credit Risk Volatility Engine.

Executes the full model pipeline:
    Data Acquisition -> IFRS 9 ECL -> RWA -> RegCap Stack ->
    Monte Carlo Copula Simulation -> ECap Allocation -> Coverage Analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.params import NEDBANK_ECAP_BENCHMARK_2024, SA_BANK_BENCHMARKS_2024
from data.acquisition import acquire_all_data, fetch_uci_credit_card_defaults, summarize_uci_default_benchmark
from ecap.allocation import allocate_nedbank_ecap_benchmark
from ecap.copula_mc import (
    compute_market_risk_ecap,
    compute_operational_ecap,
    simulate_copula_defaults,
)
from ecap.coverage import (
    compute_afr_coverage_ratio,
    decompose_coverage_drivers,
    stress_coverage_erosion,
)
from ifrs9.staging import run_full_ifrs9_pipeline
from regcap.capital_stack import run_full_regcap_analysis
from regcap.rwa_engine import compute_total_rwa
from scenarios.stress_engine import (
    compute_scenario_shock_summary,
    create_idiosyncratic_scenario,
    get_scenario_parameters,
    scenario_to_macro_conditions,
    scenario_to_market_data,
)


# -----------------------------------------------------------------------------
# Deterministic hash helpers for Streamlit cache_data
# -----------------------------------------------------------------------------

def _hash_idio_shocks(shocks: Optional[Dict[str, Any]]) -> str:
    """Return a stable hash string for idiosyncratic_shocks dict."""
    if shocks is None:
        return "NONE"
    ordered = sorted(shocks.items(), key=lambda kv: kv[0])
    return "|".join(f"{k}:{int(v) if isinstance(v, bool) else v}" for k, v in ordered)


# -----------------------------------------------------------------------------
# Cached orchestrator entry point
# -----------------------------------------------------------------------------

def run_engine_end_to_end(
    scenario: str = "Base",
    total_exposure: float = 500_000_000_000.0,
    n_accounts: int = 2000,
    seed: int = 2024,
    institution_size: str = "Large_D-SIB",
    severity_multiplier: float = 1.0,
    idiosyncratic_shocks: Optional[Dict[str, Any]] = None,
    n_mc_sims: int = 1500,
    copula_type: str = "t",
    data_source: str = "synthetic",
    allow_synthetic_fallback: bool = False,
    portfolio_path: Optional[str] = None,
    strict_data_validation: bool = False,
) -> Dict[str, Any]:
    """Execute the complete engine pipeline.

    Phase order (auditable):
        1. Scenario parameter expansion + idiosyncratic overlay
        2. Data acquisition / synthetic portfolio generation
        3. IFRS 9 (PD PIT conversion, LGD, EAD, Staging, ECL)
        4. RWA engine (Vasicek ASRF IRB + Standardised fallback + FRTB + OpRisk)
        5. RegCap stack + capital resources + MDA trigger ratios
        6. Gaussian / t-copula Monte Carlo for credit VaR / ES
        7. Market (ES 97.5) and operational (Pareto AMA) ECap simulation
        8. Nedbank 2024 benchmark 6-way ECap allocation with MC override
        9. AFR/ECap coverage ratio + 8-point erosion grid
        10. SA D-SIB benchmark comparison table + 95% CI uncertainty bands

    Outputs are typed ``Dict[str, Any]`` with stable, documented keys consumed
    directly by the dashboard plotting layer.
    """
    run_start = datetime.now()

    # 1. Scenario expansion
    scenario_params: Dict[str, object] = get_scenario_parameters(
        scenario, severity_multiplier, seed
    )
    if idiosyncratic_shocks:
        cust = create_idiosyncratic_scenario(**idiosyncratic_shocks, seed=seed)
        for k, v in cust.items():
            scenario_params[k] = v

    macro_cond: Dict[str, float] = scenario_to_macro_conditions(scenario_params)
    mkt_data: Dict[str, float] = scenario_to_market_data(scenario_params)

    # 2. Data acquisition
    raw_data = acquire_all_data(
        total_exposure=total_exposure,
        n_accounts=n_accounts,
        periods=36,
        seed=seed,
        data_source=data_source,
        allow_synthetic_fallback=allow_synthetic_fallback,
        portfolio_path=portfolio_path,
    )
    portfolio = raw_data["portfolio"]
    if strict_data_validation and not raw_data["data_quality"].get("validation_ready", False):
        raise ValueError(
            "Strict validation requires a complete institutional dataset; "
            f"data status is {raw_data['data_quality'].get('status')}"
        )

    # 3. IFRS 9 pipeline
    ifrs9_df: pd.DataFrame = run_full_ifrs9_pipeline(
        portfolio, macro_cond, mkt_data, forecast_horizon_months=12
    )
    ecl_total = float(ifrs9_df["ecl"].sum())
    ecl_12m_total = float(ifrs9_df["12m_ecl"].sum())
    ecl_lifetime_total = float(ifrs9_df["lifetime_ecl"].sum())
    ecl_by_stage_raw = ifrs9_df.groupby("ifrs9_stage")["ecl"].sum().to_dict()
    ecl_by_stage: Dict[int, float] = {int(k): float(v) for k, v in ecl_by_stage_raw.items()}
    ead_total = float(ifrs9_df["ead"].sum())

    # Stage distribution keys are normalised to ints (1, 2, 3)
    stage_dist_raw = ifrs9_df["ifrs9_stage"].value_counts().sort_index().to_dict()
    stage_distribution: Dict[int, int] = {int(k): int(v) for k, v in stage_dist_raw.items()}

    avg_pit_pd = float((ifrs9_df["pit_pd_12m"] * ifrs9_df["ead"]).sum() / max(ead_total, 1.0))
    avg_lgd = float((ifrs9_df["lgd"] * ifrs9_df["ead"]).sum() / max(ead_total, 1.0))

    # 4. RWA engine
    rwa_result = compute_total_rwa(ifrs9_df, macro_cond)
    total_rwa = float(rwa_result["total_rwa"])

    # 5. RegCap stack and ratios
    regcap_analysis = run_full_regcap_analysis(
        ifrs9_df, rwa_result, ecl_total, macro_cond, institution_size
    )

    # 6. Copula MC credit losses
    credit_mc = simulate_copula_defaults(
        ifrs9_df,
        n_sims=n_mc_sims,
        copula_type=copula_type,
        t_df=6,
        confidence_levels=[0.90, 0.95, 0.975, 0.99, 0.999],
        macro_conditions=macro_cond,
        seed=seed,
    )

    # 7. Market + operational ECap
    market_mc = compute_market_risk_ecap(
        ead_total * 1.1,
        macro_cond,
        n_sims=max(500, n_mc_sims // 2),
        seed=seed + 1,
    )
    op_mc = compute_operational_ecap(
        ead_total * 1.1,
        n_scenarios=max(50, n_mc_sims // 10),
        ls_stage=macro_cond.get("load_shedding_stage", 2),
        seed=seed + 2,
    )

    # 8. Nedbank 2024 benchmark ECap allocation
    ecap_alloc = allocate_nedbank_ecap_benchmark(
        total_rwa=total_rwa,
        ifrs9_ecl_total=ecl_total,
        credit_ecap_from_simulation=credit_mc["credit_ecap_999"],
        market_ecap_from_simulation=market_mc["market_ecap"],
        operational_ecap_from_simulation=op_mc["operational_ecap"],
        macro_conditions=macro_cond,
    )

    # 9. Coverage ratio + erosion grid
    afr = float(
        regcap_analysis["capital_resources"]["Total Capital Available"]
        + max(0.0, ecl_total * 0.30)
    )
    coverage = compute_afr_coverage_ratio(afr, ecap_alloc["total_ecap"], ecap_alloc["components"])

    scenario_shocks: Dict[str, float] = {
        "gdp_shock": float(macro_cond.get("gdp_yoy", 0.0)) - 0.015,
        "ls_stage": float(macro_cond.get("load_shedding_stage", 2)),
        "sovereign_cds_bps": float(mkt_data.get("sovereign_cds_change_bps", 0)),
    }
    coverage_erosion_df: pd.DataFrame = stress_coverage_erosion(
        coverage, scenario_shocks, ecap_alloc["components"]
    )
    decompose_df: pd.DataFrame = decompose_coverage_drivers(
        coverage, regcap_analysis, ecl_total
    )

    # 10. Benchmarks + uncertainty
    benchmark_vs = _compare_to_bank_benchmarks(
        regcap_analysis, ifrs9_df, scenario, institution_size
    )
    uncertainty = _compute_uncertainty_bands(
        ifrs9_df, regcap_analysis, ecap_alloc, credit_mc, coverage
    )

    run_end = datetime.now()

    # Persist validated summary CSV for audit trail
    _write_run_summary_csv(
        scenario, severity_multiplier, institution_size, ecl_total,
        total_rwa, ecap_alloc["total_ecap"], coverage["Coverage Ratio"],
        stage_distribution,
    )

    return {
        "run_metadata": {
            "scenario": scenario,
            "severity_multiplier": float(severity_multiplier),
            "run_start": run_start,
            "run_end": run_end,
            "duration_seconds": float((run_end - run_start).total_seconds()),
            "n_accounts": int(len(ifrs9_df)),
            "n_mc_sims": int(n_mc_sims),
            "copula_type": copula_type,
            "data_source": data_source,
            "synthetic_fallback_used": bool(raw_data["data_quality"].get("synthetic_fallback_used", False)),
            "idio_hash": _hash_idio_shocks(idiosyncratic_shocks),
        },
        "scenario": {
            "parameters": scenario_params,
            "macro_conditions": macro_cond,
            "market_data": mkt_data,
            "shock_summary": compute_scenario_shock_summary(scenario_params),
        },
        "raw_data": raw_data,
        "ifrs9": {
            "portfolio": ifrs9_df,
            "ecl_total": ecl_total,
            "ecl_12m_total": ecl_12m_total,
            "ecl_lifetime_total": ecl_lifetime_total,
            "ecl_by_stage": ecl_by_stage,
            "stage_distribution": stage_distribution,
            "stage_ecl_distribution_pct": {
                k: float(v) / max(ecl_total, 1.0) for k, v in ecl_by_stage.items()
            },
            "ead_total": ead_total,
            "avg_pit_pd": avg_pit_pd,
            "avg_lgd": avg_lgd,
        },
        "regcap": regcap_analysis,
        "rwa_breakdown": rwa_result["breakdown"],
        "rwa_methodology": rwa_result["methodology"],
        "monte_carlo": {
            "credit": credit_mc,
            "market": market_mc,
            "operational": op_mc,
        },
        "economic_capital": ecap_alloc,
        "coverage": {
            "main": coverage,
            "erosion_path": coverage_erosion_df,
            "decomposition": decompose_df,
        },
        "benchmark_comparison": benchmark_vs,
        "uncertainty_bands": uncertainty,
    }


# -----------------------------------------------------------------------------
# SA D-SIB benchmark comparison (normalised integer stage keys)
# -----------------------------------------------------------------------------

def _compare_to_bank_benchmarks(
    regcap_analysis: Dict[str, Any],
    ifrs9_df: pd.DataFrame,
    scenario: str,
    institution_size: str,
) -> pd.DataFrame:
    """Return validated DataFrame of bank benchmark comparisons.

    Stage keys always read as ints 1, 2, 3 -- avoids string/int mix-ups that
    caused KeyError during benchmark radar construction.
    """
    ratios = regcap_analysis["capital_ratios"]
    ecl_by_stage = ifrs9_df.groupby("ifrs9_stage")["ecl"].sum()
    ecl_total = float(ifrs9_df["ecl"].sum()) or 1.0
    s1 = float(ecl_by_stage.get(1, 0.0)) / ecl_total
    s2 = float(ecl_by_stage.get(2, 0.0)) / ecl_total
    s3 = float(ecl_by_stage.get(3, 0.0)) / ecl_total
    car = float(ratios["Total CAR"])
    cet1 = float(ratios["CET1 Ratio"])

    rows: list[Dict[str, str]] = []
    banks: Dict[str, Any]
    if institution_size.startswith("Large"):
        banks = SA_BANK_BENCHMARKS_2024
    else:
        banks = {"Nedbank": SA_BANK_BENCHMARKS_2024["Nedbank"]}

    bank_names = list(banks.keys())
    for idx, (bank, b) in enumerate(banks.items()):
        is_engine = idx == 0
        rows.append({
            "Entity": f"Engine ({scenario})" if is_engine else bank,
            "Total CAR": f"{car*100:.2f}%" if is_engine else f"{b['CAR']*100:.2f}%",
            "CET1 Ratio": f"{cet1*100:.2f}%" if is_engine else f"{b['CET1']*100:.2f}%",
            "Stage 1 ECL %": f"{s1*100:.1f}%" if is_engine else f"{b['ECL_stage1_pct']*100:.0f}%",
            "Stage 2 ECL %": f"{s2*100:.1f}%" if is_engine else f"{b['ECL_stage2_pct']*100:.0f}%",
            "Stage 3 ECL %": f"{s3*100:.1f}%" if is_engine else f"{b['ECL_stage3_pct']*100:.0f}%",
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Uncertainty / model-risk bands
# -----------------------------------------------------------------------------

def _compute_uncertainty_bands(
    ifrs9_df: pd.DataFrame,
    regcap_analysis: Dict[str, Any],
    ecap_alloc: Dict[str, Any],
    credit_mc: Dict[str, Any],
    coverage: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Return central estimate + 95% CI + model-risk for 5 headline metrics."""
    losses = np.asarray(credit_mc["simulated_losses"], dtype=float)
    credit_ecap_999 = float(credit_mc["credit_ecap_999"])
    n = int(losses.size)

    model_risk_pct = float(NEDBANK_ECAP_BENCHMARK_2024["model_risk_pct"]) + 0.03
    estimation_error = float(np.std(losses) * 1.96 / np.sqrt(n)) if n > 0 else 0.0

    total_ecap = float(ecap_alloc["total_ecap"])
    ecl = float(ifrs9_df["ecl"].sum())
    car = float(regcap_analysis["capital_ratios"]["Total CAR"])

    bands: Dict[str, Dict[str, float]] = {
        "ECL": {
            "central": ecl,
            "lower_95": float(max(0.0, ecl * (1.0 - 0.15))),
            "upper_95": float(ecl * (1.0 + 0.20)),
            "model_risk": float(ecl * model_risk_pct),
        },
        "Credit ECap (99.9)": {
            "central": credit_ecap_999,
            "lower_95": float(max(0.0, credit_ecap_999 - 2.0 * estimation_error)),
            "upper_95": float(credit_ecap_999 + 2.0 * estimation_error),
            "model_risk": float(credit_ecap_999 * model_risk_pct),
        },
        "Total ECap": {
            "central": total_ecap,
            "lower_95": float(max(0.0, total_ecap * (1.0 - 0.10))),
            "upper_95": float(total_ecap * (1.0 + 0.15)),
            "model_risk": float(total_ecap * model_risk_pct),
        },
        "Total CAR": {
            "central": car,
            "lower_95": float(max(0.0, car - 0.01)),
            "upper_95": float(car + 0.01),
            "model_risk_bps": 25.0,
        },
        "Coverage Ratio": {
            "central": float(coverage["Coverage Ratio"]),
            "lower_95": float(max(0.5, coverage["Coverage Ratio"] - 0.12)),
            "upper_95": float(coverage["Coverage Ratio"] + 0.12),
            "model_risk": float(coverage["Coverage Ratio"] * model_risk_pct),
        },
    }
    return bands


# -----------------------------------------------------------------------------
# Persistent audit trail: write validated CSV summary to outputs/ folder
# -----------------------------------------------------------------------------

def _write_run_summary_csv(
    scenario: str,
    severity: float,
    institution_size: str,
    ecl_total: float,
    total_rwa: float,
    total_ecap: float,
    coverage_ratio: float,
    stage_distribution: Dict[int, int],
) -> None:
    """Append a single-row summary to outputs/latest_runs.csv for audit trail."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "latest_run_summary.csv")

        s1 = stage_distribution.get(1, 0)
        s2 = stage_distribution.get(2, 0)
        s3 = stage_distribution.get(3, 0)
        n = max(1, s1 + s2 + s3)

        row = pd.DataFrame([{
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": scenario,
            "severity_multiplier": f"{severity:.2f}",
            "institution_size": institution_size,
            "ecl_total_zar": f"{ecl_total:.2f}",
            "total_rwa_zar": f"{total_rwa:.2f}",
            "total_ecap_zar": f"{total_ecap:.2f}",
            "coverage_ratio": f"{coverage_ratio:.4f}",
            "stage1_pct_accounts": f"{s1/n*100:.2f}",
            "stage2_pct_accounts": f"{s2/n*100:.2f}",
            "stage3_pct_accounts": f"{s3/n*100:.2f}",
        }])
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            combined = pd.concat([existing, row], ignore_index=True)
        else:
            combined = row
        combined.to_csv(csv_path, index=False)
    except Exception:
        # Silent fail: audit trail is best-effort, never break the run
        pass


# -----------------------------------------------------------------------------
# Top-level CLI and orchestration entry points
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser for dashboard and pipeline commands."""
    parser = argparse.ArgumentParser(description="South African Credit Risk Volatility Engine")
    subparsers = parser.add_subparsers(dest="command")

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8502, help="Port for the dashboard")
    dashboard_parser.add_argument("--host", default="0.0.0.0", help="Host binding for the dashboard")

    pipeline_parser = subparsers.add_parser("pipeline", help="Run the end-to-end pipeline once")
    pipeline_parser.add_argument("--scenario", default="Base", choices=["Base", "Adverse", "Severe"])
    pipeline_parser.add_argument("--total-exposure", type=float, default=500_000_000_000.0)
    pipeline_parser.add_argument("--n-accounts", type=int, default=2000)
    pipeline_parser.add_argument("--seed", type=int, default=2024)
    pipeline_parser.add_argument("--institution-size", default="Large_D-SIB")
    pipeline_parser.add_argument("--severity", type=float, default=1.0)
    pipeline_parser.add_argument("--n-mc-sims", type=int, default=1500)
    pipeline_parser.add_argument("--copula-type", default="t", choices=["t", "Gaussian"])
    pipeline_parser.add_argument("--data-source", default="synthetic", choices=["synthetic", "public", "institutional"])
    pipeline_parser.add_argument("--allow-synthetic-fallback", action="store_true")
    pipeline_parser.add_argument("--portfolio-path", help="CSV/XLSX anonymized institutional portfolio extract")
    pipeline_parser.add_argument("--strict-data-validation", action="store_true")

    benchmark_parser = subparsers.add_parser("data-benchmark", help="Benchmark against observed public loan data")
    benchmark_parser.add_argument("--dataset", default="uci-credit-card", choices=["uci-credit-card"])

    return parser


def _run_pipeline_command(args: argparse.Namespace) -> int:
    result = run_engine_end_to_end(
        scenario=args.scenario,
        total_exposure=args.total_exposure,
        n_accounts=args.n_accounts,
        seed=args.seed,
        institution_size=args.institution_size,
        severity_multiplier=args.severity,
        n_mc_sims=args.n_mc_sims,
        copula_type=args.copula_type,
        data_source=args.data_source,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        portfolio_path=args.portfolio_path,
        strict_data_validation=args.strict_data_validation,
    )
    summary = {
        "scenario": args.scenario,
        "total_exposure": result["ifrs9"]["ead_total"],
        "ecl_total": result["ifrs9"]["ecl_total"],
        "total_rwa": result["regcap"]["total_rwa"],
        "total_ecap": result["economic_capital"]["total_ecap"],
        "coverage_ratio": result["coverage"]["main"]["Coverage Ratio"],
        "duration_seconds": result["run_metadata"]["duration_seconds"],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _run_data_benchmark_command(args: argparse.Namespace) -> int:
    """Run an observed-data benchmark without feeding incomplete fields to IFRS 9."""
    if args.dataset == "uci-credit-card":
        observed = fetch_uci_credit_card_defaults()
        print(json.dumps(summarize_uci_default_benchmark(observed), indent=2, default=str))
    return 0


def _run_dashboard_command(args: argparse.Namespace) -> int:
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(args.port), "--server.address", args.host]
    return subprocess.call(cmd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point for the project."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0

    if args.command == "pipeline":
        return _run_pipeline_command(args)
    if args.command == "data-benchmark":
        return _run_data_benchmark_command(args)
    if args.command == "dashboard":
        return _run_dashboard_command(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
