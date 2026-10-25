"""Build a tiny, deterministic sample package for the prototype documentation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import run_engine_end_to_end


OUT = ROOT / "sample_runs" / "sample_2026-10-25"
AS_OF = "2026-10-20"
CONFIG = {
    "scenario_set": ["Base", "Adverse", "Severe"],
    "total_exposure_zar": 100_000_000_000,
    "n_accounts": 100,
    "seed": 2024,
    "n_mc_sims": 300,
    "copula_type": "t",
    "t_df": 6,
    "data_source": "synthetic",
    "as_of_date": AS_OF,
    "bank_profile": "D_SIB_AVERAGE",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_dir = OUT / "input_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    summary_rows = []
    run_metadata = {}
    calibration_diagnostics = {}
    input_hashes = {}
    warnings = []

    for scenario in CONFIG["scenario_set"]:
        result = run_engine_end_to_end(
            scenario=scenario,
            total_exposure=CONFIG["total_exposure_zar"],
            n_accounts=CONFIG["n_accounts"],
            seed=CONFIG["seed"],
            n_mc_sims=CONFIG["n_mc_sims"],
            copula_type=CONFIG["copula_type"],
            t_df=CONFIG["t_df"],
            data_source=CONFIG["data_source"],
            as_of_date=CONFIG["as_of_date"],
            bank_profile=CONFIG["bank_profile"],
        )
        if scenario == "Base":
            raw = result["raw_data"]
            for name, value in raw.items():
                if isinstance(value, pd.DataFrame):
                    path = snapshot_dir / f"{name}.csv"
                    value.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.12g")
                    input_hashes[name] = {
                        "file": str(path.relative_to(OUT)),
                        "sha256": _sha256(path),
                        "rows": int(len(value)),
                        "columns": list(value.columns),
                    }
                elif isinstance(value, dict):
                    (snapshot_dir / f"{name}.json").write_text(
                        json.dumps(value, indent=2, sort_keys=True, default=str),
                        encoding="utf-8",
                    )
        meta = result["run_metadata"]
        calibration_diagnostics[scenario] = result["calibration"]
        run_metadata[scenario] = {
            "run_id": meta["run_id"],
            "config_digest": meta["config_digest"],
            "snapshot_path": meta["snapshot_path"],
        }
        credit = result["monte_carlo"]["credit"]
        matrix_diag = credit["correlation_matrix_diagnostics"]
        floor = result["output_floor"]
        coverage = result["coverage"]["main"]
        quality = result["raw_data"]["data_quality"]
        summary_rows.append({
            "scenario": scenario,
            "ead_zar": result["ifrs9"]["ead_total"],
            "ecl_zar": result["ifrs9"]["ecl_total"],
            "ecl_ead_ratio": result["ifrs9"]["ecl_ead_ratio"],
            "total_rwa_zar": result["regcap"]["total_rwa"],
            "rwa_density": result["rwa_density"],
            "cet1_ratio": result["regcap"]["capital_ratios"]["CET1 Ratio"],
            "total_ecap_indicator_zar": result["economic_capital"]["total_ecap"],
            "afr_ecap_coverage_ratio": coverage["Coverage Ratio"],
            "afr_basis": coverage["AFR basis"],
            "output_floor_pct": floor["floor_pct"],
            "output_floor_proxy_binding": floor["floor_applied"],
            "credit_tail_observations_999": credit["tail_observations_999"],
            "credit_tail_expected_observations_999": credit["expected_tail_observations_999"],
            "credit_tail_warning": credit["tail_estimate_warning"],
            "independent_credit_mc_replicates": credit["independent_replicates"],
            "credit_ecap_999_zar": credit["credit_ecap_999"],
            "correlation_matrix_repaired": matrix_diag["matrix_repaired"],
            "correlation_matrix_frobenius_adjustment": matrix_diag["frobenius_adjustment"],
            "correlation_matrix_warning": matrix_diag["warning"],
            "input_quality_status": quality["status"],
        })
        if credit["tail_estimate_warning"]:
            warnings.append(
                f"{scenario}: credit 99.9% tail is exploratory; only "
                f"{credit['tail_observations_999']} order-statistic observation(s) "
                "and one seed/replicate. No formal confidence interval is reported."
            )
        if not result["raw_data"]["data_quality"].get("validation_ready", False):
            warnings.append(f"{scenario}: synthetic input is not validation-ready.")
        if matrix_diag["warning"]:
            warnings.append(
                f"{scenario}: the factor correlation matrix required a material repair "
                f"(Frobenius adjustment {matrix_diag['frobenius_adjustment']:.6g})."
            )
        if not result["calibration"].get("ecl_target_attainable", True):
            warnings.append(f"{scenario}: configured ECL calibration target was not attainable.")
        if not result["calibration"].get("credit_rwa_target_attainable", True):
            warnings.append(f"{scenario}: configured credit RWA density target was not attainable.")
        source_snapshot = Path(meta["snapshot_path"])
        if source_snapshot.exists():
            target = snapshot_dir / f"run_request_{scenario.lower()}.json"
            target.write_bytes(source_snapshot.read_bytes())

    (OUT / "run_config.json").write_text(
        json.dumps(CONFIG, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUT / "scenario_results.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(OUT / "scenario_results.csv", index=False)
    (OUT / "calibration_diagnostics.json").write_text(
        json.dumps(calibration_diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )

    tracked = [
        "config/params.py", "config/rates.json", "data/acquisition.py", "run.py",
        "scenarios/stress_engine.py", "ifrs9/pd_model.py", "ifrs9/staging.py",
        "regcap/rwa_engine.py", "ecap/copula_mc.py", "ecap/allocation.py",
        "ecap/coverage.py",
    ]
    code_hashes = {
        path: _sha256(ROOT / path) for path in tracked if (ROOT / path).exists()
    }
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    manifest = {
        "package_schema_version": 1,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": CONFIG,
        "git_revision": revision,
        "code_sha256": code_hashes,
        "input_snapshots": input_hashes,
        "run_metadata": run_metadata,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "warnings": warnings,
        "limits": [
            "Synthetic-known-truth demonstration only; not real-bank validation.",
            "A 100-account, 300-simulation smoke run is not adequate for 99.9% tail accuracy.",
            "Public-source raw snapshots are not represented by this synthetic package.",
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUT / "warnings.md").write_text(
        "# Sample run warnings\n\n" + "\n".join(f"- {item}" for item in warnings)
        + "\n\nSee `manifest.json` for input/code hashes and runtime details.\n",
        encoding="utf-8",
    )
    print(json.dumps({"package": str(OUT), "manifest": str(OUT / "manifest.json"), "scenarios": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
