import json
import unittest
from pathlib import Path

from engine.reproducibility import config_digest
from run import ReproductionError, reproduce_run, run_engine_end_to_end


class ReproducibilityTests(unittest.TestCase):
    def test_material_inputs_change_digest(self):
        base = run_engine_end_to_end(n_accounts=8, n_mc_sims=8, seed=11, as_of_date="2026-09-16")
        changed = run_engine_end_to_end(
            n_accounts=8, n_mc_sims=8, seed=11, total_exposure=400_000_000_000,
            as_of_date="2026-09-16",
        )
        self.assertNotEqual(
            base["run_metadata"]["config_digest"],
            changed["run_metadata"]["config_digest"],
        )

    def test_invalid_requests_fail_before_execution(self):
        with self.assertRaises(ValueError):
            run_engine_end_to_end(n_accounts=0)
        with self.assertRaises(ValueError):
            run_engine_end_to_end(copula_type="t", t_df=2)

    def test_single_digest_field_changes_identity(self):
        snapshot = {
            "scenario": "Base", "total_exposure": 500_000_000_000.0,
            "severity_multiplier": 1.0, "seed": 1,
            "institution_size": "Large_D-SIB", "n_accounts": 10, "n_mc_sims": 10,
            "copula_type": "t", "t_df": 6, "data_source": "synthetic",
            "idiosyncratic_shocks": {}, "as_of_date": "2026-09-16",
            "engine_params_version": "dev", "reference_data_versions": {"SA_POLICY": 2},
            "nca_in_duplum_enabled": False, "allow_synthetic_fallback": False,
            "portfolio_path": None, "strict_data_validation": False,
        }
        first = config_digest(snapshot)
        snapshot["seed"] = 2
        self.assertNotEqual(first, config_digest(snapshot))

    def test_identical_runs_reproduce_and_sequence(self):
        kwargs = {
            "n_accounts": 12, "n_mc_sims": 12, "seed": 55,
            "as_of_date": "2026-09-16",
        }
        first = run_engine_end_to_end(**kwargs)
        second = run_engine_end_to_end(**kwargs)
        self.assertEqual(first["run_metadata"]["config_digest"], second["run_metadata"]["config_digest"])
        self.assertNotEqual(first["run_metadata"]["run_id"], second["run_metadata"]["run_id"])
        reproduced = reproduce_run(first["run_metadata"]["run_id"])
        self.assertEqual(reproduced["run_metadata"]["config_digest"], first["run_metadata"]["config_digest"])
        self.assertAlmostEqual(reproduced["ifrs9"]["ecl_total"], first["ifrs9"]["ecl_total"], places=2)
        self.assertAlmostEqual(reproduced["regcap"]["total_rwa"], first["regcap"]["total_rwa"], places=2)

        snapshot_path = Path(first["run_metadata"]["snapshot_path"])
        original = snapshot_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(original)
            payload["seed"] = int(payload["seed"]) + 1
            snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ReproductionError):
                reproduce_run(first["run_metadata"]["run_id"])
        finally:
            snapshot_path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
