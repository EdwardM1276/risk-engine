"""Deterministic run identity based on inputs and governed versions."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict

import numpy as np

DIGEST_FIELDS = (
    "scenario", "total_exposure", "severity_multiplier", "seed", "institution_size",
    "n_accounts", "n_mc_sims", "copula_type", "t_df", "data_source",
    "idiosyncratic_shocks", "as_of_date", "engine_params_version",
    "reference_data_versions", "nca_in_duplum_enabled", "allow_synthetic_fallback",
    "portfolio_path", "strict_data_validation",
)


def _default(value: Any):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not digest-serialisable: {type(value)!r}")


def canonical_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_default).encode("utf-8")


def config_digest(snapshot: Dict[str, Any]) -> str:
    missing = [field for field in DIGEST_FIELDS if field not in snapshot]
    if missing:
        raise ValueError(f"Snapshot missing digest fields: {missing}")
    return hashlib.sha256(canonical_json({field: snapshot[field] for field in DIGEST_FIELDS})).hexdigest()