"""Effective-dated, versioned reference data loaded from JSON records."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple

VALID_STATUSES = ("DRAFT", "APPROVED", "ACTIVE", "SUPERSEDED")


class RateUnavailable(Exception):
    """Raised when a rate has no valid effective record for a date."""


@dataclass(frozen=True)
class RateRecord:
    code: str
    value: Decimal
    effective_from: date
    source: str
    source_observed_at: date
    approval_ref: str
    version: int
    status: str = "ACTIVE"

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}")


class RateSeries:
    """Effective-dated series with no interpolation or stale fallback."""

    def __init__(self, code: str, records: List[RateRecord]):
        usable = [record for record in records if record.status in ("ACTIVE", "SUPERSEDED")]
        if not usable:
            raise ValueError(f"RateSeries {code}: no usable records")
        starts = [record.effective_from for record in usable]
        if len(starts) != len(set(starts)):
            raise ValueError(f"RateSeries {code}: duplicate effective_from dates")
        usable.sort(key=lambda record: record.effective_from)
        self.code = code
        self._records: Tuple[RateRecord, ...] = tuple(usable)
        self._starts = [record.effective_from for record in usable]

    def at(self, query_date: date) -> RateRecord:
        index = bisect.bisect_right(self._starts, query_date) - 1
        if index < 0:
            raise RateUnavailable(
                f"{self.code}: no rate effective at {query_date} "
                f"(earliest {self._starts[0]})"
            )
        record = self._records[index]
        if record.status == "SUPERSEDED" and index == len(self._records) - 1:
            raise RateUnavailable(f"{self.code}: benchmark unavailable after {record.effective_from}")
        return record

    def latest(self) -> RateRecord:
        return self._records[-1]


def load_rates(path=None) -> Dict[str, RateSeries]:
    path = Path(path or Path(__file__).with_name("rates.json"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        code: RateSeries(code, [
            RateRecord(
                code=code,
                value=Decimal(record["value"]),
                effective_from=date.fromisoformat(record["effective_from"]),
                source=record["source"],
                source_observed_at=date.fromisoformat(record["source_observed_at"]),
                approval_ref=record["approval_ref"],
                version=int(record["version"]),
                status=record.get("status", "ACTIVE"),
            ) for record in records
        ]) for code, records in raw.items()
    }


_RATES = None


def get_rate(code: str, as_of_date: date) -> RateRecord:
    """Return the effective rate record for ``code`` and ``as_of_date``."""
    global _RATES
    if _RATES is None:
        _RATES = load_rates()
    try:
        return _RATES[code].at(as_of_date)
    except KeyError as exc:
        raise RateUnavailable(f"Unknown rate code: {code}") from exc