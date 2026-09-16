"""NCA s103(5) statutory in-duplum computational gate only.

This module deliberately implements arithmetic, not legal policy. Cure
treatment, renewed default, payment allocation, court orders, and common-law
interaction must be supplied by an approved versioned policy before use in a
production decision process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict

from engine.money import round_post

ZERO = Decimal("0")

CONTROLLED_CATEGORIES = (
    "arrears_interest",
    "fees",
    "credit_insurance",
    "default_admin_charges",
    "collection_costs",
)


def _d(value) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True)
class DefaultEpisode:
    """Default-event state with principal frozen at the event date."""

    account_id: str
    default_date: date
    principal_at_default: Decimal
    legal_policy_version: str
    controlled_accrued: Dict[str, Decimal] = field(default_factory=dict)

    def __post_init__(self):
        principal = _d(self.principal_at_default)
        if principal < ZERO:
            raise ValueError("principal_at_default must be non-negative")
        unknown = set(self.controlled_accrued) - set(CONTROLLED_CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown controlled categories: {unknown}")
        if any(_d(value) < ZERO for value in self.controlled_accrued.values()):
            raise ValueError("controlled accruals must be non-negative")

    @property
    def cap_consumed(self) -> Decimal:
        return sum((_d(value) for value in self.controlled_accrued.values()), ZERO)

    @property
    def cap_remaining(self) -> Decimal:
        return max(ZERO, _d(self.principal_at_default) - self.cap_consumed)


@dataclass(frozen=True)
class CapResult:
    category: str
    proposed: Decimal
    recognised: Decimal
    suppressed: Decimal
    cap_consumed_before: Decimal
    cap_consumed_after: Decimal
    cap_remaining_after: Decimal


def apply_in_duplum_cap(
    proposed,
    principal_at_default,
    cap_consumed_before,
    category: str = "arrears_interest",
) -> CapResult:
    """Apply the controlled aggregate cap to one proposed category posting."""
    if category not in CONTROLLED_CATEGORIES:
        raise ValueError(
            f"category must be one of {CONTROLLED_CATEGORIES}, got {category!r}"
        )
    proposed = round_post(_d(proposed))
    principal_at_default = round_post(_d(principal_at_default))
    consumed_before = round_post(_d(cap_consumed_before))
    if min(proposed, principal_at_default, consumed_before) < ZERO:
        raise ValueError("All amounts must be non-negative")

    remaining = max(ZERO, principal_at_default - consumed_before)
    recognised = min(proposed, remaining)
    suppressed = proposed - recognised
    consumed_after = consumed_before + recognised
    return CapResult(
        category=category,
        proposed=proposed,
        recognised=recognised,
        suppressed=suppressed,
        cap_consumed_before=consumed_before,
        cap_consumed_after=consumed_after,
        cap_remaining_after=max(ZERO, principal_at_default - consumed_after),
    )


def accrue_with_episode(episode: DefaultEpisode, category: str, proposed) -> CapResult:
    """Cap a proposed accrual against the episode's aggregate pool."""
    return apply_in_duplum_cap(
        proposed=proposed,
        principal_at_default=episode.principal_at_default,
        cap_consumed_before=episode.cap_consumed,
        category=category,
    )