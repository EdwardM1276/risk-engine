"""Monetary arithmetic policy.

Internal simulation arrays remain float64 for performance. Posted monetary
outputs cross this boundary once and use half-even rounding to cents.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN

CENT = Decimal("0.01")


def as_decimal(value) -> Decimal:
    """Convert through text so binary float artefacts are not copied."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def round_post(value) -> Decimal:
    """Round a monetary value at the posting boundary."""
    return as_decimal(value).quantize(CENT, rounding=ROUND_HALF_EVEN)


def post_sum(amounts) -> Decimal:
    """Sum exact values and round once at the posting boundary."""
    return round_post(sum((as_decimal(amount) for amount in amounts), Decimal("0")))