"""Paired bootstrap confidence intervals for retrieval comparisons.

Why resample contracts, not queries
-----------------------------------
Queries are nested inside contracts: every query about one contract shares that
contract's PDF, its chunking, and its index. Treating those queries as
independent observations understates the interval, sometimes badly. Each
resample here draws *contracts* with replacement and carries all of a drawn
contract's queries with it, which is the paired-by-contract design the project
rules require.

The statistic is the mean paired difference ``left - right`` over queries.
Per-contract sums and counts are precomputed, so a resample is a sum over
contracts rather than a re-flattening of every query.

A difference whose interval excludes zero is the only kind this project calls
real. Everything else is noise until a larger sample says otherwise.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

DEFAULT_SEED: Final[int] = 20261018
DEFAULT_RESAMPLES: Final[int] = 10_000


@dataclass(frozen=True)
class PairedDifference:
    """Mean paired difference between two configurations, with its interval."""

    metric: str
    left_label: str
    right_label: str
    n_contracts: int
    n_pairs: int
    point: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval sits on one side of zero."""
        return self.low > 0.0 or self.high < 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "left": self.left_label,
            "right": self.right_label,
            "n_contracts": self.n_contracts,
            "n_pairs": self.n_pairs,
            "point": round(self.point, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "excludes_zero": self.excludes_zero,
        }


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank quantile over an already-sorted sequence."""
    if not ordered:
        raise ValueError("quantile is undefined over an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def paired_bootstrap(
    *,
    metric: str,
    left_label: str,
    right_label: str,
    pairs: Sequence[tuple[str, float, float]],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> PairedDifference:
    """Bootstrap the mean of ``left - right``, resampling contracts.

    Parameters
    ----------
    metric:
        Name recorded in the result, for printing.
    left_label, right_label:
        Configuration names recorded in the result.
    pairs:
        ``(contract, left_value, right_value)`` for every paired query. The
        same query must appear once on each side; pairing is by position in the
        caller's list, so both sides must come from the same query set.
    n_resamples:
        Bootstrap draws. 10000 is the project default.
    seed:
        Fixed seed; a rerun on the same inputs must give the same interval.
    alpha:
        Two-sided error rate. 0.05 gives a 95 percent interval.

    Raises
    ------
    ValueError
        If ``pairs`` is empty, ``n_resamples`` < 1, or alpha is out of range.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Per-contract sums and counts: a resample then costs O(contracts).
    sums: dict[str, list[float]] = {}
    for contract, left, right in pairs:
        acc = sums.setdefault(contract, [0.0, 0.0, 0.0])
        acc[0] += left
        acc[1] += right
        acc[2] += 1.0

    contracts = sorted(sums)
    n_contracts = len(contracts)

    point = sum(left for _, left, _ in pairs) / len(pairs) - sum(
        right for _, _, right in pairs
    ) / len(pairs)

    # A local RNG, never the global one. Mersenne Twister's stream is part of
    # Python's documented behaviour, so a fixed seed reproduces the interval.
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        left_total = 0.0
        right_total = 0.0
        count = 0.0
        for _ in range(n_contracts):
            picked = rng.choice(contracts)
            l_sum, r_sum, n = sums[picked]
            left_total += l_sum
            right_total += r_sum
            count += n
        diffs.append((left_total - right_total) / count)

    diffs.sort()
    return PairedDifference(
        metric=metric,
        left_label=left_label,
        right_label=right_label,
        n_contracts=n_contracts,
        n_pairs=len(pairs),
        point=point,
        low=_quantile(diffs, alpha / 2.0),
        high=_quantile(diffs, 1.0 - alpha / 2.0),
    )


def mean_or_nan(values: Sequence[float]) -> float:
    """Arithmetic mean, NaN when empty (explicit rather than a silent zero)."""
    return sum(values) / len(values) if values else math.nan
