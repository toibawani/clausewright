"""Tests for the declared default-selection rule.

Each case is a hand-built report: the rule's job is mechanical application, so
the expected answer is computed by reading the fixture, not by running the
model.
"""

from __future__ import annotations

import pytest

from scripts.select_default import evaluate


def _report(scores: dict[str, list[float]], n_contracts: int = 4) -> dict[str, object]:
    """Build a minimal factorial report with per-query scores per chunking."""
    per_query: dict[str, list[dict[str, object]]] = {}
    for chunking, values in scores.items():
        rows = []
        for index, value in enumerate(values):
            rows.append(
                {
                    "item_id": f"item{index}",
                    "contract": f"contract{index % n_contracts}",
                    "recall_at_1": value,
                    "recall_at_5": value,
                    "recall_at_10": value,
                    "mrr": value,
                    "recall_at_1500w": value,
                }
            )
        per_query[f"{chunking}:hybrid_rerank"] = rows
    return {"config": {"seed": 7}, "per_query": per_query}


def test_ties_prefer_the_structure_respecting_variant() -> None:
    report = _report(
        {
            "win150": [0.8, 0.8, 0.8, 0.8],
            "win400": [0.8, 0.8, 0.8, 0.8],
            "struct_packed_300": [0.8, 0.8, 0.8, 0.8],
        }
    )

    decision = evaluate(report)  # type: ignore[arg-type]

    # All three are indistinguishable from the best; the rule prefers the
    # structure-respecting variant even though win150 is smaller.
    assert decision["chosen"] == "struct_packed_300"
    assert decision["pool_used"] == "structure-respecting"


def test_smallest_eligible_window_wins_when_no_structure_variant_qualifies() -> None:
    report = _report(
        {
            "struct": [0.2, 0.2, 0.2, 0.2],
            "win300": [0.8, 0.8, 0.8, 0.8],
            "win400": [0.8, 0.8, 0.8, 0.8],
        }
    )

    decision = evaluate(report)  # type: ignore[arg-type]

    assert decision["chosen"] == "win300"
    assert decision["pool_used"] == "all chunkings"


def test_clearly_worse_configuration_is_not_eligible() -> None:
    report = _report(
        {
            "win150": [0.1, 0.1, 0.1, 0.1],
            "win300": [0.9, 0.9, 0.9, 0.9],
        }
    )

    decision = evaluate(report)  # type: ignore[arg-type]

    rows = {row["chunking"]: row for row in decision["rows"]}
    assert rows["win150"]["indistinguishable_from_best"] is False
    assert decision["chosen"] == "win300"


def test_missing_per_query_is_an_error() -> None:
    with pytest.raises(ValueError):
        evaluate({"config": {}, "per_query": {}})  # type: ignore[arg-type]
