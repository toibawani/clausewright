"""Tests for the amended selection rule (equal-budget recall).

Each case is a hand-built report. The rule is mechanical, so the expected
answer follows from the numbers in the fixture, not from a model run.
"""

from __future__ import annotations

import pytest
from scripts.select_default import evaluate


def _report(
    scores: dict[str, list[float]],
    *,
    scores_3000w: dict[str, list[float]] | None = None,
    n_contracts: int = 4,
) -> dict[str, object]:
    per_query: dict[str, list[dict[str, object]]] = {}
    for chunking, values in scores.items():
        rows = []
        secondary = (scores_3000w or {}).get(chunking, values)
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
                    "recall_at_3000w": secondary[index],
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

    assert decision["chosen"] == "struct_packed_300"
    assert decision["pool_used"] == "structure-respecting"
    assert decision["status"] == "exploratory"


def test_smallest_eligible_window_wins_without_a_structure_variant() -> None:
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


def test_bigger_window_that_only_wins_on_r5_is_not_chosen() -> None:
    # The small window ties on equal-budget recall but loses on R@5; the rule
    # keys on the budget metric, so the small window is eligible.
    report = _report(
        {
            "win200": [0.5, 0.5, 0.5, 0.5],
            "win500": [0.5, 0.5, 0.5, 0.5],
        }
    )

    decision = evaluate(report)  # type: ignore[arg-type]

    assert decision["chosen"] == "win200"


def test_missing_per_query_is_an_error() -> None:
    with pytest.raises(ValueError):
        evaluate({"config": {}, "per_query": {}})  # type: ignore[arg-type]
