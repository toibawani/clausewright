"""Tests for the Part C answer-eval set, the cost estimate, and the dry run.

The dry run uses a stub retriever, a stub reranker, and a fake model. It checks
the plumbing (prompt assembly, verification, abstention, metrics, provenance),
not answer quality.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from clausewright.eval.dataset import CUAD_SMALL, build_eval_set
from clausewright.eval.part_c import (
    ABSENT_SHARE,
    ANSWER_EVAL_SIZE,
    build_answer_eval_set,
)

ROOT = Path(__file__).resolve().parents[2]


def _has_cuad() -> bool:
    """True when the CUAD annotations are on disk; CI fetches them separately."""
    try:
        build_eval_set(CUAD_SMALL)
    except FileNotFoundError:
        return False
    return True


requires_cuad = pytest.mark.skipif(not _has_cuad(), reason="CUAD annotations not present")


def _script(name: str) -> Any:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_cuad
def test_the_set_has_the_declared_size_and_mix() -> None:
    items = build_answer_eval_set(build_eval_set(CUAD_SMALL))

    assert len(items) == ANSWER_EVAL_SIZE
    n_absent = sum(1 for item in items if item.clause_absent)
    assert n_absent == round(ANSWER_EVAL_SIZE * ABSENT_SHARE)
    assert all(item.split == "dev" for item in items)


@requires_cuad
def test_the_draw_is_deterministic() -> None:
    eval_set = build_eval_set(CUAD_SMALL)
    first = [item.item_id for item in build_answer_eval_set(eval_set)]
    second = [item.item_id for item in build_answer_eval_set(eval_set)]

    assert first == second
    assert first == sorted(first)


@requires_cuad
def test_a_different_seed_changes_the_draw() -> None:
    eval_set = build_eval_set(CUAD_SMALL)
    other = [item.item_id for item in build_answer_eval_set(eval_set, seed=7)]

    assert other != [item.item_id for item in build_answer_eval_set(eval_set)]


def test_a_bad_size_is_rejected() -> None:
    # The size guard runs before the set is touched, so no CUAD file is needed.
    with pytest.raises(ValueError):
        build_answer_eval_set(None, n=0)  # type: ignore[arg-type]


@requires_cuad
def test_cost_counts_one_call_per_question_and_marks_the_dry_run() -> None:
    result = _script("part_c_dry_run").run(4)

    assert result["cost"]["total_calls"] == result["cost"]["questions"]
    assert result["cost"]["prompt_tokens_estimate"] > 0
    assert result["dry_run"] is True
    assert isinstance(result["provenance"]["clean_tree"], bool)


@requires_cuad
def test_the_dry_run_table_is_marked_as_a_dry_run() -> None:
    module = _script("part_c_dry_run")
    table = module.render_table(module.run(3))

    assert "Dry run" in table
    assert "Not a measurement" in table
    assert "| `n_questions` | 3 |" in table
