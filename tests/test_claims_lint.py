"""Tests for the claims lint.

The lint fails on the vocabulary of a pitch and on metric numbers typed into
the README outside a generated block. Both rules are exercised here on text
fixtures and once on the repository's own documents.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _lint() -> Any:
    path = ROOT / "scripts" / "claims_lint.py"
    spec = importlib.util.spec_from_file_location("claims_lint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_repository_documents_pass() -> None:
    assert _lint().lint_docs(ROOT) == []


@pytest.mark.parametrize("word", _lint().BANNED_WORDS)
def test_each_banned_word_is_flagged(word: str) -> None:
    hits = _lint().banned_word_hits(f"This is {word} work.\n")

    assert hits and hits[0][1] == word


def test_banned_word_is_case_insensitive() -> None:
    assert _lint().banned_word_hits("An EXCELLENT result")[0][1] == "excellent"


def test_banned_word_inside_a_longer_word_is_ignored() -> None:
    # "best" must not fire on "bestowed".
    assert _lint().banned_word_hits("the bestowed right") == []


def test_metric_number_in_prose_is_flagged() -> None:
    hits = _lint().number_hits("Recall was 0.6302 on dev.\n")

    assert hits and hits[0][1] == "0.6302"


def test_metric_number_inside_a_generated_block_is_allowed() -> None:
    text = (
        "<!-- BEGIN GENERATED: factorial_metrics -->\n"
        "| struct | 0.6302 |\n"
        "<!-- END GENERATED: factorial_metrics -->\n"
    )

    assert _lint().number_hits(text) == []


def test_decimal_percentage_in_prose_is_flagged() -> None:
    hits = _lint().number_hits("Coverage reached 87.5% of items.\n")

    assert hits and hits[0][1] == "87.5%"


def test_whole_numbers_are_not_metric_numbers() -> None:
    assert _lint().number_hits("180 answerable queries, 20 contracts.\n") == []


def test_lint_points_at_file_and_line(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Title\n\nA production-ready system.\n")
    (tmp_path / "docs" / "notes.md").write_text("No claims here.\n")

    problems = _lint().lint_docs(tmp_path)

    assert problems == ["README.md:3: banned word 'production-ready'"]


def test_readme_number_outside_a_block_is_reported(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("R@1 is 0.6302.\n")

    problems = _lint().lint_docs(tmp_path)

    assert len(problems) == 1
    assert "metric number" in problems[0]
