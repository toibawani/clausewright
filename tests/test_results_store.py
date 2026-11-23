"""Tests for the committed aggregate result store.

The store is the only place a number may come from, so its rules are enforced
by a test: a dirty-tree result, a query string, a contract quote, or a long
prose field cannot be committed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from clausewright.eval.result_store import (
    BANNED_KEYS,
    MAX_STRING_LENGTH,
    check_store,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "benchmarks" / "results"


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "experiment_id": "demo",
        "commit": "abc1234",
        "provenance": {"commit": "abc1234", "clean_tree": True, "python": "3.13.4"},
        "dataset": {"hash": "deadbeef", "n_contracts": 20, "n_queries": 180},
        "metrics": {"struct:bm25": {"recall_at_1": 0.5067, "mrr": 0.6748}},
        "comparisons": [
            {
                "metric": "recall_at_1",
                "left": "struct:bm25",
                "right": "win300:bm25",
                "n_contracts": 20,
                "point": -0.1147,
                "ci_low": -0.1693,
                "ci_high": -0.0603,
                "excludes_zero": True,
            }
        ],
    }
    record.update(overrides)
    return record


def test_the_committed_store_passes() -> None:
    assert check_store(STORE) == []


def test_valid_record_passes() -> None:
    assert validate_result(_record()) == []


def test_missing_provenance_is_rejected() -> None:
    record = _record()
    del record["provenance"]

    assert "missing provenance block" in validate_result(record)


def test_dirty_tree_is_rejected() -> None:
    record = _record(provenance={"commit": "abc1234", "clean_tree": False})

    assert any("clean_tree" in problem for problem in validate_result(record))


def test_banned_key_is_rejected_at_any_depth() -> None:
    record = _record(extra={"results": [{"query": "what is the notice period"}]})
    problems = validate_result(record)

    assert any("banned key" in problem for problem in problems)


@pytest.mark.parametrize("key", sorted(BANNED_KEYS))
def test_each_banned_key_is_rejected(key: str) -> None:
    problems = validate_result(_record(extra={key: "x"}))

    assert any("banned key" in problem for problem in problems)


def test_long_string_is_rejected() -> None:
    record = _record(note="x" * (MAX_STRING_LENGTH + 1))

    assert any("string longer than" in problem for problem in validate_result(record))


def test_string_at_the_limit_is_allowed() -> None:
    assert validate_result(_record(note="x" * MAX_STRING_LENGTH)) == []


def test_unindexed_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text(json.dumps({"experiments": {}}))
    (tmp_path / "stray.json").write_text(json.dumps(_record(experiment_id="stray")))

    assert any("not listed in index.json" in problem for problem in check_store(tmp_path))


def test_index_pointing_at_a_missing_file_is_reported(tmp_path: Path) -> None:
    index = {"experiments": {"ghost": {"file": "ghost.json"}}}
    (tmp_path / "index.json").write_text(json.dumps(index))

    assert any("file is missing" in problem for problem in check_store(tmp_path))


def _converter() -> Any:
    path = ROOT / "scripts" / "to_aggregate.py"
    spec = importlib.util.spec_from_file_location("to_aggregate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_drops_per_query_and_keeps_metrics() -> None:
    raw = {
        "provenance": {"commit": "abc\nabc", "commit_short": "abc", "clean_tree": True},
        "config": {"split": "dev"},
        "dataset_hash": "deadbeef",
        "n_contracts": 20,
        "n_queries": 180,
        "metrics": {"struct:bm25": {"recall_at_1": 0.5067}},
        "per_query": {"struct:bm25": [{"text": "contract prose"}]},
    }

    aggregate = _converter().factorial_to_aggregate(raw, experiment_id="x", protocol="E-008")

    assert "per_query" not in aggregate
    assert aggregate["provenance"]["commit"] == "abc"
    assert aggregate["metrics"]["struct:bm25"]["recall_at_1"] == 0.5067
    assert validate_result(aggregate) == []
