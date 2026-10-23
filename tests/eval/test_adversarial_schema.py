"""Tests for tests/eval/adversarial.yaml schema and case distributions.

Validates that:
- All 6 adversarial categories have exactly 10 seed cases (60 cases total).
- All cases have status: draft (excluded from headline numbers until approved).
- Required schema fields are present, validly typed, and non-empty.
- All IDs are unique.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ADVERSARIAL_YAML_PATH = Path(__file__).resolve().parents[1] / "eval" / "adversarial.yaml"

REQUIRED_TYPES = {
    "cross_reference",
    "amendment",
    "defined_term",
    "prompt_injection",
    "qualifier_trap",
    "split_clause",
}

REQUIRED_FIELDS = {
    "id": str,
    "type": str,
    "status": str,
    "contract_context": str,
    "question": str,
    "expected_behavior": str,
    "gold_answer": str,
    "gold_citations": list,
    "should_abstain": bool,
    "notes": str,
}


@pytest.fixture
def adversarial_data() -> dict[str, Any]:
    assert ADVERSARIAL_YAML_PATH.is_file(), f"Missing {ADVERSARIAL_YAML_PATH}"
    with ADVERSARIAL_YAML_PATH.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    assert isinstance(data, dict), "Root must be a dictionary"
    assert "adversarial_cases" in data, "Must contain 'adversarial_cases' key"
    return data


def test_adversarial_file_exists() -> None:
    assert ADVERSARIAL_YAML_PATH.is_file()


def test_adversarial_total_count(adversarial_data: dict[str, Any]) -> None:
    cases = adversarial_data["adversarial_cases"]
    assert len(cases) == 60, f"Expected 60 cases, got {len(cases)}"


def test_adversarial_ten_cases_per_type(adversarial_data: dict[str, Any]) -> None:
    cases = adversarial_data["adversarial_cases"]
    counts: dict[str, int] = {}
    for c in cases:
        t = c.get("type", "")
        counts[t] = counts.get(t, 0) + 1

    assert set(counts.keys()) == REQUIRED_TYPES
    for req_type in REQUIRED_TYPES:
        assert counts[req_type] == 10, f"Type '{req_type}' has {counts[req_type]} cases (expected 10)"


def test_adversarial_all_status_draft(adversarial_data: dict[str, Any]) -> None:
    cases = adversarial_data["adversarial_cases"]
    for c in cases:
        assert c.get("status") == "draft", f"Case {c.get('id')} has status '{c.get('status')}', must be 'draft'"


def test_adversarial_unique_ids(adversarial_data: dict[str, Any]) -> None:
    cases = adversarial_data["adversarial_cases"]
    ids = [c.get("id") for c in cases]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"


def test_adversarial_fields_and_types(adversarial_data: dict[str, Any]) -> None:
    cases = adversarial_data["adversarial_cases"]
    for c in cases:
        cid = c.get("id", "UNKNOWN")
        for field, expected_type in REQUIRED_FIELDS.items():
            assert field in c, f"Case {cid} is missing field '{field}'"
            val = c[field]
            assert isinstance(val, expected_type), f"Case {cid} field '{field}' has type {type(val)}, expected {expected_type}"
            if expected_type is str:
                assert val.strip(), f"Case {cid} field '{field}' cannot be empty string"
            elif expected_type is list:
                assert all(isinstance(elem, str) and elem.strip() for elem in val), f"Case {cid} field '{field}' must contain non-empty strings"
