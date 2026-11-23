"""Shape rules for the committed aggregate result store.

Raw eval runs live under ``data/`` and stay out of git: they embed per-query
records and paths that are not portable. What gets committed is an aggregate:
metrics, intervals, per-contract counts, and a provenance block. This module
holds the checks that decide whether a file is allowed in ``benchmarks/results``.

The rules are mechanical on purpose, so a leak is a failing test rather than a
judgement call in review:

* the file carries a provenance block whose ``clean_tree`` is true;
* no key is named ``text``, ``query``, ``quote``, or ``span``, so no record can
  smuggle a query string or a contract quote under a familiar name;
* no string value is longer than ``MAX_STRING_LENGTH`` characters, which keeps
  prose, prompts, and contract fragments out;
* the file id matches its name and is listed in the index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Keys that could carry query text, prompts, or contract quotes.
BANNED_KEYS = frozenset({"text", "query", "quote", "span"})

#: Longest string allowed in an aggregate file.
MAX_STRING_LENGTH = 200


def _walk(value: Any, path: str = "") -> list[tuple[str, Any]]:
    """Yield ``(json-path, value)`` for every node in the structure."""
    found: list[tuple[str, Any]] = [(path or "$", value)]
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk(child, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk(child, f"{path}[{index}]"))
    return found


def validate_result(record: dict[str, Any]) -> list[str]:
    """Return every rule violation in ``record``; an empty list means it passes."""
    problems: list[str] = []

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        problems.append("missing provenance block")
    elif provenance.get("clean_tree") is not True:
        problems.append("provenance.clean_tree is not true")

    for path, value in _walk(record):
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if isinstance(value, dict):
            for name in value:
                if name in BANNED_KEYS:
                    problems.append(f"banned key {path}.{name}")
        if key in BANNED_KEYS:
            problems.append(f"banned key {path}")
        if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
            problems.append(f"string longer than {MAX_STRING_LENGTH} at {path} ({len(value)} chars)")

    return problems


def load_index(root: Path) -> dict[str, Any]:
    """Load ``benchmarks/results/index.json``."""
    data: Any = json.loads((root / "index.json").read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{root / 'index.json'} is not a JSON object")
    return data


def check_store(directory: Path) -> list[str]:
    """Check every aggregate file in ``directory`` against the index."""
    problems: list[str] = []
    index = load_index(directory)
    experiments = index.get("experiments", {})

    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        experiment_id = path.stem
        if experiment_id not in experiments:
            problems.append(f"{path.name} is not listed in index.json")
            continue
        record = json.loads(path.read_text())
        expected = experiments[experiment_id]
        if record.get("experiment_id") != experiment_id:
            problems.append(f"{path.name} has experiment_id {record.get('experiment_id')!r}")
        if expected.get("file") != path.name:
            problems.append(f"index entry for {experiment_id} points at {expected.get('file')!r}")
        problems.extend(f"{path.name}: {problem}" for problem in validate_result(record))

    for experiment_id, entry in experiments.items():
        if not (directory / str(entry.get("file", ""))).is_file():
            problems.append(f"index lists {experiment_id} but its file is missing")

    return problems
