"""Every generated table in the docs must match the committed results.

The test regenerates each block from `benchmarks/results/` and fails when the
committed text differs, so a stale or hand-edited number cannot survive a push.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _module() -> Any:
    path = SCRIPTS / "render_docs_tables.py"
    spec = importlib.util.spec_from_file_location("render_docs_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docs_match_the_committed_results() -> None:
    assert _module().check_docs() == []


def test_at_least_one_block_is_rendered_in_the_docs() -> None:
    renderer = _module()
    total = 0
    for path in renderer.DOCS:
        total += len(re.findall(r"<!-- BEGIN GENERATED: [a-z0-9_]+ -->", path.read_text()))
    assert total > 0


def test_renderer_is_deterministic() -> None:
    renderer = _module()
    first = renderer.rendered_blocks()
    second = renderer.rendered_blocks()

    assert first == second
    for body in first.values():
        assert "|" in body


def test_block_replacement_keeps_unmarked_text() -> None:
    renderer = _module()
    text = "before\n<!-- BEGIN GENERATED: factorial_metrics -->\nstale\n<!-- END GENERATED: factorial_metrics -->\nafter\n"

    updated, unknown = renderer.apply_blocks(text, renderer.rendered_blocks())

    assert unknown == []
    assert updated.startswith("before\n")
    assert updated.endswith("\nafter\n")
    assert "stale" not in updated


def test_unknown_block_is_reported() -> None:
    renderer = _module()
    text = "<!-- BEGIN GENERATED: not_a_block -->\n<!-- END GENERATED: not_a_block -->\n"

    _, unknown = renderer.apply_blocks(text, renderer.rendered_blocks())

    assert unknown == ["not_a_block"]
