"""Render every generated table in the docs from the committed result store.

No number in `docs/` or `README.md` is typed. A block between

    <!-- BEGIN GENERATED: factorial_metrics -->
    ...
    <!-- END GENERATED: factorial_metrics -->

is replaced with text built from `benchmarks/results/`. `--check` exits 1 when
the committed text differs, which is what the docs-consistency test runs.

Usage
-----
    python scripts/render_docs_tables.py --write
    python scripts/render_docs_tables.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "benchmarks" / "results"
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]

BEGIN = "<!-- BEGIN GENERATED: {name} -->"
END = "<!-- END GENERATED: {name} -->"
BLOCK = re.compile(
    r"(?P<begin><!-- BEGIN GENERATED: (?P<name>[a-z0-9_]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)

#: Experiment whose aggregate feeds every block below.
EXPERIMENT = "factorial-clean-3ae9c47"

#: Public metric keys and the column labels used in the docs.
METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("recall_at_1", "R@1"),
    ("recall_at_5", "R@5"),
    ("recall_at_10", "R@10"),
    ("mrr", "MRR"),
    ("recall_at_1500_words", "R@1500w"),
    ("recall_at_3000_words", "R@3000w"),
)

CHUNK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("n_chunks", "Chunks"),
    ("mean_words", "Mean words"),
    ("median_words", "Median"),
    ("p90_words", "p90"),
    ("max_words", "Max"),
)

#: Comparisons reported in the docs, as (metric key, label).
COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("recall_at_1", "R@1"),
    ("recall_at_5", "R@5"),
    ("recall_at_1500w", "R@1500w"),
)


def load_experiment(experiment_id: str = EXPERIMENT) -> dict[str, Any]:
    """Load one aggregate file through the store index."""
    index = json.loads((STORE / "index.json").read_text())
    entry = index["experiments"][experiment_id]
    data: Any = json.loads((STORE / entry["file"]).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{entry['file']} is not a JSON object")
    return data


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _header(labels: list[str]) -> list[str]:
    return [_row(labels), "|" + "|".join(["---"] * len(labels)) + "|"]


def render_factorial_metrics(record: dict[str, Any]) -> str:
    """Metrics per chunking x retriever, with the run's provenance."""
    dataset = record["dataset"]
    lines = [
        f"Run `{record['commit']}`, split `{dataset['name']}`, "
        f"{dataset['n_queries']} answerable queries on {dataset['n_contracts']} contracts, "
        f"dataset hash `{dataset['hash']}`. Generated from "
        f"`benchmarks/results/{record['experiment_id']}.json`.",
        "",
    ]
    labels = ["Config", *[label for _, label in METRIC_COLUMNS]]
    lines.extend(_header(labels))
    for config in sorted(record["metrics"]):
        metrics = record["metrics"][config]
        cells = [f"`{config}`"] + [f"{metrics[key]:.4f}" for key, _ in METRIC_COLUMNS]
        lines.append(_row(cells))
    return "\n".join(lines)


def render_factorial_chunks(record: dict[str, Any]) -> str:
    """Chunk statistics per chunking."""
    labels = ["Chunking", *[label for _, label in CHUNK_COLUMNS]]
    lines = _header(labels)
    for chunking in sorted(record["chunk_stats"]):
        stats = record["chunk_stats"][chunking]
        cells = [f"`{chunking}`"] + [str(stats[key]) for key, _ in CHUNK_COLUMNS]
        lines.append(_row(cells))
    return "\n".join(lines)


def render_structure_vs_reference(record: dict[str, Any]) -> str:
    """Paired differences of the product chunker against its size-matched window."""
    reference = record["config"].get("reference_chunking", "win300")
    lines = [
        f"Structure-aware chunking against the size-matched `{reference}` window, "
        "paired by contract. A difference whose interval includes zero is reported "
        "as noise.",
        "",
    ]
    lines.extend(_header(["Retriever", "Metric", "Difference", "95% CI", "Verdict"]))
    labels = dict(COMPARISON_METRICS)
    for comparison in record["comparisons"]:
        metric = comparison["metric"]
        if metric not in labels:
            continue
        sides = {comparison["left"], comparison["right"]}
        struct_side = next((side for side in sides if side.startswith("struct:")), None)
        reference_side = next((side for side in sides if side.startswith(f"{reference}:")), None)
        if struct_side is None or reference_side is None:
            continue
        retriever = struct_side.split(":", 1)[1]
        if reference_side.split(":", 1)[1] != retriever:
            continue
        verdict = "structure differs" if comparison["excludes_zero"] else "noise"
        point = comparison["point"]
        sign = "+" if point >= 0 else "-"
        low, high = sorted((comparison["ci_low"], comparison["ci_high"]))
        cells = [
            f"`{retriever}`",
            labels[metric],
            f"{sign}{abs(point):.4f}",
            f"[{low:+.4f}, {high:+.4f}]",
            verdict,
        ]
        lines.append(_row(cells))
    return "\n".join(lines)


RENDERERS = {
    "factorial_metrics": render_factorial_metrics,
    "factorial_chunks": render_factorial_chunks,
    "factorial_struct_vs_reference": render_structure_vs_reference,
}


def rendered_blocks() -> dict[str, str]:
    """Every block available for insertion, keyed by marker name."""
    record = load_experiment()
    return {name: render(record) for name, render in RENDERERS.items()}


def apply_blocks(text: str, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """Replace marked blocks in ``text``; return the new text and unknown names."""
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in blocks:
            unknown.append(name)
            return match.group(0)
        return f"{BEGIN.format(name=name)}\n{blocks[name]}\n{END.format(name=name)}"

    return BLOCK.sub(replace, text), unknown


def check_docs() -> list[str]:
    """Return one message per file whose generated text is out of date."""
    blocks = rendered_blocks()
    stale: list[str] = []
    for path in DOCS:
        text = path.read_text()
        updated, unknown = apply_blocks(text, blocks)
        if unknown:
            stale.extend(f"{path.name}: unknown block {name}" for name in sorted(set(unknown)))
        if updated != text:
            stale.append(f"{path.name}: generated block out of date")
    return stale


def write_docs() -> None:
    blocks = rendered_blocks()
    for path in DOCS:
        text = path.read_text()
        updated, unknown = apply_blocks(text, blocks)
        if unknown:
            raise SystemExit(f"{path}: unknown block(s) {sorted(set(unknown))}")
        if updated != text:
            path.write_text(updated)
            print(f"updated {path.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    if args.write:
        write_docs()
        return 0

    stale = check_docs()
    for message in stale:
        print(message, file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
