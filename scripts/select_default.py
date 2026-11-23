"""Apply the amended default-selection rule to a factorial report.

Amended rule (2026-11-22)
-------------------------
The first version used paired R@5. That metric mechanically rewards bigger
windows: five 500-word chunks put far more text in front of the reader than
five 200-word chunks, so a large-window configuration wins on R@5 without
retrieving the right clause any better. The primary metric is therefore:

* primary: recall at an equal retrieved budget of 1500 words;
* secondary: recall at 3000 words;
* once span localization exists: localized-span hit rate and character
  precision at the same budget.

Mechanics are unchanged: pick the smallest chunk-size configuration whose
paired difference against the best configuration has a 95 percent confidence
interval that includes zero for ``hybrid_rerank``; if a structure-respecting
variant meets that rule, prefer it.

This selection is labelled exploratory: the rule was amended after results had
been seen, so the choice needs confirming on a source that has not been used
for tuning (LegalBench-RAG, or a contract-disjoint split).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clausewright.eval.stats import paired_bootstrap

#: Retriever the rule is defined on.
RETRIEVER = "hybrid_rerank"
#: Primary selection metric: recall with a fixed 1500-word context budget.
PRIMARY_METRIC = "recall_at_1500w"
#: Secondary metric reported next to the primary one.
SECONDARY_METRIC = "recall_at_3000w"

#: Configured word budget per chunking name.
CHUNK_WORDS: dict[str, int] = {
    "struct": 300,
    "struct_packed_300": 300,
    "struct_packed_400": 400,
    "win150": 150,
    "win200": 200,
    "win300": 300,
    "win400": 400,
    "win500": 500,
}

#: Chunkings that respect section boundaries.
STRUCTURE_RESPECTING: frozenset[str] = frozenset(
    {"struct", "struct_packed_300", "struct_packed_400"}
)


def _rows(report: dict[str, Any], chunking: str) -> list[dict[str, Any]]:
    return report["per_query"][f"{chunking}:{RETRIEVER}"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def evaluate(report: dict[str, Any]) -> dict[str, Any]:
    """Compute the rule's ordering, its decision, and the budget trend."""
    per_query = report["per_query"]
    available = [
        name.split(":", 1)[0]
        for name in per_query
        if name.endswith(f":{RETRIEVER}") and name.split(":", 1)[0] in CHUNK_WORDS
    ]
    if not available:
        raise ValueError(f"no {RETRIEVER} cells in this report")

    means = {
        name: _mean([float(row[PRIMARY_METRIC]) for row in _rows(report, name)])
        for name in available
    }
    best = max(available, key=lambda name: (means[name], name))

    rows: list[dict[str, Any]] = []
    for name in available:
        keyed = {row["item_id"]: row for row in _rows(report, name)}
        reference = {row["item_id"]: row for row in _rows(report, best)}
        item_ids = sorted(set(keyed) & set(reference))
        diff = paired_bootstrap(
            metric=PRIMARY_METRIC,
            left_label=f"{name}:{RETRIEVER}",
            right_label=f"{best}:{RETRIEVER}",
            pairs=[
                (reference[i]["contract"], keyed[i][PRIMARY_METRIC], reference[i][PRIMARY_METRIC])
                for i in item_ids
            ],
            seed=int(report["config"].get("seed", 20261018)),
        )
        secondary = paired_bootstrap(
            metric=SECONDARY_METRIC,
            left_label=f"{name}:{RETRIEVER}",
            right_label=f"{best}:{RETRIEVER}",
            pairs=[
                (reference[i]["contract"], keyed[i][SECONDARY_METRIC], reference[i][SECONDARY_METRIC])
                for i in item_ids
            ],
            seed=int(report["config"].get("seed", 20261018)),
        )
        rows.append(
            {
                "chunking": name,
                "word_budget": CHUNK_WORDS[name],
                "structure_respecting": name in STRUCTURE_RESPECTING,
                "mean_recall_at_1500w": round(means[name], 4),
                "diff_1500w": round(diff.point, 4),
                "ci_low_1500w": round(diff.low, 4),
                "ci_high_1500w": round(diff.high, 4),
                "indistinguishable_at_1500w": not diff.excludes_zero,
                "diff_3000w": round(secondary.point, 4),
                "ci_low_3000w": round(secondary.low, 4),
                "ci_high_3000w": round(secondary.high, 4),
                "indistinguishable_at_3000w": not secondary.excludes_zero,
            }
        )

    eligible = [row for row in rows if row["indistinguishable_at_1500w"]]
    structure_eligible = [row for row in eligible if row["structure_respecting"]]
    pool = structure_eligible or eligible
    choice = min(pool, key=lambda row: (row["word_budget"], row["chunking"]))

    trend = [
        {
            "chunking": row["chunking"],
            "word_budget": row["word_budget"],
            "mean_recall_at_1500w": row["mean_recall_at_1500w"],
        }
        for row in sorted(rows, key=lambda r: (r["word_budget"], r["chunking"]))
    ]

    return {
        "rule_version": "amended 2026-11-22: equal-budget recall",
        "status": "exploratory",
        "retriever": RETRIEVER,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "best": best,
        "best_mean": round(means[best], 4),
        "rows": trend,
        "detail": rows,
        "chosen": choice["chunking"],
        "chosen_word_budget": choice["word_budget"],
        "pool_used": "structure-respecting" if structure_eligible else "all chunkings",
    }


def print_report(decision: dict[str, Any]) -> None:
    print(
        f"rule ({decision['rule_version']}, {decision['status']}): smallest chunking whose "
        f"{decision['primary_metric']} diff vs the best ({decision['best']}) has a CI "
        f"including zero, for {decision['retriever']}; structure-respecting variants preferred"
    )
    head = (
        f"{'chunking':<22} {'words':>6} {'struct':>7} {'R@1500w':>8} {'diff':>8} "
        f"{'CI low':>8} {'CI high':>8} {'elig':>5} {'R@3000w d':>10}"
    )
    print(head)
    print("-" * len(head))
    for row in sorted(decision["detail"], key=lambda r: (r["word_budget"], r["chunking"])):
        print(
            f"{row['chunking']:<22} {row['word_budget']:>6} "
            f"{'yes' if row['structure_respecting'] else 'no':>7} "
            f"{row['mean_recall_at_1500w']:>8.4f} {row['diff_1500w']:>+8.4f} "
            f"{row['ci_low_1500w']:>+8.4f} {row['ci_high_1500w']:>+8.4f} "
            f"{'yes' if row['indistinguishable_at_1500w'] else 'no':>5} "
            f"{row['diff_3000w']:>+10.4f}"
        )
    print(
        f"\nchosen: {decision['chosen']} at {decision['chosen_word_budget']} words "
        f"(pool: {decision['pool_used']})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="factorial JSON with --dump-per-query")
    parser.add_argument("--out", default="data/eval/default_selection_amended.json")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if "per_query" not in report:
        raise SystemExit("this report has no per_query section; rerun with --dump-per-query")

    decision = evaluate(report)
    decision["source_report"] = str(args.report)
    decision["source_commit"] = report.get("provenance", {}).get("commit")
    decision["source_clean_tree"] = report.get("provenance", {}).get("clean_tree")
    print_report(decision)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
