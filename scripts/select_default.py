"""Apply the declared default-selection rule to a factorial report.

The rule, fixed before the run that decides it
----------------------------------------------
Among the chunking configurations, pick the smallest chunk-size configuration
whose paired R@5 difference versus the best configuration has a 95 percent
confidence interval that includes zero, for ``hybrid_rerank``; if a
structure-respecting variant meets that rule, prefer it.

"Smallest" is the configured word budget. ``struct`` is a 300-word windowed
configuration, so it competes at 300. When two structure-respecting variants
are eligible, the smaller budget wins; when none is eligible, the smallest
window is chosen instead.

This script does not decide anything by hand: it reads per-query scores from
the factorial JSON, recomputes the interval by contract, and prints the
ordering and the single choice the rule forces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clausewright.eval.stats import paired_bootstrap

#: Retriever the rule is defined on.
RETRIEVER = "hybrid_rerank"
#: Metric the rule is defined on.
METRIC = "recall_at_5"

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


def _scores(report: dict[str, Any], chunking: str) -> dict[str, float]:
    """item_id -> score for one cell, from the per-query dump."""
    key = f"{chunking}:{RETRIEVER}"
    rows = report["per_query"][key]
    return {row["item_id"]: float(row[METRIC]) for row in rows}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def evaluate(report: dict[str, Any]) -> dict[str, Any]:
    """Compute the rule's ordering and its single decision."""
    per_query = report["per_query"]
    available = [
        name.split(":", 1)[0]
        for name in per_query
        if name.endswith(f":{RETRIEVER}") and name.split(":", 1)[0] in CHUNK_WORDS
    ]
    if not available:
        raise ValueError(f"no {RETRIEVER} cells in this report")

    scores = {name: _scores(report, name) for name in available}
    means = {
        name: _mean(
            [
                scores[name][item_id]
                for item_id in scores[name]
            ]
        )
        for name in available
    }
    best = max(available, key=lambda name: (means[name], name))

    # Paired by item so a differing query list cannot silently misalign.
    contracts = {
        name: {row["item_id"]: row["contract"] for row in report["per_query"][f"{name}:{RETRIEVER}"]}
        for name in available
    }

    rows: list[dict[str, Any]] = []
    for name in available:
        item_ids = sorted(set(scores[name]) & set(scores[best]))
        pairs = [
            (contracts[best][item_id], scores[name][item_id], scores[best][item_id])
            for item_id in item_ids
        ]
        diff = paired_bootstrap(
            metric=METRIC,
            left_label=f"{name}:{RETRIEVER}",
            right_label=f"{best}:{RETRIEVER}",
            pairs=pairs,
            seed=int(report["config"].get("seed", 20261018)),
        )
        rows.append(
            {
                "chunking": name,
                "word_budget": CHUNK_WORDS[name],
                "structure_respecting": name in STRUCTURE_RESPECTING,
                "mean_recall_at_5": round(means[name], 4),
                "diff_vs_best": round(diff.point, 4),
                "ci_low": round(diff.low, 4),
                "ci_high": round(diff.high, 4),
                "indistinguishable_from_best": not diff.excludes_zero,
            }
        )

    eligible = [row for row in rows if row["indistinguishable_from_best"]]
    structure_eligible = [row for row in eligible if row["structure_respecting"]]
    pool = structure_eligible or eligible
    choice = min(pool, key=lambda row: (row["word_budget"], row["chunking"]))

    return {
        "retriever": RETRIEVER,
        "metric": METRIC,
        "best": best,
        "best_mean": round(means[best], 4),
        "rows": sorted(rows, key=lambda row: (row["word_budget"], row["chunking"])),
        "chosen": choice["chunking"],
        "chosen_word_budget": choice["word_budget"],
        "pool_used": "structure-respecting" if structure_eligible else "all chunkings",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="data/eval/factorial-dev-v2.json",
        help="factorial JSON written with --dump-per-query",
    )
    parser.add_argument("--out", default="data/eval/default_selection.json")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if "per_query" not in report:
        raise SystemExit(
            "this report has no per_query section; rerun the factorial with --dump-per-query"
        )

    decision = evaluate(report)
    header = (
        f"{'chunking':<22} {'words':>6} {'struct':>7} {'R@5':>7} "
        f"{'diff':>8} {'CI low':>8} {'CI high':>8}  eligible"
    )
    print(f"rule: smallest chunking whose {METRIC} diff vs the best "
          f"({decision['best']}) has a CI including zero, for {RETRIEVER}; "
          "structure-respecting variants preferred")
    print(header)
    print("-" * len(header))
    for row in decision["rows"]:
        print(
            f"{row['chunking']:<22} {row['word_budget']:>6} "
            f"{'yes' if row['structure_respecting'] else 'no':>7} "
            f"{row['mean_recall_at_5']:>7.4f} {row['diff_vs_best']:>+8.4f} "
            f"{row['ci_low']:>+8.4f} {row['ci_high']:>+8.4f}  "
            f"{'yes' if row['indistinguishable_from_best'] else 'no'}"
        )
    print(
        f"\nchosen: {decision['chosen']} at {decision['chosen_word_budget']} words "
        f"(pool: {decision['pool_used']})"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
