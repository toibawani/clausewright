"""Compare dev retrieval runs across commits and print bootstrap confidence intervals.

Reads ``results/<config>-<short-hash>.json`` files produced by
``scripts/run_dev_eval.py``.  For every commit/config it prints R@1, R@5, R@10,
MRR, character-level precision at a 1500-token retrieved budget, mean and
median chunk size in words, number of chunks, queries, and contracts.

Character-level precision is retrieved gold characters divided by retrieved
characters: chunks are taken in ranked order until the next chunk would exceed
a 1500-token budget (2000 words at the project's 0.75 tokens/word estimate);
the numerator is the number of characters in gold spans that are contained in
at least one of those retrieved chunks, capped at the retrieved character count.

Paired bootstrap 95% intervals resample contracts with replacement (10000
resamples, fixed seed 20261018) and report the difference in the metric
between two configs over the resampled queries.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

CONFIG_ORDER = ("bm25", "dense", "hybrid", "hybrid_rerank", "fixed_window")
METRICS_FOR_CI = ("recall_at_5", "recall_at_10", "mrr", "char_precision_at_1500_tokens")
SEED = 20261018
N_RESAMPLES = 10000


def load_runs(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {commit: {config: record}}."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(results_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(record["commit"], {})[record["config"]] = record
    return grouped


def paired_bootstrap_ci(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """95% CI for mean(left metric) - mean(right metric), resampling contracts."""
    by_contract: dict[str, tuple[list[float], list[float]]] = {}
    contracts = sorted({q["contract_title"] for q in left})
    left_by = {q["item_id"]: q for q in left}
    right_by = {q["item_id"]: q for q in right}
    for contract in contracts:
        ls = [left_by[q["item_id"]][metric] for q in left if q["contract_title"] == contract]
        rs = [right_by[q["item_id"]][metric] for q in right if q["contract_title"] == contract]
        by_contract[contract] = (ls, rs)

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        sampled = [rng.choice(contracts) for _ in range(len(contracts))]
        lvals: list[float] = []
        rvals: list[float] = []
        for contract in sampled:
            lvals.extend(by_contract[contract][0])
            rvals.extend(by_contract[contract][1])
        diffs.append(sum(lvals) / len(lvals) - sum(rvals) / len(rvals))

    diffs.sort()
    lo_idx = int(0.025 * (len(diffs) - 1))
    hi_idx = int(0.975 * (len(diffs) - 1))
    point = sum(diffs) / len(diffs)
    return point, diffs[lo_idx], diffs[hi_idx]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    grouped = load_runs(Path(args.results_dir))
    if not grouped:
        raise SystemExit(f"no result JSON files found in {args.results_dir}")

    for commit in sorted(grouped):
        records = grouped[commit]
        print(f"\n=== commit {commit} ===")
        print(
            f"{'config':<16} {'R@1':>6} {'R@5':>6} {'R@10':>7} {'MRR':>7} "
            f"{'charP':>7} {'meanW':>7} {'medW':>7} {'chunks':>7} {'queries':>7} {'contracts':>9}"
        )
        print("-" * 95)
        sample = next(iter(records.values()))
        for cfg in CONFIG_ORDER:
            if cfg not in records:
                continue
            r = records[cfg]
            m = r["metrics"]
            cs = r["chunk_stats"]
            print(
                f"{cfg:<16} "
                f"{m['recall_at_1']:>6.4f} "
                f"{m['recall_at_5']:>6.4f} "
                f"{m['recall_at_10']:>7.4f} "
                f"{m['mrr']:>7.4f} "
                f"{m['char_precision_at_1500_tokens']:>7.4f} "
                f"{cs['mean_chunk_words']:>7.1f} "
                f"{cs['median_chunk_words']:>7.1f} "
                f"{cs['n_chunks']:>7} "
                f"{r['n_queries']:>7} "
                f"{r['n_contracts']:>9}"
            )
        print(f"python={sample['python_version']} seed={sample['seed']} split={sample['split']}")

        if "hybrid" in records and "fixed_window" in records:
            print("  bootstrap (hybrid structure-aware - fixed_window):")
            for metric in METRICS_FOR_CI:
                point, lo, hi = paired_bootstrap_ci(
                    records["hybrid"]["per_query"],
                    records["fixed_window"]["per_query"],
                    metric,
                )
                excludes = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
                print(f"    {metric}: {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {excludes}")

    # fixed (current main) vs buggy structure-aware
    if "3aa195f" in grouped and "395f0d1" in grouped:
        print("\n=== fixed (3aa195f) vs buggy (395f0d1) structure-aware (hybrid) ===")
        for metric in METRICS_FOR_CI:
            point, lo, hi = paired_bootstrap_ci(
                grouped["3aa195f"]["hybrid"]["per_query"],
                grouped["395f0d1"]["hybrid"]["per_query"],
                metric,
            )
            excludes = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
            print(f"  {metric}: {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {excludes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
