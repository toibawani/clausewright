"""Turn a raw factorial run into a committed aggregate.

The raw file stays in gitignored ``data/eval/``: it carries a per-query record
for every configuration, which is useful locally and unfit for a repository.
The aggregate keeps what a reader needs to check a table: the configuration,
the contract and query counts, the dataset hash, the chunk statistics, the
metrics, the paired comparisons, the provenance block, and the cache counters.

Usage
-----
    python scripts/to_aggregate.py \
        --input data/eval/factorial-clean-3ae9c47.json \
        --experiment-id factorial-clean-3ae9c47 \
        --protocol E-008 \
        --output benchmarks/results/factorial-clean-3ae9c47.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _single_commit(value: str) -> str:
    """Keep one commit id.

    Runs made before the provenance fix recorded the id twice, separated by a
    newline, because ``rev-parse`` received ``HEAD`` twice. The value is the
    same id; keep the first line so the store holds one id per file.
    """
    return value.split("\n")[0].strip()


def factorial_to_aggregate(
    raw: dict[str, Any],
    *,
    experiment_id: str,
    protocol: str,
) -> dict[str, Any]:
    """Convert a raw ``factorial_eval`` report into the aggregate format."""
    provenance = dict(raw.get("provenance") or {})
    if "commit" in provenance:
        provenance["commit"] = _single_commit(str(provenance["commit"]))

    commit = provenance.get("commit_short") or provenance.get("commit", "")
    return {
        "experiment_id": experiment_id,
        "kind": "factorial_retrieval",
        "protocol": protocol,
        "commit": commit,
        "provenance": provenance,
        "cache": raw.get("cache", {}),
        "dataset": {
            "name": (raw.get("config") or {}).get("split", "dev"),
            "hash": raw.get("dataset_hash"),
            "n_contracts": raw.get("n_contracts"),
            "n_queries": raw.get("n_queries"),
        },
        "config": raw.get("config", {}),
        "chunk_stats": raw.get("chunk_stats", {}),
        "metrics": raw.get("metrics", {}),
        "comparisons": raw.get("comparisons", []),
        "elapsed_seconds": raw.get("elapsed_seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--protocol", required=True, help="experiment id in docs/experiments.md")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    raw = json.loads(args.input.read_text())
    aggregate = factorial_to_aggregate(raw, experiment_id=args.experiment_id, protocol=args.protocol)

    if args.output.exists():
        print(f"refusing to overwrite {args.output}; a new run is a new file")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
