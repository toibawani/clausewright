"""Command-line interface for evaluation harness.

Subcommands:
- dataset: build deterministic dev and test manifests from CUAD annotations
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clausewright.eval.dataset import (
    DEFAULT_CATEGORIES,
    SuiteConfig,
    build_eval_set,
    save_manifests,
)


def _cmd_dataset(args: argparse.Namespace) -> int:
    suite = SuiteConfig(
        name=f"cuad-{args.contracts}c",
        n_contracts=args.contracts,
        categories=DEFAULT_CATEGORIES,
        seed=args.seed,
    )
    print(f"building eval set: {suite.n_contracts} contracts, {len(suite.categories)} categories, seed={suite.seed}")
    eval_set = build_eval_set(suite)
    dev_path, test_path = save_manifests(eval_set, output_dir=Path(args.out_dir) if args.out_dir else None)
    print(f"saved dev manifest: {dev_path} ({len(eval_set.by_split('dev'))} items)")
    print(f"saved test manifest: {test_path} ({len(eval_set.by_split('test'))} items)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m clausewright.eval", description="Clausewright evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dataset subcommand
    ds_parser = subparsers.add_parser("dataset", help="Build CUAD eval manifests")
    ds_parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic contract selection")
    ds_parser.add_argument("--contracts", type=int, default=30, help="Number of contracts to include")
    ds_parser.add_argument("--out-dir", type=str, default=None, help="Directory to save manifests")
    ds_parser.set_defaults(func=_cmd_dataset)

    args = parser.parse_args(argv)
    return args.func(args)  # type: ignore[no-any-return]


if __name__ == "__main__":
    raise SystemExit(main())
