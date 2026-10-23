"""Command-line interface for the Clausewright evaluation harness.

Subcommands:
- dataset   : build deterministic dev and test manifests from CUAD annotations
- retrieval : run retrieval ablation (bm25 / dense / hybrid / rerank / window)
- run       : full eval runner — retrieval + prints table + saves JSON with
              reproducibility fields (git hash, date, split, dataset hash)
- calibrate : sweep min_rerank_score thresholds; print precision vs. coverage
              table (closes #22)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    """Current HEAD SHA for reproducibility; 'unknown' if git is not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


# ---------------------------------------------------------------------------
# dataset subcommand
# ---------------------------------------------------------------------------


def _cmd_dataset(args: argparse.Namespace) -> int:
    from clausewright.eval.dataset import (
        DEFAULT_CATEGORIES,
        SuiteConfig,
        build_eval_set,
        save_manifests,
    )

    suite = SuiteConfig(
        name=f"cuad-{args.contracts}c",
        n_contracts=args.contracts,
        categories=DEFAULT_CATEGORIES,
        seed=args.seed,
    )
    print(
        f"building eval set: {suite.n_contracts} contracts, "
        f"{len(suite.categories)} categories, seed={suite.seed}"
    )
    eval_set = build_eval_set(suite)
    dev_path, test_path = save_manifests(
        eval_set, output_dir=Path(args.out_dir) if args.out_dir else None
    )
    print(f"saved dev manifest:  {dev_path} ({len(eval_set.by_split('dev'))} items)")
    print(f"saved test manifest: {test_path} ({len(eval_set.by_split('test'))} items)")
    return 0


# ---------------------------------------------------------------------------
# retrieval subcommand
# ---------------------------------------------------------------------------


def _cmd_retrieval(args: argparse.Namespace) -> int:
    from clausewright.eval.retrieval import CONFIGS, run_and_save_retrieval_eval

    if args.config == "all":
        configs_to_run = CONFIGS
    else:
        configs_to_run = tuple(c.strip() for c in args.config.split(","))

    print(f"running retrieval evaluation on split '{args.split}' for configs: {configs_to_run}")
    out_path = run_and_save_retrieval_eval(
        output_path=Path(args.out) if args.out else None,
        split=args.split,
        configs=configs_to_run,
        sample_contracts=args.sample,
    )
    print(f"saved retrieval results: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# run subcommand — full eval runner with reproducibility fields
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Run full evaluation (retrieval + answers) and save reproducible reports."""
    from clausewright.eval.retrieval import CONFIGS
    from clausewright.eval.runner import run_full_eval

    configs_to_run: tuple[str, ...] = (
        CONFIGS if args.config == "all" else tuple(c.strip() for c in args.config.split(","))
    )

    print(f"\n{'='*60}")
    print(f"  Clausewright Full Eval Runner  |  split={args.split}")
    print(f"  git={_git_sha()}")
    print(f"{'='*60}\n")

    record = run_full_eval(
        split=args.split,
        configs=configs_to_run,
        sample_contracts=args.sample,
        max_llm_calls=args.max_llm_calls,
        replay_cache=args.replay_cache,
        output_path=args.out,
    )

    _print_retrieval_table({"configs": record.get("retrieval_metrics", {})})
    ans = record.get("answer_metrics", {})
    if ans:
        print("\nAnswer Quality Metrics:")
        print(f"  Questions evaluated:     {ans.get('n_questions')}")
        print(f"  Abstention precision:    {ans.get('abstention_precision')}")
        print(f"  Abstention recall:       {ans.get('abstention_recall')}")
        print(f"  False abstention rate:   {ans.get('false_abstention_rate')}")
        print(f"  Mean claims per answer:  {ans.get('mean_claims_per_answer')}")

    out_file = args.out or "data/eval/retrieval_results.json"
    print(f"\nSaved report to: {out_file}")
    print("Appended run to: data/eval/runs.jsonl\n")
    return 0


def _print_retrieval_table(results: dict[str, Any]) -> None:
    """Print an ASCII table of retrieval metrics to stdout."""
    configs = results.get("configs", {})
    if not configs:
        print("(no results)")
        return

    header = f"{'config':<18} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>7} {'MRR':>7} {'n_q':>5}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for cfg, scores in configs.items():
        print(
            f"{cfg:<18} "
            f"{scores['recall_at_1']:>6.3f} "
            f"{scores['recall_at_3']:>6.3f} "
            f"{scores['recall_at_5']:>6.3f} "
            f"{scores['recall_at_10']:>7.3f} "
            f"{scores['mrr']:>7.3f} "
            f"{scores['n_queries']:>5}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# calibrate subcommand — sweep min_rerank_score on dev split (closes #22)
# ---------------------------------------------------------------------------


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Sweep min_rerank_score thresholds and print precision vs. coverage table."""
    from clausewright.eval.calibration import calibrate, print_calibration_table

    print(f"\n{'='*60}")
    print(f"  Clausewright Calibration  |  split={args.split}")
    print(f"  git={_git_sha()}")
    print(f"{'='*60}\n")

    rows = calibrate(split=args.split, sample_contracts=args.sample)
    print_calibration_table(rows)

    out_path = Path(args.out) if args.out else Path("data/eval/calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "split": args.split,
        "description": (
            "Precision vs. coverage sweep over min_rerank_score thresholds. "
            "Choose the threshold that maximises abstention_recall while keeping "
            "false_abstention_rate below your firm's coverage SLA."
        ),
        "thresholds": rows,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved calibration results: {out_path}\n")
    return 0


# ---------------------------------------------------------------------------
# Parser assembly
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m clausewright.eval",
        description="Clausewright evaluation harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dataset subcommand
    ds_parser = subparsers.add_parser("dataset", help="Build CUAD eval manifests")
    ds_parser.add_argument("--seed", type=int, default=42)
    ds_parser.add_argument("--contracts", type=int, default=30)
    ds_parser.add_argument("--out-dir", type=str, default=None)
    ds_parser.set_defaults(func=_cmd_dataset)

    # retrieval subcommand
    ret_parser = subparsers.add_parser("retrieval", help="Run retrieval ablation")
    ret_parser.add_argument("--config", type=str, default="all")
    ret_parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    ret_parser.add_argument("--sample", type=int, default=None)
    ret_parser.add_argument("--out", type=str, default=None)
    ret_parser.set_defaults(func=_cmd_retrieval)

    # run subcommand — full eval with reproducibility envelope
    run_parser = subparsers.add_parser(
        "run",
        help="Full eval runner: retrieval metrics + reproducibility JSON (closes #21)",
    )
    run_parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Dataset split to evaluate (default: dev; use test only for final reporting)",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default="all",
        help="Configs to run: all, or comma-separated subset of bm25,dense,hybrid,hybrid_rerank,fixed_window",
    )
    run_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Evaluate only the first N contracts (quick smoke-test)",
    )
    run_parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=0,
        help="Max real LLM calls permitted (default: 0, uses replay cache or abstains)",
    )
    run_parser.add_argument(
        "--replay-cache",
        type=str,
        default=None,
        help="Path to record/replay cache JSON/JSONL for LLM completions",
    )
    run_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: data/eval/retrieval_results.json)",
    )
    run_parser.set_defaults(func=_cmd_run)

    # calibrate subcommand
    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Sweep min_rerank_score thresholds; print precision vs. coverage table (closes #22)",
    )
    cal_parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Split to calibrate on; always use dev, never test.",
    )
    cal_parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N contracts for a quick run.",
    )
    cal_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: data/eval/calibration.json)",
    )
    cal_parser.set_defaults(func=_cmd_calibrate)

    args = parser.parse_args(argv)
    return args.func(args)  # type: ignore[no-any-return]


if __name__ == "__main__":
    raise SystemExit(main())
