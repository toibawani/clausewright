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
import sys
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
# held-out split helpers
# ---------------------------------------------------------------------------


def _heldout_eval_set(split: str, finalist: str | None) -> Any:
    """The frozen held-out split, or None for dev/test.

    Reading the held-out split is refused unless this is a declared final run
    against the frozen finalist file; the guard raises before a model loads.
    """
    if split != "heldout":
        return None
    from clausewright.eval import heldout

    heldout.require_final_run(finalist_path=finalist)
    return heldout.build_heldout_eval_set()


def _log_heldout_look(result_path: Path, *, configuration: str) -> None:
    from clausewright.eval import heldout

    heldout.record_look(
        configuration=configuration,
        result_path=result_path,
        note=heldout.describe_command(sys.argv),
    )


# ---------------------------------------------------------------------------
# retrieval subcommand
# ---------------------------------------------------------------------------


def _cmd_retrieval(args: argparse.Namespace) -> int:
    from clausewright.eval.retrieval import (
        CONFIGS,
        baseline_regressions,
        run_and_save_retrieval_eval,
    )

    if args.config == "all":
        configs_to_run = CONFIGS
    else:
        configs_to_run = tuple(c.strip() for c in args.config.split(","))

    eval_set = _heldout_eval_set(args.split, args.finalist)
    print(f"running retrieval evaluation on split '{args.split}' for configs: {configs_to_run}")
    out_path = run_and_save_retrieval_eval(
        output_path=Path(args.out) if args.out else None,
        split=args.split,
        configs=configs_to_run,
        sample_contracts=args.sample,
        eval_set=eval_set,
    )
    print(f"saved retrieval results: {out_path}")
    if eval_set is not None:
        _log_heldout_look(out_path, configuration=f"retrieval:{','.join(configs_to_run)}")

    if args.baseline:
        baseline_path = Path(args.baseline)
        report = json.loads(out_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        drops = baseline_regressions(report, baseline, tolerance=args.tolerance)
        if drops:
            print("\nregression against baseline:", file=sys.stderr)
            for drop in drops:
                print(f"  {drop}", file=sys.stderr)
            return 1
        print(
            f"baseline check passed (tolerance {args.tolerance:.3f})"
        )
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

    eval_set = _heldout_eval_set(args.split, args.finalist)
    record = run_full_eval(
        split=args.split,
        configs=configs_to_run,
        sample_contracts=args.sample,
        max_llm_calls=args.max_llm_calls,
        replay_cache=args.replay_cache,
        output_path=args.out,
        eval_set=eval_set,
    )
    if eval_set is not None:
        _log_heldout_look(
            Path(args.out or "data/eval/retrieval_results.json"),
            configuration=f"run:{','.join(configs_to_run)}",
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
# heldout subcommand — freeze or inspect the contract-disjoint split
# ---------------------------------------------------------------------------


def _cmd_heldout(args: argparse.Namespace) -> int:
    from clausewright.eval import heldout

    if not args.freeze:
        manifest = heldout.load_manifest()
        print(json.dumps(manifest, indent=2))
        return 0

    path, file_hash = heldout.freeze(seed=args.seed, n_contracts=args.contracts)
    manifest = heldout.load_manifest(path)
    print(f"frozen held-out manifest: {path}")
    print(f"contracts: {manifest['n_contracts']} (disjoint from {manifest['excluded']['suite']})")
    print(f"file-list sha256: {file_hash}")
    return 0


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
    ret_parser.add_argument(
        "--split", type=str, default="dev", choices=["dev", "test", "heldout"]
    )
    ret_parser.add_argument(
        "--finalist",
        type=str,
        default=None,
        help="Frozen finalist file; required (and checked) for --split heldout",
    )
    ret_parser.add_argument("--sample", type=int, default=None)
    ret_parser.add_argument("--out", type=str, default=None)
    ret_parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline JSON to compare against; exit 1 on a metric drop",
    )
    ret_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Allowed drop below the baseline before failing (default 0.02)",
    )
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
        choices=["dev", "test", "heldout"],
        help=(
            "Dataset split to evaluate (default: dev; use test only for final "
            "reporting; heldout only for the declared final run)"
        ),
    )
    run_parser.add_argument(
        "--finalist",
        type=str,
        default=None,
        help="Frozen finalist file; required (and checked) for --split heldout",
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

    # heldout subcommand
    ho_parser = subparsers.add_parser(
        "heldout",
        help="Freeze (once) or inspect the contract-disjoint held-out split",
    )
    ho_parser.add_argument(
        "--freeze",
        action="store_true",
        help="write the manifest if it is not frozen yet; refuses to change one",
    )
    ho_parser.add_argument("--seed", type=int, default=20261123)
    ho_parser.add_argument("--contracts", type=int, default=30)
    ho_parser.set_defaults(func=_cmd_heldout)

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

    from clausewright.eval.heldout import HeldoutRefused

    args = parser.parse_args(argv)
    try:
        return args.func(args)  # type: ignore[no-any-return]
    except HeldoutRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
