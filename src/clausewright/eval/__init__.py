"""Evaluation harness: datasets, metrics, runner, and calibration."""

from clausewright.eval.baselines import fixed_window_chunks
from clausewright.eval.dataset import (
    CUAD_SMALL,
    DEFAULT_CATEGORIES,
    SUITES,
    EvalItem,
    EvalSet,
    SuiteConfig,
    build_eval_set,
    dataset_hash,
    save_manifests,
)
from clausewright.eval.metrics import (
    SPAN_OVERLAP_THRESHOLD,
    abstention_scores,
    first_hit_rank,
    mean,
    mrr_score,
    overlap_ratio,
    percentile,
    recall_at_k,
    span_hit,
)

__all__ = [
    "CUAD_SMALL",
    "DEFAULT_CATEGORIES",
    "SPAN_OVERLAP_THRESHOLD",
    "SUITES",
    "EvalItem",
    "EvalSet",
    "SuiteConfig",
    "abstention_scores",
    "build_eval_set",
    "dataset_hash",
    "first_hit_rank",
    "fixed_window_chunks",
    "mean",
    "mrr_score",
    "overlap_ratio",
    "percentile",
    "recall_at_k",
    "save_manifests",
    "span_hit",
]
