"""Evaluation harness: datasets, metrics, runner, and calibration."""

from clausewright.eval.dataset import (
    CUAD_SMALL,
    SUITES,
    EvalItem,
    EvalSet,
    SuiteConfig,
    build_eval_set,
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
    "SPAN_OVERLAP_THRESHOLD",
    "SUITES",
    "EvalItem",
    "EvalSet",
    "SuiteConfig",
    "abstention_scores",
    "build_eval_set",
    "first_hit_rank",
    "mean",
    "mrr_score",
    "overlap_ratio",
    "percentile",
    "recall_at_k",
    "span_hit",
]
