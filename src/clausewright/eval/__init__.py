"""Evaluation harness: datasets, metrics, runner, and calibration."""

from clausewright.eval.answer_metrics import (
    abstention_metrics,
    answer_quality_report,
    citation_quote_rate,
    citation_validity_rate,
    faithfulness_rate,
    mean_claims_per_answer,
    qualifier_guard_rate,
)
from clausewright.eval.baselines import fixed_window_chunks
from clausewright.eval.calibration import (
    CalibrationRow,
    calibrate,
    print_calibration_table,
    run_and_save_calibration,
)
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
from clausewright.eval.faithfulness import (
    QUALIFIER_WORDS,
    extract_literals,
    is_faithful,
    violations,
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
from clausewright.eval.retrieval import (
    CONFIGS,
    evaluate_retrieval,
    run_and_save_retrieval_eval,
)
from clausewright.eval.runner import (
    ReplayLLMClient,
    evaluate_answers,
    run_full_eval,
)

__all__ = [
    "CONFIGS",
    "CUAD_SMALL",
    "DEFAULT_CATEGORIES",
    "QUALIFIER_WORDS",
    "SPAN_OVERLAP_THRESHOLD",
    "SUITES",
    "CalibrationRow",
    "EvalItem",
    "EvalSet",
    "ReplayLLMClient",
    "SuiteConfig",
    "abstention_metrics",
    "abstention_scores",
    "answer_quality_report",
    "build_eval_set",
    "calibrate",
    "citation_quote_rate",
    "citation_validity_rate",
    "dataset_hash",
    "evaluate_answers",
    "evaluate_retrieval",
    "extract_literals",
    "faithfulness_rate",
    "first_hit_rank",
    "fixed_window_chunks",
    "is_faithful",
    "mean",
    "mean_claims_per_answer",
    "mrr_score",
    "overlap_ratio",
    "percentile",
    "print_calibration_table",
    "qualifier_guard_rate",
    "recall_at_k",
    "run_and_save_calibration",
    "run_and_save_retrieval_eval",
    "run_full_eval",
    "save_manifests",
    "span_hit",
    "violations",
]
