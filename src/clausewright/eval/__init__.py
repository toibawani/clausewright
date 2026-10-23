"""Evaluation harness: datasets, metrics, runner, and calibration."""

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

__all__ = [
    "CUAD_SMALL",
    "DEFAULT_CATEGORIES",
    "SUITES",
    "EvalItem",
    "EvalSet",
    "SuiteConfig",
    "build_eval_set",
    "dataset_hash",
    "save_manifests",
]
