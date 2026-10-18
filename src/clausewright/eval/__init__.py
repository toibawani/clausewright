"""Evaluation harness: datasets, metrics, runner, and calibration."""

from clausewright.eval.dataset import (
    CUAD_SMALL,
    SUITES,
    EvalItem,
    EvalSet,
    SuiteConfig,
    build_eval_set,
)

__all__ = [
    "CUAD_SMALL",
    "SUITES",
    "EvalItem",
    "EvalSet",
    "SuiteConfig",
    "build_eval_set",
]
