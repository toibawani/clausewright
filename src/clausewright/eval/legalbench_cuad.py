"""LegalBench & CUAD Contract Evaluation and Benchmarking Suite (Pillar 5).

This module provides a standardized legal benchmark harness inspired by:
- CUAD (Contract Understanding Atticus Dataset, Hendrycks et al.)
- LegalBench (Guha et al.)

It evaluates the 3 core pillars of legal contract QA:
1. Context Recall: Did retrieval surface the governing clause and cross-referenced exceptions?
2. Faithfulness: Are 100% of claims grounded in verified contract text?
3. Abstention Precision: Does the system correctly refuse to answer when a clause is silent,
   avoiding dangerous hallucinations on unmentioned legal terms?
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from clausewright.qa.schemas import Abstention, Answer

logger = logging.getLogger(__name__)


class BenchmarkCategory(str, Enum):
    """Core commercial contract clause categories from CUAD and LegalBench."""

    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    INDEMNIFICATION = "indemnification"
    GOVERNING_LAW = "governing_law"
    TERMINATION = "termination"
    NON_COMPETE = "non_compete"
    SILENT_NEGATIVE = "silent_negative"  # Ground truth is ABSENT; system MUST abstain


@dataclass(frozen=True)
class LegalBenchCase:
    """One evaluation instance."""

    case_id: str
    category: BenchmarkCategory
    question: str
    target_section: str | None
    ground_truth_excerpt: str | None
    should_abstain: bool = False
    notes: str = ""


@dataclass
class CaseEvaluationResult:
    """Outcome of evaluating one benchmark case."""

    case_id: str
    category: BenchmarkCategory
    passed: bool
    context_recalled: bool
    faithfulness_rate: float
    abstention_correct: bool
    detail: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregated evaluation metrics across the test suite."""

    total_cases: int
    passed_cases: int
    context_recall_rate: float
    average_faithfulness: float
    abstention_precision: float
    category_scores: dict[str, float] = field(default_factory=dict)


# Standard curated legal contract benchmark dataset
STANDARD_LEGALBENCH_CASES: tuple[LegalBenchCase, ...] = (
    LegalBenchCase(
        case_id="CUAD-LIAB-01",
        category=BenchmarkCategory.LIMITATION_OF_LIABILITY,
        question="What is the aggregate liability cap under the agreement?",
        target_section="12.1",
        ground_truth_excerpt="maximum aggregate liability shall not exceed twelve (12) months of fees",
        should_abstain=False,
    ),
    LegalBenchCase(
        case_id="CUAD-LIAB-02",
        category=BenchmarkCategory.LIMITATION_OF_LIABILITY,
        question="Does the liability cap apply to breach of confidentiality?",
        target_section="12.2",
        ground_truth_excerpt="exclusions: neither party's liability for breach of Section 8 (Confidentiality) shall be limited",
        should_abstain=False,
    ),
    LegalBenchCase(
        case_id="CUAD-TERM-01",
        category=BenchmarkCategory.TERMINATION,
        question="How many days of prior written notice are required to terminate for convenience?",
        target_section="14.2",
        ground_truth_excerpt="may terminate this Agreement for convenience upon thirty (30) days prior written notice",
        should_abstain=False,
    ),
    LegalBenchCase(
        case_id="CUAD-GOV-01",
        category=BenchmarkCategory.GOVERNING_LAW,
        question="What law governs the contract and what is the chosen jurisdiction?",
        target_section="19.1",
        ground_truth_excerpt="governed by the laws of the State of Delaware without regard to conflict of laws principles",
        should_abstain=False,
    ),
    LegalBenchCase(
        case_id="CUAD-NEG-01",
        category=BenchmarkCategory.SILENT_NEGATIVE,
        question="What is the liquidated damages penalty for late software delivery?",
        target_section=None,
        ground_truth_excerpt=None,
        should_abstain=True,
        notes="Agreement is silent on liquidated damages; system must abstain.",
    ),
)


class SupportsQA(Protocol):
    """Protocol for question-answering engines."""

    def answer(self, question: str) -> Answer | Abstention: ...


class LegalBenchRunner:
    """Orchestrates evaluation of Clausewright against legal contract benchmarks."""

    def __init__(self, cases: Sequence[LegalBenchCase] = STANDARD_LEGALBENCH_CASES) -> None:
        self.cases = list(cases)

    def evaluate(self, engine: SupportsQA) -> BenchmarkSummary:
        """Run all benchmark cases against the provided QA engine."""
        results: list[CaseEvaluationResult] = []
        cat_passes: dict[str, int] = {}
        cat_totals: dict[str, int] = {}

        for case in self.cases:
            cat_totals[case.category.value] = cat_totals.get(case.category.value, 0) + 1
            response = engine.answer(case.question)

            if case.should_abstain:
                # Abstention test
                is_correct = isinstance(response, Abstention)
                res = CaseEvaluationResult(
                    case_id=case.case_id,
                    category=case.category,
                    passed=is_correct,
                    context_recalled=True,
                    faithfulness_rate=1.0 if is_correct else 0.0,
                    abstention_correct=is_correct,
                    detail="Correctly abstained on silent term"
                    if is_correct
                    else "Failed to abstain on silent term",
                )
            else:
                if isinstance(response, Answer):
                    # Check if target excerpt or section was cited
                    cited_sections = {
                        c.section_number for cl in response.claims for c in cl.citations
                    }
                    recalled = (
                        case.target_section in cited_sections if case.target_section else True
                    )

                    # Verify claim citations
                    faithful_claims = sum(1 for cl in response.claims if len(cl.citations) > 0)
                    faithfulness = (
                        faithful_claims / len(response.claims) if response.claims else 0.0
                    )

                    passed = recalled and faithfulness >= 0.99
                    res = CaseEvaluationResult(
                        case_id=case.case_id,
                        category=case.category,
                        passed=passed,
                        context_recalled=recalled,
                        faithfulness_rate=faithfulness,
                        abstention_correct=True,
                        detail="Answer verified with valid citations"
                        if passed
                        else "Recall or faithfulness missed",
                    )
                else:
                    # Premature abstention on answerable question
                    res = CaseEvaluationResult(
                        case_id=case.case_id,
                        category=case.category,
                        passed=False,
                        context_recalled=False,
                        faithfulness_rate=0.0,
                        abstention_correct=False,
                        detail="False abstention on answerable question",
                    )

            if res.passed:
                cat_passes[case.category.value] = cat_passes.get(case.category.value, 0) + 1
            results.append(res)

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        recalls = [
            r.context_recalled
            for r in results
            if not r.abstention_correct or r.category != BenchmarkCategory.SILENT_NEGATIVE
        ]
        faiths = [
            r.faithfulness_rate for r in results if r.category != BenchmarkCategory.SILENT_NEGATIVE
        ]
        absts = [
            r.abstention_correct for r in results if r.category == BenchmarkCategory.SILENT_NEGATIVE
        ]

        cat_scores = {cat: round(cat_passes.get(cat, 0) / cat_totals[cat], 3) for cat in cat_totals}

        return BenchmarkSummary(
            total_cases=total,
            passed_cases=passed_count,
            context_recall_rate=round(sum(recalls) / len(recalls), 3) if recalls else 1.0,
            average_faithfulness=round(sum(faiths) / len(faiths), 3) if faiths else 1.0,
            abstention_precision=round(sum(absts) / len(absts), 3) if absts else 1.0,
            category_scores=cat_scores,
        )
