"""Unit tests for LegalBench & CUAD benchmark evaluation harness (Pillar 5)."""

from __future__ import annotations

from clausewright.eval.legalbench_cuad import (
    BenchmarkCategory,
    LegalBenchCase,
    LegalBenchRunner,
)
from clausewright.qa.schemas import (
    Abstention,
    AbstentionReason,
    Answer,
    Citation,
    Claim,
)


class MockEngine:
    """Mock engine simulating realistic QA behavior on benchmark cases."""

    def answer(self, question: str) -> Answer | Abstention:
        q = question.lower()
        if "liquidated damages" in q:
            # Silent term -> correctly abstain
            return Abstention(
                question=question,
                reason=AbstentionReason.NO_EVIDENCE,
                detail="Agreement contains no mention of liquidated damages.",
            )

        citation = Citation(
            evidence_id=1,
            doc_id="doc-cuad",
            section_number="12.1" if "liability" in q else "14.2",
            page_range=(1, 1),
            score=0.95,
        )
        claim = Claim(
            text="The liability is capped at 12 months fees.",
            quote="maximum aggregate liability shall not exceed twelve (12) months of fees",
            citations=(citation,),
        )
        return Answer(
            question=question,
            text=claim.text,
            claims=(claim,),
            model="mock-legal-llm",
        )


def test_legalbench_runner_evaluation() -> None:
    cases = [
        LegalBenchCase(
            case_id="TEST-1",
            category=BenchmarkCategory.LIMITATION_OF_LIABILITY,
            question="What is the liability cap under the agreement?",
            target_section="12.1",
            ground_truth_excerpt="maximum aggregate liability",
            should_abstain=False,
        ),
        LegalBenchCase(
            case_id="TEST-2",
            category=BenchmarkCategory.SILENT_NEGATIVE,
            question="What is the liquidated damages penalty?",
            target_section=None,
            ground_truth_excerpt=None,
            should_abstain=True,
        ),
    ]

    runner = LegalBenchRunner(cases=cases)
    summary = runner.evaluate(MockEngine())

    assert summary.total_cases == 2
    assert summary.passed_cases == 2
    assert summary.context_recall_rate == 1.0
    assert summary.average_faithfulness == 1.0
    assert summary.abstention_precision == 1.0
