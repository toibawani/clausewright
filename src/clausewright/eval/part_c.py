"""Part C readiness: the answer-eval set, and what answering it would cost.

No model is called in this module. The set is a deterministic draw from the dev
split that mixes answerable items with absent-clause items, so abstention is
measured rather than assumed. The cost estimate is built from the real system
prompt and the real user-message builder, so the call count and prompt size are
not guesses.

Token counts are approximations: ``CHARS_PER_TOKEN`` is the usual English
ratio for this family of models. The published estimate says it is an estimate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from clausewright.eval.dataset import EvalItem, EvalSet, Split
from clausewright.qa.prompts import SYSTEM_PROMPT, build_user_message
from clausewright.qa.schemas import EvidenceItem

#: Size of the answer-eval set.
ANSWER_EVAL_SIZE = 100

#: Seed for the draw, fixed so the set is reproducible.
ANSWER_EVAL_SEED = 20261123

#: Share of absent-clause items in the set.
ABSENT_SHARE = 0.4

#: Characters per token used for the estimate.
CHARS_PER_TOKEN = 4

#: Completion budget assumed per call.
COMPLETION_TOKENS_PER_CALL = 350


def build_answer_eval_set(
    eval_set: EvalSet,
    *,
    split: Split = "dev",
    n: int = ANSWER_EVAL_SIZE,
    seed: int = ANSWER_EVAL_SEED,
    absent_share: float = ABSENT_SHARE,
) -> list[EvalItem]:
    """Draw *n* items from *split*, mixing answerable and absent-clause items.

    Ordering is by ``item_id`` so the same seed always produces the same list in
    the same order, which matters when a run is resumed or re-inspected.
    """
    if not 0.0 <= absent_share <= 1.0:
        raise ValueError(f"absent_share must be between 0 and 1, got {absent_share}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    items = list(eval_set.by_split(split))
    answerable = sorted((item for item in items if item.is_answerable), key=lambda i: i.item_id)
    absent = sorted((item for item in items if item.clause_absent), key=lambda i: i.item_id)

    want_absent = min(round(n * absent_share), len(absent))
    want_answerable = min(n - want_absent, len(answerable))
    if want_answerable + want_absent < n:
        raise ValueError(
            f"split {split!r} holds {len(answerable) + len(absent)} items, "
            f"which cannot supply {n}"
        )

    rng = random.Random(seed)
    picked = rng.sample(answerable, want_answerable) + rng.sample(absent, want_absent)
    return sorted(picked, key=lambda item: item.item_id)


@dataclass(frozen=True)
class PhaseCost:
    """Provider calls and tokens one phase of the pipeline needs."""

    phase: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "calls": self.calls,
            "prompt_tokens_estimate": self.prompt_tokens,
            "completion_tokens_estimate": self.completion_tokens,
            "note": self.note,
        }


def estimate_answer_cost(
    pairs: Sequence[tuple[str, Sequence[EvidenceItem]]],
    *,
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """Estimate provider calls and tokens for answering *pairs*.

    Parameters
    ----------
    pairs:
        One ``(question, evidence)`` pair per item, exactly the evidence the
        engine would send. The prompt text is assembled with the production
        builder, so a change to the prompt changes the estimate.
    model:
        Model name recorded in the estimate.

    Returns
    -------
    dict with per-phase rows, totals, and the assumptions used.
    """
    calls = len(pairs)
    prompt_chars = sum(
        len(SYSTEM_PROMPT) + len(build_user_message(question, evidence))
        for question, evidence in pairs
    )
    prompt_tokens = math.ceil(prompt_chars / CHARS_PER_TOKEN)
    completion_tokens = calls * COMPLETION_TOKENS_PER_CALL

    phases = [
        PhaseCost(
            "retrieval",
            0,
            0,
            0,
            "local embedding and cross-encoder models; no provider calls",
        ),
        PhaseCost(
            "generation",
            calls,
            prompt_tokens,
            completion_tokens,
            "one completion per question; no retries assumed",
        ),
        PhaseCost(
            "verification",
            0,
            0,
            0,
            "deterministic quote, number, and qualifier checks; no provider calls",
        ),
    ]
    return {
        "model": model,
        "questions": calls,
        "phases": [phase.as_dict() for phase in phases],
        "total_calls": calls,
        "prompt_tokens_estimate": prompt_tokens,
        "completion_tokens_estimate": completion_tokens,
        "chars_per_token": CHARS_PER_TOKEN,
        "completion_tokens_per_call": COMPLETION_TOKENS_PER_CALL,
    }
