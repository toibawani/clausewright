# Experiments

Every retrieval or answer change is recorded here with its hypothesis, the
change, dev numbers with intervals, and the decision. Negative results stay in
the file. A change without an entry here does not exist as far as the project
is concerned.

Rules this log follows:

- Dev only. The CUAD test split is reported once per milestone and never used
  for a decision.
- Intervals are paired bootstrap 95 percent intervals resampling contracts
  (10000 resamples, seed 20261018), computed by
  `src/clausewright/eval/stats.py`. A difference whose interval includes zero
  is described as noise, not as a win.
- Numbers come from a named script or result file. Nothing here is estimated.
- Adversarial cases stay `status: draft` until the owner approves them; drafts
  never enter headline numbers.

---

## E-001 — Keep the final chunk when merging small chunks

**Date:** 2026-11-10
**Commit under test:** `3aa195f` against `9ef612c`
**Status:** merged (PR #117)

**Hypothesis.** The chunker merged chunks below the 80-word floor forward into
the next window and silently dropped the document's final chunk when it was
short. Restoring it should recover gold spans that live at the end of a
contract, and should help the structure-aware configurations only — the
fixed-window baseline never merges.

**Change.** `_merge_small_chunks` keeps the final chunk, merging it backwards
into the previous chunk instead of discarding it.

**Dev numbers** (180 answerable queries, 20 contracts), from
`results/*-9ef612c.json` and `results/*-3aa195f.json`:

| Config | Metric | 9ef612c | 3aa195f | Difference | 95% CI |
|--------|--------|---------|---------|------------|--------|
| bm25 | R@1 | 0.4357 | 0.5067 | +0.0709 | [+0.0100, +0.1423] |
| bm25 | R@5 | 0.7665 | 0.7965 | +0.0300 | [-0.0132, +0.0759] |
| bm25 | MRR | 0.6106 | 0.6748 | +0.0642 | [+0.0180, +0.1160] |
| hybrid | R@1 | 0.3304 | 0.4261 | +0.0957 | [+0.0368, +0.1634] |
| hybrid | R@5 | 0.6470 | 0.7459 | +0.0989 | [+0.0268, +0.1975] |
| hybrid | MRR | 0.4824 | 0.6040 | +0.1217 | [+0.0601, +0.1967] |
| fixed_window | R@1 | 0.6781 | 0.6781 | 0.0000 | unchanged by design |

**Decision.** Keep. R@1 and MRR gains exclude zero for both structure-aware
retrievers. BM25 R@5 does not (the interval includes zero), so the honest
statement is "R@1 and MRR improved; R@5 is inconclusive". The fixed-window
baseline is unchanged to four decimals, which is the expected control.

**What surprised us.** The fix moved structure-aware BM25 R@1 by seven points
while fixed windows did not move at all. The bug was costing more than the
section prior was earning.

---

## E-002 — Does the section prior help retrieval at matched window size?

**Date:** 2026-11-21
**Commit:** `3aa195f`
**Status:** measured; open

**Hypothesis.** Splitting on section boundaries should place a clause's text
in a chunk that also contains its heading context, improving both exact-term
and semantic matching.

**Change under test.** None. This compares the two existing chunkings:
`struct` (sections then 300-word windows at 50 percent overlap) and
`fixed_window` (one pseudo-section, 300-word windows at 50 percent overlap).

**Dev numbers** (180 answerable queries), hybrid retrieval on both sides,
paired by contract:

| Metric | struct − fixed_window | 95% CI | Verdict |
|--------|----------------------|--------|---------|
| R@1 | -0.2520 | [-0.3483, -0.1663] | structure loses |
| R@5 | -0.0948 | [-0.1411, -0.0476] | structure loses |
| MRR | -0.1648 | [-0.2490, -0.0869] | structure loses |

**Confound found while checking this.** The comparison is *not* size-matched.
Structure-aware chunks average 202.7 words (median 195.5, 872 chunks);
fixed windows average 298.5 words (median 300, 915 chunks), because many
sections are shorter than 300 words and the 80-word floor merges the rest.
Span containment rewards the longer chunk, so part of the 25-point R@1 gap is
a length effect, not a structure effect. The factorial in `E-004` and the
equal-word-budget metric exist to separate them.

**Decision.** Keep structure-aware chunking for citation and presentation
(section paths, span highlighting), but stop describing it as a retrieval
improvement. The README and docs/eval.md say this explicitly.

---

## E-003 — Fusion tie-breaking must not depend on channel order

**Date:** 2026-11-21
**Commit:** `feature/121-fusion-diagnosis` (this change)
**Status:** fix merged pending review; aggregate effect not yet measured

**Hypothesis.** `HybridRetriever.search` sorted fused results by RRF score
alone. RRF scores tie whenever one document per channel lands at the same rank,
and Python's sort is stable, so ties were resolved by dict insertion order —
the dense channel was inserted first and therefore won every tie. Fusion is
symmetric in its inputs; the output must be too.

**Change.** `_fused_order` sorts by `(-score, chunk_id)`. Three unit tests in
`tests/test_hybrid_search.py` pin the behaviour: ties break by id, reversing
channel order leaves the fused list unchanged, and a strictly higher score
still outranks a smaller id.

**Numbers.** No aggregate number yet. The tie case is rare at the top of the
list, and claiming a recall change without measuring it would be exactly the
kind of fake number this project forbids. The factorial run records the
post-fix numbers; if they differ from `E-002`, the difference is attributable
to this fix plus nothing else.

**Decision.** Keep. It is a determinism and correctness fix regardless of
aggregate effect.

---

## E-004 — Full factorial: chunking × retriever × window size

**Date:** 2026-11-21
**Commit:** `3aa195f` plus the `_fused_order` fix above
**Status:** measured; decision pending product review

**Hypothesis.** The five-config comparison conflated chunking with retriever and
compared unequal chunk sizes (203 vs 298 words). Crossing the two variables at a
matched nominal window size, and reporting recall at an equal retrieved word
budget, should show whether the section prior helps once the length confound is
removed.

**Change.** `scripts/factorial_eval.py`: five chunkings (`struct`, `win150`,
`win200`, `win300`, `win400`) × four retrievers (`bm25`, `dense`, `hybrid`,
`hybrid_rerank`), 180 answerable dev queries, plus paired bootstrap intervals
by contract on four metrics.

**Result.** Structure loses at matched size under every retriever. Full tables
are in `docs/eval.md`; the headline intervals:

| Retriever | Metric | struct − win300 | 95% CI |
|-----------|--------|-----------------|--------|
| bm25 | R@1 | -0.1147 | [-0.1693, -0.0603] |
| dense | R@1 | -0.1169 | [-0.1826, -0.0540] |
| hybrid | R@1 | -0.2498 | [-0.3489, -0.1636] |
| hybrid_rerank | R@1 | -0.0507 | [-0.1145, +0.0187] |
| hybrid | R@1500w | -0.0542 | [-0.0913, -0.0178] |

Equal-word-budget recall removes the bm25 and dense losses (intervals include
zero) but not the hybrid one, so the hybrid gap is not purely a length effect.
Window size dominates: `win400` has the best R@1 for all four retrievers
(0.7190 bm25, 0.6323 dense, 0.6987 hybrid, 0.7400 hybrid_rerank), though at a
fixed 1500-word budget `win300` is better for bm25 (0.8559 vs 0.8316).

**Decision.** Do not claim the section prior improves retrieval. Keep it for
citations and presentation. Open follow-ups: why section-first chunking hurts
hybrid specifically, and whether 400-word windows are worth the citation
granularity they cost.

**What surprised us.** The section prior is not neutral. It is actively
harmful for recall, and more so for the fused retriever than for either channel
alone.
