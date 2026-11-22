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

**Numbers.** The tie-break change was bundled with the duplicate-id fix in
`E-005`, so its own aggregate effect is not isolated. The tie case is rare at
the top of the list, and claiming a recall change without measuring it would be
exactly the kind of fake number this project forbids.

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

**Superseded in part.** The `hybrid` and `hybrid_rerank` rows here predate the
duplicate-id fix in `E-005`. After the fix, `struct:hybrid` reaches R@1 0.5604
and `struct:hybrid_rerank` 0.6302. The `bm25`, `dense`, and `winN` rows are
unaffected, because those paths never resolve results through the id map. See
`docs/eval.md` for the post-fix table and intervals.

---

## E-005 — Duplicate chunk ids made fused results unreachable

**Date:** 2026-11-22
**Branch:** `feature/121-fusion-diagnosis`
**Status:** fixed; supersedes the fusion conclusion in E-002/E-004

**Hypothesis under test.** Hybrid loses to BM25 because equal-weight RRF
discards BM25's score magnitude and lets a weak dense channel outvote it.

**What the diagnosis actually found.** One case was logically impossible: the
gold chunk at BM25 rank 1 and dense rank 2, yet absent from the fused list. A
chunk ranked first by one channel and second by the other cannot fall out of a
100-candidate fusion. Instrumenting the contract showed 19 chunks with only 13
unique `chunk_id` values. `parse_sections` emits repeated section numbers
(two sections both numbered "1"), and the id format is `doc-section-window`, so
those chunks collided. `HybridRetriever._chunk_map` keeps one chunk per id and
RRF deduplicates by id, so the shadowed chunk became unreachable through the
fused path even while `search_sparse` returned it first. `bm25` and `dense`
return chunk objects directly and were unaffected.

**Change.** `_ensure_unique_chunk_ids` gives later occurrences a deterministic
suffix (`doc-1-0-1`). Unit test: two sections numbered "1" yield unique ids and
the documented format. The same commit adds the RRF tie-break from `E-003`.

**Dev numbers** (180 answerable queries), paired by contract:

| Comparison | Metric | Difference | 95% CI |
|------------|--------|-----------|--------|
| struct:hybrid, post − pre fix | R@1 | +0.1343 | [+0.0381, +0.2439] |
| struct:hybrid, post − pre fix | R@5 | +0.0446 | [+0.0113, +0.0833] |
| struct:hybrid, post − pre fix | MRR | +0.0830 | [+0.0135, +0.1644] |
| struct:hybrid_rerank, post − pre fix | R@1 | +0.0093 | [+0.0000, +0.0238] |
| struct:hybrid − struct:bm25, post fix | R@1 | +0.0537 | [-0.0297, +0.1411] |

**Decision.** Keep. Two conclusions change: fusion is not worse than BM25
(the deficit was the bug), and the original magnitude hypothesis was never
confirmed. The RRF sweep in the same run (depth 20 and k=10 scoring 0.5194 R@1
against 0.4261 at k=60/depth 50) is **pre-fix and therefore not actionable**; it
must be rerun before anything is chosen from it.

**What surprised us.** A silent id collision cost 13 R@1 points and produced a
confident, wrong conclusion about the fusion algorithm.

---

## E-006 — Where the date-category misses actually come from

**Date:** 2026-11-22
**Script:** `scripts/classify_date_misses.py --split dev`
**Status:** measured; follow-ups in #122, related to #115

**Hypothesis.** Date questions fail because PDF extraction drops the date, or
because it lands in a chunk the retriever never sees.

**Method.** For every dev Agreement Date and Effective Date query with no hit
in the top 10, check three things in order: is the gold token run present in
the parsed document at all (extraction), is it present in any chunk
(chunk assignment), is it present in a chunk but outside the top 10 (ranking).
The script prints only counts, section numbers, and booleans, never contract
text.

**Result.** 19 misses: **17 ranking, 2 extraction, 0 chunk-assignment.** In both
extraction cases the lost span was six tokens long. The ranking cases have
gold spans of three to five tokens (a date) sitting inside real chunks that the
query did not surface: the retrieved top three sections are plausible contract
sections, just not the one holding the date.

**Decision.** Extraction is not the main story and chunk assignment is not a
problem at all. The dominant failure is ranking for questions with almost no
lexical signal ("Highlight the parts ... Agreement Date"), which points at
query-side category expansion rather than at the parser. Follow-up issues:
extraction for the two lost spans, and a category-aware query expansion
experiment, both under #122.

**What surprised us.** The existing docs blamed PDF extraction for date
failures. On dev that accounts for two of nineteen.

## E-008: selection rule amended to equal-budget recall (2026-11-22)

**What changed and why.** The selection rule in E-007 used paired R@5. That
metric rewards bigger windows mechanically: five 500-word chunks put far more
text in front of the reader than five 200-word chunks, so a large-window
configuration can win R@5 without retrieving the right clause any better. Using
it to pick the default would have selected the largest window for the wrong
reason.

**Amended rule.** Primary metric: recall at an equal retrieved budget of 1500
words. Secondary: recall at 3000 words. Once B4 exists, localized-span hit rate
and character precision join at the same budget. The mechanics are unchanged:
the smallest chunk-size configuration whose paired difference against the best
has a 95 percent interval including zero for `hybrid_rerank`, preferring a
structure-respecting variant when one qualifies.

**Status: exploratory.** The rule was amended after results had been seen, so
the choice it produces is not confirmatory. The final default will be confirmed
once on a source that has not been used for tuning: LegalBench-RAG, or a
contract-disjoint split of CUAD.

**Consequence for B6.** The baseline freeze for the R@5-selected configuration
is held; it is not a decision that survives this amendment.

**Applied by.** `scripts/select_default.py` (rewritten for equal-budget
recall), which prints the ordering, the decision, and the R@3000w difference for
every configuration.
