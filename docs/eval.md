# Evaluation

Every number here comes from a named script and is reproducible from a commit.
Nothing is estimated. If a feature has no number in this file, the docs call it
experimental.

## What is measured

**Corpus.** CUAD v1, 30 contracts across 15 clause categories. The split is by
contract, seeded (`20261018`), and never tuned across. See `docs/data.md` for
provenance.

| Split | Contracts | Items | Answerable | Absent-clause |
|-------|-----------|-------|------------|---------------|
| dev | 20 | 300 | 180 | 120 |
| test | 10 | 150 | 81 | 69 |

Counts verified against `build_eval_set(CUAD_SMALL)` on 2026-11-21; dataset hash
`5174daa11351d36c02eee257e924db79`.

Absent-clause items are the abstention tests. They are excluded from retrieval
metrics, which are defined only over items with a gold span.

## Metric definitions

**Span matching.** CUAD gold spans come from CUAD's own text extraction; our
chunks come from our PDF extraction, so character offsets never line up. A
chunk hits a gold span when at least 50 percent of the span's tokens appear in
it (multiset containment, lowercased, whitespace collapsed). The threshold is
`SPAN_OVERLAP_THRESHOLD` in `src/clausewright/eval/metrics.py`, with
hand-computed unit tests in `tests/eval/test_retrieval_metrics.py`.

**Recall@k.** Fraction of an item's gold spans hit by at least one of the top-k
chunks; mean over answerable items. R@1 answers "is it first", R@5 "is it in the
window a model would read", R@10 "was it retrieved at all".

**MRR.** Mean of `1/rank` of the first chunk that hits any gold span; 0.0 when
nothing hits.

**Recall at an equal retrieved budget.** Chunks are taken in ranked order until
the next one would exceed 1500 (or 3000) words. Recall is the fraction of gold
spans hit by that selected set. This is the metric that does not reward a
chunker merely for producing longer chunks.

**Paired bootstrap interval.** 95 percent interval for the mean difference
between two configurations, resampling *contracts* with replacement (10000
resamples, seed `20261018`). Queries inside one contract are not independent, so
resampling queries would understate the interval. Implemented in
`src/clausewright/eval/stats.py`; a difference whose interval includes zero is
reported as noise.

**Abstention.** Precision and recall of the abstain decision, where a positive
is an abstention on an absent-clause item. A false abstention is an abstention
on an answerable item.

## Split policy

The CUAD test split counts as **seen**. It has been reported, inspected, and
written about, so it can no longer support a claim of generalisation. Final
claims about accuracy must come from one frozen configuration evaluated once on
either LegalBench-RAG or a fresh contract-disjoint split that has not been
looked at. Until that exists, every number in this file is a dev number and is
labelled as such.

## Dev results, current main (`3aa195f`)

180 answerable queries, 20 contracts. Produced by
`scripts/run_dev_eval.py`; per-config JSON in `results/*-3aa195f.json`.

| Config | R@1 | R@3 | R@5 | R@10 | MRR |
|--------|-----|-----|-----|------|-----|
| bm25 | 0.5067 | 0.7328 | 0.7965 | 0.8594 | 0.6748 |
| dense | 0.5106 | — | 0.7433 | 0.8433 | 0.6519 |
| hybrid | 0.4261 | 0.6662 | 0.7459 | 0.8436 | 0.6040 |
| hybrid_rerank | 0.6209 | — | 0.8216 | 0.8872 | 0.7401 |
| fixed_window | 0.6781 | — | 0.8407 | 0.8722 | 0.7688 |

Chunk sizes: structure-aware 872 chunks, mean 202.7 words, median 195.5.
Fixed-window 915 chunks, mean 298.5 words, median 300. The comparison is not
size-matched; the factorial below is.

## Full factorial (`3aa195f`, dev, 2026-11-21)

Five chunkings crossed with four retrievers, 180 answerable queries, 20
contracts. Produced by `scripts/factorial_eval.py --split dev`; JSON written to
gitignored `data/eval/factorial-dev.json`. `struct` is the product chunker;
`winN` is the same windowing over one pseudo-section at N words, 50 percent
overlap.

Chunk sizes over all 20 contracts:

| Chunking | Chunks | Mean words | Median | p90 | Max |
|----------|--------|-----------|--------|-----|-----|
| struct | 872 | 202.7 | 195.5 | 300.0 | 370 |
| win150 | 1839 | 149.8 | 150 | 150.0 | 226 |
| win200 | 1379 | 199.2 | 200 | 200.0 | 200 |
| win300 | 915 | 298.5 | 300 | 300.0 | 300 |
| win400 | 685 | 396.8 | 400 | 400.0 | 400 |

| Config | R@1 | R@5 | R@10 | MRR | R@1500w | R@3000w |
|--------|-----|-----|------|-----|---------|---------|
| struct:bm25 | 0.5067 | 0.7965 | 0.8594 | 0.6748 | 0.8353 | 0.8594 |
| struct:dense | 0.5106 | 0.7433 | 0.8433 | 0.6519 | 0.8116 | 0.8433 |
| struct:hybrid | 0.4261 | 0.7459 | 0.8436 | 0.6040 | 0.7866 | 0.8436 |
| struct:hybrid_rerank | 0.6209 | 0.8216 | 0.8872 | 0.7401 | 0.8570 | 0.8872 |
| win150:bm25 | 0.3839 | 0.6891 | 0.7762 | 0.5610 | 0.7762 | 0.7762 |
| win150:dense | 0.4606 | 0.7118 | 0.8123 | 0.6115 | 0.8123 | 0.8123 |
| win150:hybrid | 0.5379 | 0.7179 | 0.7943 | 0.6556 | 0.7943 | 0.7943 |
| win150:hybrid_rerank | 0.5331 | 0.7379 | 0.8181 | 0.6707 | 0.8181 | 0.8181 |
| win200:bm25 | 0.4845 | 0.8031 | 0.8438 | 0.6684 | 0.8192 | 0.8438 |
| win200:dense | 0.5060 | 0.7579 | 0.8316 | 0.6594 | 0.8045 | 0.8316 |
| win200:hybrid | 0.5984 | 0.7780 | 0.8375 | 0.7154 | 0.7986 | 0.8375 |
| win200:hybrid_rerank | 0.5712 | 0.8320 | 0.8528 | 0.7150 | 0.8450 | 0.8528 |
| win300:bm25 | 0.6214 | 0.8559 | 0.8907 | 0.7614 | 0.8559 | 0.8907 |
| win300:dense | 0.6275 | 0.8133 | 0.8732 | 0.7433 | 0.8133 | 0.8732 |
| win300:hybrid | 0.6759 | 0.8407 | 0.8722 | 0.7651 | 0.8407 | 0.8722 |
| win300:hybrid_rerank | 0.6716 | 0.8785 | 0.9065 | 0.7994 | 0.8785 | 0.9065 |
| win400:bm25 | 0.7190 | 0.8787 | 0.9148 | 0.8072 | 0.8316 | 0.9019 |
| win400:dense | 0.6323 | 0.8438 | 0.9028 | 0.7492 | 0.8080 | 0.8683 |
| win400:hybrid | 0.6987 | 0.8676 | 0.8889 | 0.7866 | 0.8264 | 0.8759 |
| win400:hybrid_rerank | 0.7400 | 0.8880 | 0.9315 | 0.8338 | 0.8815 | 0.9046 |

Structure against a size-matched fixed window (300 words), paired by contract,
10000 resamples:

| Retriever | Metric | Difference | 95% CI | Verdict |
|-----------|--------|-----------|--------|---------|
| bm25 | R@1 | -0.1147 | [-0.1693, -0.0603] | structure loses |
| bm25 | R@5 | -0.0595 | [-0.1074, -0.0144] | structure loses |
| bm25 | R@1500w | -0.0206 | [-0.0560, +0.0137] | noise |
| dense | R@1 | -0.1169 | [-0.1826, -0.0540] | structure loses |
| dense | R@5 | -0.0700 | [-0.1215, -0.0200] | structure loses |
| dense | R@1500w | -0.0017 | [-0.0450, +0.0433] | noise |
| hybrid | R@1 | -0.2498 | [-0.3489, -0.1636] | structure loses |
| hybrid | R@5 | -0.0948 | [-0.1411, -0.0476] | structure loses |
| hybrid | R@1500w | -0.0542 | [-0.0913, -0.0178] | structure loses |
| hybrid_rerank | R@1 | -0.0507 | [-0.1145, +0.0187] | noise |
| hybrid_rerank | R@5 | -0.0570 | [-0.1032, -0.0121] | structure loses |
| hybrid_rerank | R@1500w | -0.0215 | [-0.0519, +0.0083] | noise |

Three things follow, and none of them flatters the design:

- At matched nominal size, structure loses on recall@k for every retriever.
  With bm25 and dense the loss shrinks to noise once the budget is equal
  words, which says the penalty for section-first chunking is mostly that it
  produces shorter chunks. With hybrid the loss survives the budget control.
- Window size dominates every other choice here. `win400` beats `win300` on
  R@1 for all four retrievers, but part of that is packing 400 words into a
  slot; at a 1500-word budget `win300` is better for bm25 (0.8559 vs 0.8316).
- The best cell in the table is `win400:hybrid_rerank` at R@1 0.7400. The
  product chunker's best cell is 0.6209.

## Post-fix structure rerun (2026-11-22)

The factorial above was run before the duplicate-chunk-id fix (E-005). That bug
only affected configurations that resolve results through the retriever's
id-keyed map, which is why `bm25` and `dense` did not move while `hybrid` did.
The table below reruns the `struct` chunking after the fix; the `winN` rows are
unaffected because fixed windows never produce duplicate ids.

| Config | R@1 | R@5 | R@10 | MRR | R@1500w | R@3000w |
|--------|-----|-----|------|-----|---------|---------|
| struct:bm25 | 0.5067 | 0.7965 | 0.8594 | 0.6748 | 0.8353 | 0.8594 |
| struct:dense | 0.5106 | 0.7433 | 0.8433 | 0.6519 | 0.8116 | 0.8433 |
| struct:hybrid | 0.5604 | 0.7905 | 0.8760 | 0.6871 | 0.8093 | 0.8760 |
| struct:hybrid_rerank | 0.6302 | 0.8350 | 0.9117 | 0.7518 | 0.8704 | 0.9062 |

What changed, paired by contract:

| Comparison | Metric | Difference | 95% CI |
|------------|--------|-----------|--------|
| struct:hybrid, post − pre fix | R@1 | +0.1343 | [+0.0381, +0.2439] |
| struct:hybrid, post − pre fix | R@5 | +0.0446 | [+0.0113, +0.0833] |
| struct:hybrid, post − pre fix | MRR | +0.0830 | [+0.0135, +0.1644] |
| struct:hybrid_rerank, post − pre fix | R@1 | +0.0093 | [+0.0000, +0.0238] |
| struct:hybrid − struct:bm25 (post fix) | R@1 | +0.0537 | [-0.0297, +0.1411] |
| struct:hybrid − struct:bm25 (post fix) | R@5 | -0.0059 | [-0.0471, +0.0359] |
| struct:hybrid − fixed_window (post fix) | R@1 | -0.1177 | [-0.1952, -0.0430] |
| struct:hybrid_rerank − fixed_window (post fix) | R@1 | -0.0479 | [-0.1148, +0.0172] |
| struct:hybrid_rerank − fixed_window (post fix) | R@5 | -0.0057 | [-0.0592, +0.0463] |

Three conclusions replace the earlier ones:

- **Hybrid is not worse than BM25.** The earlier five-point R@1 deficit was the
  duplicate-id bug. After the fix the difference is +0.0537 with an interval
  that includes zero: the two are indistinguishable at this sample size.
- **Structure still loses to fixed 300-word windows without reranking**
  (-0.1177 R@1, CI [-0.1952, -0.0430]).
- **With reranking, the two are statistically tied** (R@1 -0.0479, CI includes
  zero). The reranker recovers what the section-first chunking costs, and the
  case for structure now rests on citation granularity, not recall.

## P1 baseline (historical, `9ef612c`)

Kept because it is the number the original docs quoted. It predates the chunker
fix in E-001 and should not be cited as current.

| Config | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| bm25 | 0.4357 | 0.7665 | 0.8407 | 0.6106 |
| dense | 0.4890 | 0.7193 | 0.8087 | 0.6269 |
| hybrid | 0.3304 | 0.6470 | 0.7842 | 0.4824 |
| hybrid_rerank | 0.5364 | 0.7791 | 0.8345 | 0.6701 |
| fixed_window | 0.6781 | 0.8407 | 0.8722 | 0.7688 |

## Test results (`9ef612c`, seen split)

81 answerable queries, 10 contracts. Reported once for the original milestone
and not used for tuning. Kept for the record; the split is seen.

| Config | R@1 | R@3 | R@5 | R@10 | MRR |
|--------|-----|-----|-----|------|-----|
| bm25 | 0.407 | 0.640 | 0.754 | 0.814 | 0.618 |
| dense | 0.499 | 0.662 | 0.748 | 0.817 | 0.691 |
| hybrid | 0.384 | 0.615 | 0.727 | 0.837 | 0.610 |
| hybrid_rerank | 0.437 | 0.663 | 0.769 | 0.865 | 0.651 |
| fixed_window | 0.575 | 0.760 | 0.818 | 0.900 | 0.759 |

## What the numbers say

1. **Structure-aware chunking loses to fixed windows on retrieval**, and size
   matching does not rescue it: at 300 words the loss is -0.1147 R@1 for bm25
   ([-0.1693, -0.0603]) and -0.2498 for hybrid ([-0.3489, -0.1636]). Equalizing
   the retrieved budget turns the bm25 and dense losses into noise but leaves
   the hybrid loss standing (-0.0542, [-0.0913, -0.0178]).
2. **Equal-weight RRF fusion is not worse than BM25 alone** once the
   duplicate-id bug is fixed: hybrid R@1 is 0.5604 against 0.5067 for bm25, a
   difference whose interval includes zero. The pre-fix table in the factorial
   section shows the 0.4261 that made fusion look broken.
3. **The chunker fix mattered more than the section prior.** Restoring the
   final short chunk moved structure-aware BM25 R@1 from 0.4357 to 0.5067
   (+0.0709, CI [+0.0100, +0.1423]) while fixed windows did not move.
4. **Date categories dominate the worst misses.** Agreement Date and Effective
   Date usually score 0.00 overlap: the retrieved sections are real sections,
   but the gold date text is not in them. This points at extraction or section
   assignment, not at the ranker.

## Not measured yet

**Answer quality.** Citation validity, abstention precision and recall, the
qualifier and number guard, and judged faithfulness are implemented and unit
tested, but no answer run has been made. Producing one requires sending
contract text to an LLM provider, which needs the owner's approval and a stated
call budget. Until then this document, the README, and the UI do not claim
answer quality. Retrieval-only numbers are the honest baseline.

## Limits

- CUAD questions are annotation prompts ("Highlight the parts ... related to
  X"), not the way a lawyer asks. Absolute numbers will not transfer to ad-hoc
  questions.
- 15 of CUAD's 41 categories and 30 of 510 contracts are used; only 199
  contracts ship with PDFs.
- Span containment rewards longer chunks. That bias is a real confound in the
  headline comparison, which is why the factorial reports chunk statistics and
  equal-word-budget recall next to every recall@k.
- One machine, CPU, one run per config. With n=180 on dev, differences below
  roughly 3 points should not be treated as real without an interval.
- Retrieval misses are recorded per query in the run JSON, which lives under
  gitignored `data/eval/` because it names contracts.

## Reproducing

```bash
python -m clausewright.eval dataset            # rebuild split manifests
python -m clausewright.eval retrieval --split dev
scripts/run_dev_eval.py                        # per-config JSON in results/
scripts/factorial_eval.py --split dev          # chunking x retriever factorial
```

## Selection status (2026-11-22)

The default selected earlier by the R@5 rule (win500 with hybrid_rerank) is
**not adopted**. The rule was amended to equal-budget recall because R@5
rewards bigger windows mechanically; see E-008 in `docs/experiments.md`. The
selection is exploratory, the baseline freeze for win500 is held, and the
retrieval table above is reported as measurement rather than as a decision.
