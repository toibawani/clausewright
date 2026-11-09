# Evaluation

Numbers here come from `python -m clausewright.eval retrieval`. Nothing in this file is
estimated, and the test split is reported once per milestone.

## What is measured

**Corpus.** CUAD v1, 30 contracts across 15 clause categories. The split is by contract,
seeded (`20261018`), and never tuned across. See `docs/data.md` for provenance.

| Split | Contracts | Items | Answerable | Absent-clause |
|-------|-----------|-------|------------|---------------|
| dev | 20 | 300 | 180 | 120 |
| test | 10 | 150 | 81 | 69 |

Absent-clause items are the abstention tests. They are excluded from retrieval metrics,
which are defined only over items with a gold span.

**Span matching.** CUAD gold spans come from CUAD's own text extraction; our chunks come
from our PDF extraction, so offsets never line up. A chunk hits a gold span when at least
50% of the span's tokens appear in it (multiset containment, lowercased). `recall@k` is the
fraction of an item's gold spans hit by the top-k chunks; MRR is `1/rank` of the first hit.
The threshold is `SPAN_OVERLAP_THRESHOLD` in `src/clausewright/eval/metrics.py`, with
hand-computed unit tests.

**Configurations.**

| Config | What it is |
|--------|------------|
| `bm25` | BM25 over structure-aware section chunks |
| `dense` | BGE-small-en-v1.5 + FAISS inner product over the same chunks |
| `hybrid` | BM25 + dense fused with Reciprocal Rank Fusion (k=60) |
| `hybrid_rerank` | Hybrid top-25 reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-10 kept |
| `fixed_window` | The same chunker and hybrid retrieval over one pseudo-section: 300-word windows at 50% overlap, ignoring section boundaries |

The fixed-window config exists to isolate the value of the section prior: same tokenizer,
same window size, same retrieval, no structure.

**Environment.** Apple arm64, CPU only, Python 3.14, `OMP_NUM_THREADS=1` (set by the
embedder to avoid a faiss/torch OpenMP crash). No sampling, so runs are deterministic for a
fixed commit.

## Dev results

180 answerable queries, 20 contracts.

| Config | R@1 | R@3 | R@5 | R@10 | MRR |
|--------|-----|-----|-----|------|-----|
| bm25 | 0.436 | 0.692 | 0.766 | 0.841 | 0.611 |
| dense | 0.489 | 0.660 | 0.719 | 0.809 | 0.627 |
| hybrid | 0.330 | 0.490 | 0.647 | 0.784 | 0.482 |
| hybrid_rerank | 0.536 | 0.738 | 0.779 | 0.835 | 0.670 |
| fixed_window | **0.678** | **0.786** | **0.841** | **0.872** | **0.769** |

## Test results

81 answerable queries, 10 contracts. Reported once; not used for tuning.

| Config | R@1 | R@3 | R@5 | R@10 | MRR |
|--------|-----|-----|-----|------|-----|
| bm25 | 0.407 | 0.640 | 0.754 | 0.814 | 0.618 |
| dense | 0.499 | 0.662 | 0.748 | 0.817 | 0.691 |
| hybrid | 0.384 | 0.615 | 0.727 | 0.837 | 0.610 |
| hybrid_rerank | 0.437 | 0.663 | 0.769 | 0.865 | 0.651 |
| fixed_window | **0.575** | **0.760** | **0.818** | **0.900** | **0.759** |

## What the numbers say

1. **The structure-aware chunker loses to fixed windows at every cutoff, on both splits.**
   This contradicts the premise the product was built on, and it is the most important
   result in the repo. Structure helps answer *presentation* (section paths, citation
   precision) but is currently costing retrieval recall.
2. **RRF fusion without reranking is worse than either channel alone at R@1 and R@5 on both
   splits** (dev R@1 0.330 vs 0.436 BM25 and 0.489 dense; test 0.384 vs 0.407 and 0.499).
   The reranker repairs most of the damage (dev 0.536, test 0.437) but does not close the gap
   to fixed windows. On test, reranked MRR (0.651) is still below dense alone (0.691).
3. **Ranking, not recall, is the near-term bottleneck.** 21 of 59 dev misses have a gold
   span in ranks 6-10. The evidence is retrieved; it is ranked too low. The test split is
   kinder (16 misses in 81 queries) but shows the same shape: reranking moves R@5 by 4
   points and leaves R@1 close to the un-reranked hybrid.
4. **Date categories dominate the worst misses.** Agreement Date (12 misses) and Effective
   Date (8) usually score 0.00 overlap: the retrieved sections are real sections, but the
   gold date text is not in them. That points at extraction and section assignment, not at
   the ranker.

## Ten worst dev failures

Lowest gold-token overlap in the top 10 for the base `hybrid` config. Contract names are
public CUAD filenames; gold clause text is not reproduced here.

| Overlap | Category | Contract (truncated) | Top sections |
|---------|----------|----------------------|--------------|
| 0.00 | Agreement Date | ClickstreamCorp_20200330_1-A | 1, 8, 8 |
| 0.00 | Agreement Date | CoherusBiosciencesInc_20200227_10-K | 16, 15.2.6, 1.74 |
| 0.00 | Effective Date | CoherusBiosciencesInc_20200227_10-K | 12.1, 1.20, 15.2.1 |
| 0.00 | Agreement Date | EhaveInc_20190515_20-F | 12, 3, 10 |
| 0.00 | Effective Date | EhaveInc_20190515_20-F | 12, 10, 3 |
| 0.00 | Agreement Date | ExactSciencesCorp_20180822_8-K | 52, 53, 35 |
| 0.00 | Agreement Date | GopageCorp_20140221_10-K | 9, 10, 1 |
| 0.00 | Effective Date | GopageCorp_20140221_10-K | 9, 1, 10 |
| 0.00 | Effective Date | InmodeLtd_20190729_F-1A | 9, 7, 4 |
| 0.17 | Agreement Date | HfEnterprisesInc_20191223_S-1 | 3, 2, 1 |

Likely causes, in order of suspicion: PDF extraction dropping cover-page or preamble text;
the section parser attaching a date to the wrong heading; and CUAD's gold span normalising a
date differently from the extracted text. Each needs a per-case inspection before a fix is
claimed.

## Not measured yet

**Answer quality.** Citation validity, abstention precision and recall, the qualifier guard,
and judged faithfulness are implemented and unit tested, but no answer run has been made.
Producing one requires sending contract text to an LLM provider, which is waiting on the
owner's approval. Until then this document, the README, and the UI will not claim answer
quality. Retrieval-only numbers are the honest baseline.

## Limits

- CUAD questions are annotation prompts ("Highlight the parts ... related to X"), not the
  way a lawyer asks. Absolute numbers will not transfer to ad-hoc questions.
- 15 of CUAD's 41 categories and 30 of 510 contracts are used; only 199 contracts ship with
  PDFs.
- Span containment rewards longer chunks. That bias favours `fixed_window` and is a real
  confound in the headline comparison, not a neutral metric.
- One machine, CPU, one run per config. With n=180 on dev, differences below roughly 3
  points should not be treated as real.
- Retrieval misses are recorded per query (`failures` in the run JSON), but the saved JSON
  lives under gitignored `data/eval/` because it names contracts.

## Reproducing

```bash
python -m clausewright.eval dataset            # rebuild split manifests
python -m clausewright.eval retrieval --split dev
python -m clausewright.eval retrieval --split test
```

Each run prints the table and writes `data/eval/retrieval_results.json` with the split,
counts, threshold, and per-query failures.
