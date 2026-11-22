# Clausewright

Retrieval and grounded question answering over contracts, built from scratch in Python.

Contracts are not flat text. They have a section hierarchy, cross-references
("subject to Section 12.3(b)"), and defined terms. Clausewright chunks along that
structure, retrieves with a dense + lexical hybrid, and refuses to answer when the
retrieved evidence is weak.

Nothing here is legal advice.

## Status

Pre-alpha. Ingestion (PDF to hierarchical sections to chunks), retrieval
(BM25, dense embeddings, RRF fusion, cross-encoder reranking), and a grounded
answer engine (retrieve, generate, verify every citation against retrieved
text, abstain when evidence is weak) are built and unit tested.

Retrieval is measured on CUAD dev (180 answerable queries, 20 contracts).
Numbers and confidence intervals live in `docs/eval.md`; this file does not
quote its own. What the measurements currently say:

- Structure-aware chunking (sections, then 300-word windows) **loses** to fixed
  300-word windows for un-reranked hybrid retrieval: R@1 is 0.1177 lower (95% CI
  [-0.1952, -0.0430]). With the cross-encoder reranker the difference is
  [-0.1148, +0.0172], which includes zero. Structure is kept for citation
  granularity, not because it retrieves better.
- Hybrid fusion is **not** worse than BM25 alone: post-fix R@1 is 0.5604
  against 0.5067, and the interval on the difference includes zero. An earlier
  five-point deficit was a duplicate-chunk-id bug, now fixed.
- Answer quality (citation validity, abstention precision and recall, judged
  faithfulness) has **no published run yet**. The code paths exist and are
  unit tested. Every answer-side feature is experimental until `docs/eval.md`
  carries a number for it.

## Scope

- Public contracts only (see `data/manifest.csv` for source and license of each file).
- Answers must cite section numbers that were actually retrieved.
- When evidence is insufficient, the system abstains instead of guessing.

## Non-goals

- Legal advice. This is an engineering project.
- Multi-tenant auth, deployment, or scale beyond a single machine.
- General-purpose "chat with any PDF".

## Evaluation

- `docs/eval.md` — measured retrieval results, metric definitions, and the
  split policy.
- `docs/experiments.md` — hypothesis, change, dev numbers with intervals, and
  decision for every experiment, negative results included.
- `docs/data.md` — CUAD provenance and the local layout.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
