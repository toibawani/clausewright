# Clausewright

Retrieval and grounded question answering over contracts, built from scratch in Python.

Contracts are not flat text. They have a section hierarchy, cross-references
("subject to Section 12.3(b)"), and defined terms. Clausewright chunks along that
structure, retrieves with a dense + lexical hybrid, and refuses to answer when the
retrieved evidence is weak.

## Status

Pre-alpha. Ingestion (PDF to hierarchical sections to chunks), hybrid
retrieval (BM25 + dense embeddings, fused with RRF, cross-encoder reranked),
and a grounded answer engine (retrieve, generate, verify every citation
against retrieved text, abstain when evidence is weak) are built and unit
tested. Retrieval is measured on a hand-labelled CUAD sample and the table
below is the only accuracy claim in this README. Answer quality is implemented
and unit tested, but it is not measured yet, so nothing here covers it.

## Measurement

Retrieval runs on CUAD v1: 30 public contracts across 15 clause categories,
split by contract with a fixed seed, so the test half is never tuned against.
The labels are CUAD's own annotations. The block below is generated from the
committed result in `benchmarks/results/`, and `tests/test_docs_consistency.py`
fails when the committed text drifts from what that result produces.

<!-- BEGIN GENERATED: readme_headline -->
Dev retrieval, 20 contracts, 180 answerable queries, run `3ae9c47`, dataset hash `5174daa11351d36c02eee257e924db79`.

| Config | R@1 | R@5 | R@1500w |
|---|---|---|---|
| `struct:hybrid_rerank` | 0.6302 | 0.8350 | 0.8704 |
| `win300:hybrid_rerank` | 0.6716 | 0.8785 | 0.8785 |

Structure minus the size-matched window at a 1500-word budget: -0.0081 [-0.0417, +0.0233]; the interval includes zero, so at this sample size the two are not distinguishable.
<!-- END GENERATED: readme_headline -->

Read the two rows against each other rather than alone: structure-respecting
chunking is not better than the strongest fixed window at an equal retrieved
budget, and the difference sits inside the interval. The reranked hybrid is the
retrieval setting those numbers come from. Citation validity, abstention
precision and recall, and the qualifier guard have metric code and hand-computed
unit tests but no measurement yet.

## Scope

- Public contracts only (see `data/manifest.csv` for source and license of each file).
- Answers must cite section numbers that were actually retrieved.
- When evidence is insufficient, the system abstains instead of guessing.

## Non-goals

- Legal advice. This is an engineering project.
- Multi-tenant auth, deployment, or scale beyond a single machine.
- General-purpose "chat with any PDF".

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
