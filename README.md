# Clausewright

Retrieval and grounded question answering over contracts, built from scratch in Python.

Contracts are not flat text. They have a section hierarchy, cross-references
("subject to Section 12.3(b)"), and defined terms. Clausewright chunks along that
structure, retrieves with a dense + lexical hybrid, and refuses to answer when the
retrieved evidence is weak.

## Status

Milestone 1 of 9 (project skeleton). Nothing is evaluated yet, so this README makes
no accuracy claims. Results will be added in the final milestone, from a hand-labelled
gold set, whatever they turn out to be.

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
