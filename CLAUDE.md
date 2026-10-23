# Clausewright — Project Rules

This file is the authoritative source of rules for every agent and contributor working on Clausewright.
Read it completely before writing a single line of code or opening any issue.

---

## Product Rules

- Clausewright is a **structure-aware legal-contract QA backend**.  It retrieves clauses and answers grounded questions; it does **not** give legal advice.
- Every improvement must ship **before and after numbers on the eval set**.  Tune on dev only; report test numbers once per milestone.
- When evidence is insufficient the system **abstains** — it never guesses.
- Answers must cite the **exact section numbers** that were retrieved.
- Public datasets only.  Private contracts never enter the repo or any shared service.

---

## Security Rules

- **Never commit** contracts, customer data, `.env` files, built indexes, or embedding caches.
- **Never log** contract text, prompt text, or LLM completions.
- **Never send** contract text to a third-party service without explicit user approval each time.
- Every LLM-calling code path must have a **record/replay cache** and a **hard `--max-llm-calls` cap**.
- API keys come from **environment variables only** — never from code, config files, or CLI args.
- Verify the staged diff for secrets, `data/` files, and large files before every push.

---

## Git / GitHub Workflow

- `gh` CLI must be authenticated.  If it is not, stop and instruct the user to run `gh auth login`.
- **Never force-push, never rewrite pushed history, never backdate or fake timestamps.**
- Commits are made as work happens, with the repository owner's configured git identity.
- Branch naming: `feature/<issue>-<slug>`, `bugfix/<issue>-<slug>`, `refactor/<issue>-<slug>`.
- Commit style: Atomic Conventional Commits referencing the issue number, human wording, no buzzwords.
- Run `ruff check`, `mypy`, and `pytest` before every push.
- Open one PR per issue.  PR body: title, `Closes #n`, Summary (2–3 bullets), Key Changes, How to Test, eval numbers when relevant.
- Merge strategy: **merge commit or rebase merge — never squash** (atomic commits must survive).
- Delete the branch after merge.
- Do **not** make the repo public, enable branch protection, add a paid service, or delete anything without asking first.

---

## Eval Rules

- Nothing counts as an improvement without before/after numbers on the eval set.
- Gold labels are **never generated and graded by the same model**.
- Adversarial cases in `tests/eval/adversarial.yaml` are `status: draft` until the user approves them; drafts are excluded from headline numbers.
- Eval results are JSON with git commit, model id, config, seed, and dataset hash.
- Bad numbers are published as-is in `docs/eval.md`.

---

## Required GitHub Labels

```
type:feature   type:bug   type:docs   type:chore
area:retrieval area:qa    area:eval   area:api   area:security   area:infra
priority:p0    priority:p1 priority:p2
later
```

---

## Milestones

| ID | Name |
|----|------|
| M1 | Evaluation Baseline |
| M2 | Retrieval and Product Depth |
| M3 | Service Layer |
| M4 | Production Hardening |
| M5 | Presentation and Demo |

---

## Deferred (issues labeled `later`, do not build)

- Agentic RAG — hurts determinism and auditability for legal answers
- Graph RAG beyond cross-reference expansion in M2 item 9
- Multimodal / OCR ingestion — build only if M1 shows real contracts failing at parsing
- Contract comparison against a playbook
