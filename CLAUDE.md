# Clausewright — Project Rules

This file is the authoritative source of rules for every agent and contributor working on
Clausewright. Read it completely before writing code, opening an issue, or pushing a branch.

---

## What this is

Structure-aware retrieval and grounded question answering over legal contracts. The system
retrieves clauses along the document's own hierarchy, answers only from retrieved text,
verifies every citation, and abstains when the evidence is weak.

It is not legal advice, and no document, UI string, or README line may imply otherwise.

---

## Product rules

- **Numbers or it did not happen.** A quality change ships with before/after numbers on the
  dev split, and one test-split report per milestone. If it loses, merge it behind a config
  flag defaulted off and record the negative result in `docs/eval.md`.
- **Abstention is a feature.** Coverage and citation validity are published together; never
  trade one for the other silently.
- **No citation, no claim.** Every answer claim names a retrieved section and quotes it.
- **Gold labels are human.** CUAD annotations or hand-written cases only. A model never
  generates the labels it is graded against.
- **Public contracts only.** No customer data in the repo, tests, fixtures, or logs.
- **Bad numbers get published.** `docs/eval.md` reports what the system actually does.

---

## Phases

Work proceeds P0 → P8 in order. Stop at the end of each phase and report; wait for
"continue" before starting the next.

| Phase | Scope |
|-------|-------|
| P0 | Audit, rules, labels, milestones, templates, Makefile, pre-commit |
| P1 | Measurement gate: dataset, adversarial drafts, retrieval and answer metrics |
| P2 | Answer quality and product depth, each with ablation numbers |
| P3 | Backend: `/v1` API, auth, tenancy, streaming, metering, audit |
| P4 | Security: threat model, injection suite, upload hardening, supply chain |
| P5 | Frontend: Next.js app, design system, accessible citation UX |
| P6 | Reliability: compose, observability, load test, CI, release |
| P7 | Documentation: README with real results, ADRs, limits, write-up |
| P8 | Demo prep: demo mode, spend cap, deploy steps (do not deploy) |

---

## Security rules

- **Never commit** contracts, customer data, `.env` files, built indexes, embedding caches,
  or eval result files that embed contract quotes.
- **Never log** contract text, prompt text, LLM completions, or API keys.
- **Never send** contract text to a third-party service without telling the owner first.
- Every LLM-calling path has a **record/replay cache** and a hard `--max-llm-calls` cap.
- API keys come from **environment variables only** — never code, config files, or CLI args.
- Tenant filters are applied **inside** the BM25 and vector queries, not after them.
- Uploads are validated by content, parsed in a sandbox with time and memory limits.
- Before every push, scan the staged diff for secrets, `data/`, and large files.

---

## Git and GitHub rules

- `gh` must be authenticated. Never print or store tokens.
- **Never force-push, never rewrite pushed history, never backdate or fake timestamps.**
- One issue per work item: problem, acceptance criteria, how to verify, labels, milestone.
- Branch per issue: `feature/<issue>-<slug>`, `bugfix/<issue>-<slug>`, `refactor/<issue>-<slug>`.
- Atomic Conventional Commits in human wording, no buzzwords, referencing the issue number.
- **No AI attribution.** No "Generated with", no `Co-Authored-By` bot lines, no tool trailers.
- One PR per issue: `Closes #n`, Summary, Key Changes, How to Test, eval numbers when relevant.
- Merge with a merge commit or rebase merge — **never squash**. Delete the branch after merge.
- Run `ruff check`, `mypy src`, and `pytest` before every push.
- Do **not** add a `LICENSE` file; the owner decides licensing.
- Ask before: making the repo public, enabling branch protection, adding a paid service,
  deploying anything, or deleting anything.

---

## Eval rules

- Tune on dev only. Report test once per milestone, then leave test alone.
- Every run writes JSON with git SHA, model id, config, seed, and dataset hash.
- Adversarial cases stay `status: draft` until the owner approves them; drafts never count
  toward headline numbers.
- Answer metrics include a deterministic qualifier/number guard, not only a judged score.
- Metric code has hand-computed unit tests. A metrics bug is a fake number.

---

## Design rules

- It should read as a serious legal instrument, not a startup landing page: ink and paper
  palette, one restrained accent, flat surfaces, no gradient meshes, no emoji in product UI.
- Typography is deliberate: a legal-document serif for contract text, a neutral grotesk for
  interface chrome. Body copy stays at a comfortable reading size and measure.
- Contract text is **untrusted**: render it as plain text and highlight source spans by
  character offsets. Never use `dangerouslySetInnerHTML`.
- A citation is a control, not decoration: clicking it scrolls to and highlights the exact
  span, with the section path visible.
- Abstention is a real screen: say what was searched, what was missing, and what would make
  the question answerable.
- Every state is designed: loading, streaming, empty, error, offline, rate-limited, abstained.
- WCAG AA, full keyboard navigation, visible focus, `aria-live` for streaming, reduced motion
  respected, usable down to phone width.
- No secrets in the client bundle; strict CSP.

---

## Required labels

```
type:feature   type:bug   type:docs   type:chore
area:retrieval area:qa    area:eval   area:api   area:security
area:infra     area:frontend area:ops
priority:p0    priority:p1 priority:p2
later
```

---

## Milestones

P0 through P8, matching the phase table above. Legacy M1–M9 milestones are historical only.

---

## Deferred (label `later`, do not build)

- Agentic RAG — hurts determinism and auditability for legal answers.
- Graph RAG beyond cross-reference expansion.
- OCR and table ingestion — build only if P1 shows real contracts failing at parse time.
- Playbook-based contract comparison.

---

## Definition of done

A phase is done when its issues are closed through merged PRs, CI is green, the numbers in
`docs/eval.md` match the runner output, and every claim in the README or UI is backed by a
test, a measurement, or a linked document.
