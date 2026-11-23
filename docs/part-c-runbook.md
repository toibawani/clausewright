# Part C runbook: answer evaluation

No answer-quality number exists yet. This page is what to run when the owner
decides to produce one, and what to check afterwards. Nothing here has been run
against a provider.

## What is measured

The answer layer, not retrieval. For each item in a dev set:

- **citation validity** — every claim's evidence id resolved to a block that was
  actually shown to the model;
- **citation quote rate** — the claim's quote appears verbatim in the cited
  chunk;
- **abstention precision and recall** — abstentions on absent-clause items, and
  the false-abstention rate on answerable items;
- **qualifier guard violation rate** — claims asserting shall/must/may in a way
  the quote does not support;
- **faithfulness rate** — every number, date, and qualifier in a claim appears in
  its quote.

The set is `build_answer_eval_set`: 100 dev items, 60 answerable and 40
absent-clause, drawn with seed `20261123` and ordered by item id. Abstention is
measured against the absent-clause items, so it cannot be reported alone.

## Before any call

1. The key lives in the environment only:

   ```bash
   export ANTHROPIC_API_KEY=...     # never in a file that git tracks
   ```

   The client reads `ANTHROPIC_API_KEY`; it is never read from code, config, or
   a CLI argument.

2. Decide the call budget. `--max-llm-calls` is hard: when the count is reached
   the client abstains rather than calling again. `0` means no provider call at
   all.

3. Estimate first. The estimator builds the real system prompt and the real
   user message with five evidence blocks, so its token count moves when the
   prompt moves:

   ```bash
   python scripts/part_c_dry_run.py --limit 100 --evidence 5 --out data/eval/part_c_estimate.json
   ```

   It prints the number of provider calls, the prompt-token estimate, and the
   completion-token estimate. The token figures are estimates at four characters
   per token; treat them as a ceiling check, not a bill.

4. Dry run the plumbing with the fake model. This makes no provider call:

   ```bash
   python scripts/part_c_dry_run.py --limit 20
   ```

   The printed table is marked as a dry run because the retriever, the reranker,
   and the model are stubs. It proves the pipeline produces a result with
   provenance and renders a table; it says nothing about answer quality.

## The run

```bash
python -m clausewright.eval run --split dev \
  --config hybrid_rerank \
  --max-llm-calls 100 \
  --replay-cache data/llm_cache/part-c.jsonl \
  --out data/eval/part-c-run.json
```

- `--replay-cache` records every completion. A rerun with the same cache and
  `--max-llm-calls 0` replays them and spends nothing.
- Run it from a clean tree. The runner refuses a dirty worktree, and every
  result carries the commit, clean flag, model names, and threading settings.

## How to abort

- `Ctrl-C` in the foreground. The run writes its JSON only at the end, so an
  interrupted run leaves no result file and no partial numbers to quote.
- After an abort, check `git status --short` and `git rev-parse --short HEAD`
  before starting again. Do not re-run automatically; do not loop.
- If the provider returns an error, the client raises rather than retrying
  indefinitely. Record the error text; do not retype numbers from a partial run.

## After the run

1. Confirm `provenance.clean_tree` is `true` and the commit matches the pushed
   commit you ran from.
2. Convert to the aggregate store and regenerate the docs:

   ```bash
   python scripts/to_aggregate.py --kind factorial --input data/eval/part-c-run.json \
     --experiment-id part-c-<short-sha> --protocol E-010 \
     --output benchmarks/results/part-c-<short-sha>.json
   python scripts/render_docs_tables.py --write
   python -m pytest tests/test_docs_consistency.py tests/test_results_store.py -q
   ```

3. Write the result into `docs/experiments.md` before quoting it anywhere. The
   pre-registration for the run must be pushed first, as E-009 was.

## What this runbook does not cover

- Tuning against the answer metrics. The set is dev; tuning on it makes it a
  dev set for answers too, and it then stops being a check.
- The held-out split. It needs the owner's approval message and the finalist
  hash; see `docs/heldout-log.md`.
