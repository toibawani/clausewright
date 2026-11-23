# Held-out log

Every look at the contract-disjoint held-out split is a row here. The log is
append-only: rows are never edited or removed, so the number of times the split
has been seen stays countable.

The split is frozen in [`benchmarks/heldout_manifest.json`](../benchmarks/heldout_manifest.json)
and the only configuration a look may use is
[`benchmarks/heldout_finalist.json`](../benchmarks/heldout_finalist.json). A run
is refused unless `CLAUSEWRIGHT_FINAL_RUN=1` is set and `--finalist` names that
file; see `src/clausewright/eval/heldout.py`.

Fewer rows is better. Each row after the first makes the next interval wider and
the result less like a held-out test and more like another dev run.

| timestamp (UTC) | commit | configuration | result | note |
| --- | --- | --- | --- | --- |
