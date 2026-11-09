# Data: CUAD v1

Clausewright evaluates on the Contract Understanding Atticus Dataset (CUAD) v1. No other
corpus is used, and no contract text is stored in git.

## Provenance

| Field | Value |
|-------|-------|
| Dataset | Contract Understanding Atticus Dataset (CUAD) v1 |
| Curator | The Atticus Project, Inc. (non-profit), `info@atticusprojectai.org` |
| Project page | <https://www.atticusprojectai.org/cuad> |
| Archive | <https://zenodo.org/records/4595826> (DOI `10.5281/zenodo.4595826`) |
| File | `CUAD_v1.zip`, 105.9 MB |
| Published | 2021-03-11 (record last modified 2021-07-29) |
| Paper | *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021 Datasets and Benchmarks, <https://arxiv.org/abs/2103.06268> |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Checksums | SHA-256 `88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e` (pinned in `scripts/fetch_cuad.py`); MD5 `c38f490a984420b8a62600db401fafd5` (published by Zenodo) |
| Maintenance | v1 is the current release; the datasheet records no errata and says changes will be announced on the project site. The Kaggle mirror lists update frequency as "never". |
| Verified on | 2026-11-09 against the Zenodo record and the project page |

The archive contains annotations for 510 contracts, 510 text files, and 199 PDFs. Only the
199 contracts with a PDF can be evaluated, because the pipeline parses PDFs rather than
CUAD's own text extraction.

## Licence notes

- The annotations and the dataset distribution are CC BY 4.0: commercial and non-commercial
  use are permitted with attribution.
- The datasheet states that the creators make no representations about the licence status of
  the underlying contracts, which are public SEC EDGAR filings. We rely on CUAD's own
  distribution and do not redistribute the contracts.
- Attribution required by CC BY 4.0 is carried in `README.md` and this file.

## Local layout

```
data/raw/CUAD_v1/CUAD_v1.json           annotations (SQuAD 2.0 format)
data/raw/CUAD_v1/full_contract_pdf/     199 contract PDFs
data/eval/dev_manifest.json             built eval items for the dev split
data/eval/test_manifest.json            built eval items for the test split
```

All of `data/` is gitignored except the source manifest. Eval manifests contain gold clause
spans, so they are treated as contract-derived and never committed.

## Reproducing

```bash
python scripts/fetch_cuad.py            # download and verify the archive
python scripts/fetch_cuad.py --verify   # verify an existing copy
python -m clausewright.eval dataset     # rebuild dev/test manifests
```

The eval set is 30 contracts × 15 clause categories: 20 contracts / 300 items on dev
(164 answerable, 136 absent-clause), 10 contracts / 150 items on test (59 answerable,
91 absent-clause). The split is by contract, seeded, and never tuned across.
