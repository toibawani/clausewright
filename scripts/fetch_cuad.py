#!/usr/bin/env python3
"""Download CUAD v1 into data/raw/ and verify its checksum.

Source of truth: Zenodo DOI 10.5281/zenodo.4595826 (linked from
atticusprojectai.org/cuad). License: CC BY 4.0 (same record). The published
md5 for CUAD_v1.zip is c38f490a984420b8a62600db401fafd5; we refuse to
extract anything that does not match.

Known distribution gap: the zip contains 199 contract PDFs (not the 510 the
README states) plus 510 TXT files. Annotation JSON covers all 510 contracts.
Eval selection therefore only offers contracts whose PDF is present.

Usage:
    python scripts/fetch_cuad.py            # download if needed, verify, extract
    python scripts/fetch_cuad.py --verify   # verify existing files only

data/raw/ is gitignored; this script never commits anything.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
EXPECTED_MD5 = "c38f490a984420b8a62600db401fafd5"
LICENSE = "CC BY 4.0"
DATASET_VERSION = "cuad-v1 (zenodo 4595826, published 2021-03-11)"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "CUAD_v1.zip"
EXTRACT_MARKER = RAW_DIR / "CUAD_v1" / "CUAD_v1.json"


def md5_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 — integrity check against a published checksum, not security
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:  # noqa: S310 — fixed trusted URL
        while block := resp.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify existing files only")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        if args.verify:
            print(f"missing {ZIP_PATH}", file=sys.stderr)
            return 1
        download(ZENODO_URL, ZIP_PATH)

    digest = md5_file(ZIP_PATH)
    if digest != EXPECTED_MD5:
        print(
            f"checksum mismatch: got {digest}, expected {EXPECTED_MD5} — refusing to use it",
            file=sys.stderr,
        )
        return 1
    print(f"checksum ok: {digest}")

    if args.verify:
        if not EXTRACT_MARKER.exists():
            print(f"missing {EXTRACT_MARKER}", file=sys.stderr)
            return 1
        print("extraction present")
        return 0

    if not EXTRACT_MARKER.exists():
        print("extracting...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)  # noqa: S202 — archive from a checksum-verified trusted source

    n_pdfs = len(list((RAW_DIR / "CUAD_v1" / "full_contract_pdf").rglob("*.pdf")))
    n_txts = len(list((RAW_DIR / "CUAD_v1" / "full_contract_txt").glob("*.txt")))
    print(f"done: {n_pdfs} pdfs, {n_txts} txts under {RAW_DIR / 'CUAD_v1'}")
    print(f"license: {LICENSE} | version: {DATASET_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
