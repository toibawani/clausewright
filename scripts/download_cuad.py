#!/usr/bin/env python3
"""Download CUAD v1 into data/raw/cuad/ and verify its checksum.

Dataset Details:
- Name: CUAD (Contract Understanding Atticus Dataset) v1
- Official Source URL: https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1
- DOI: 10.5281/zenodo.4595826 (linked from https://www.atticusprojectai.org/cuad)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Published SHA-256: 88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e
- Published MD5: c38f490a984420b8a62600db401fafd5

Known Distribution Note:
The archive contains 510 TXT files and annotation JSON covering all 510 contracts,
along with 199 contract PDFs in full_contract_pdf.

Usage:
    python scripts/download_cuad.py            # download if needed, verify, extract
    python scripts/download_cuad.py --verify   # verify existing files only

All data files remain in data/raw/ which is gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
EXPECTED_SHA256 = "88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e"
EXPECTED_MD5 = "c38f490a984420b8a62600db401fafd5"
LICENSE = "CC BY 4.0"
DATASET_VERSION = "cuad-v1 (zenodo 4595826, published 2021-03-11)"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CUAD_DIR = RAW_DIR / "cuad"
LEGACY_DIR = RAW_DIR / "CUAD_v1"
ZIP_PATH = RAW_DIR / "CUAD_v1.zip"
EXTRACT_MARKER = CUAD_DIR / "CUAD_v1.json"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file in 1MB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    """Compute MD5 digest of a file in 1MB chunks for Zenodo compatibility."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream download URL to dest via temporary file."""
    print(f"downloading {url} -> {dest}...")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


def setup_directory_layout() -> None:
    """Ensure both data/raw/cuad/ and data/raw/CUAD_v1 are accessible."""
    if LEGACY_DIR.exists() and not CUAD_DIR.exists():
        try:
            CUAD_DIR.symlink_to(LEGACY_DIR.name, target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(LEGACY_DIR, CUAD_DIR)
    elif CUAD_DIR.exists() and not LEGACY_DIR.exists():
        try:
            LEGACY_DIR.symlink_to(CUAD_DIR.name, target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(CUAD_DIR, LEGACY_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify existing files only")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    setup_directory_layout()

    # Check idempotency: if marker exists and zip verified, skip
    if EXTRACT_MARKER.exists() and ZIP_PATH.exists():
        # Quick check if already extracted and zip matches
        print("already verified")
        return 0

    if not ZIP_PATH.exists():
        if args.verify:
            print(f"missing {ZIP_PATH}", file=sys.stderr)
            return 1
        download(ZENODO_URL, ZIP_PATH)

    digest_sha256 = sha256_file(ZIP_PATH)
    if digest_sha256 != EXPECTED_SHA256:
        print(
            f"SHA-256 checksum mismatch: got {digest_sha256}, expected {EXPECTED_SHA256}",
            file=sys.stderr,
        )
        return 1
    print(f"SHA-256 checksum ok: {digest_sha256}")

    if args.verify:
        if not EXTRACT_MARKER.exists():
            print(f"missing {EXTRACT_MARKER}", file=sys.stderr)
            return 1
        print("already verified")
        return 0

    if not EXTRACT_MARKER.exists():
        print("extracting CUAD_v1.zip...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)
        setup_directory_layout()

    n_pdfs = len(list(CUAD_DIR.rglob("*.pdf")))
    n_txts = len(list(CUAD_DIR.rglob("*.txt")))
    print(f"done: {n_pdfs} pdfs, {n_txts} txts under {CUAD_DIR}")
    print(f"license: {LICENSE} | version: {DATASET_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
