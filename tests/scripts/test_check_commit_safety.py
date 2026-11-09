"""Tests for the commit safety scanner.

The scanner is what stands between a stray key and a public commit, so the
tests assert both directions: real secrets fail, documented placeholders pass.
"""

from __future__ import annotations

from pathlib import Path

from scripts.check_commit_safety import MAX_FILE_BYTES, check_path, main

# Assembled at runtime so this test file never trips the scanner itself.
ANTHROPIC_KEY = "sk-ant-" + "api03-" + "a" * 24
AWS_KEY = "AKIA" + "1234567890ABCDEF"
PRIVATE_KEY_HEADER = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_clean_file_has_no_findings(tmp_path: Path) -> None:
    path = _write(tmp_path, "module.py", "API_KEY = os.environ['ANTHROPIC_API_KEY']\n")
    assert check_path(path) == []


def test_anthropic_key_is_flagged(tmp_path: Path) -> None:
    path = _write(tmp_path, "leak.py", f'KEY = "{ANTHROPIC_KEY}"\n')
    findings = check_path(path)
    assert len(findings) == 1
    assert "Anthropic API key" in findings[0]
    assert "leak.py:1" in findings[0]


def test_aws_key_is_flagged(tmp_path: Path) -> None:
    path = _write(tmp_path, "leak.txt", f"aws = {AWS_KEY}\n")
    assert any("AWS access key" in f for f in check_path(path))


def test_placeholder_line_is_allowed(tmp_path: Path) -> None:
    path = _write(tmp_path, "settings.py", 'JWT_SECRET = "change-me-in-production"\n')
    assert check_path(path) == []


def test_private_key_block_is_flagged(tmp_path: Path) -> None:
    path = _write(tmp_path, "id_rsa", f"{PRIVATE_KEY_HEADER}\nabc\n")
    assert any("private key block" in f for f in check_path(path))


def test_env_file_is_forbidden(tmp_path: Path) -> None:
    path = _write(tmp_path, ".env", "SAFE=1\n")
    assert any("environment file" in f for f in check_path(path))


def test_data_path_is_forbidden(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = _write(tmp_path, "data/raw/contract.pdf", "not really a pdf")
    monkeypatch.chdir(tmp_path)
    assert any("corpus, index, or result path" in f for f in check_path(path))


def test_large_file_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "big.bin"
    path.write_bytes(b"0" * (MAX_FILE_BYTES + 1))
    assert any("exceeds" in f for f in check_path(path))


def test_main_fails_on_findings(tmp_path: Path) -> None:
    path = _write(tmp_path, "leak.py", f'KEY = "{ANTHROPIC_KEY}"\n')
    assert main([str(path)]) == 1


def test_main_passes_on_clean_tree(tmp_path: Path) -> None:
    path = _write(tmp_path, "clean.py", "print('hello')\n")
    assert main([str(path)]) == 0
