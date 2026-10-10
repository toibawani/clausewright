from pathlib import Path

from clausewright.config import PATHS, PROJECT_ROOT, Paths


def test_project_root_contains_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_manifest_path_points_to_existing_file() -> None:
    assert PATHS.manifest.is_file()


def test_paths_are_derived_from_root() -> None:
    p = Paths(root=Path("/tmp/example"))
    assert p.raw_dir == Path("/tmp/example/data/raw")
    assert p.index_dir == Path("/tmp/example/data/index")
    assert p.gold_set == Path("/tmp/example/data/eval/gold.jsonl")
