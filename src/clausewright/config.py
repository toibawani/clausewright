"""Central configuration: filesystem paths and model names.

Retrieval thresholds are intentionally absent. They must be calibrated on the
gold set (milestone 7), not guessed here.
"""

from dataclasses import dataclass
from pathlib import Path

# src/clausewright/config.py -> project root is two directories above the package.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Locations of data on disk, all derived from a single root."""

    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def index_dir(self) -> Path:
        return self.root / "data" / "index"

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "manifest.csv"

    @property
    def gold_set(self) -> Path:
        return self.root / "data" / "eval" / "gold.jsonl"


# Embedding model chosen in the blueprint; its output dimension is verified in milestone 5.
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

PATHS: Paths = Paths(root=PROJECT_ROOT)
