"""Application settings loaded from environment variables (closes #33)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Clausewright service configuration."""

    environment: str = os.getenv("CLAUSEWRIGHT_ENV", "production")
    app_version: str = "0.1.0"
    database_path: str = os.getenv("DATABASE_PATH", "data/clausewright.db")
    jwt_secret: str = os.getenv("JWT_SECRET", "clausewright-insecure-secret-key-change-in-prod")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = int(os.getenv("JWT_EXPIRATION_MINUTES", "1440"))  # 24 hours
    max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))  # 25 MB
    default_max_llm_calls: int = int(os.getenv("MAX_LLM_CALLS", "2"))
    global_spend_cap_usd: float = float(os.getenv("GLOBAL_SPEND_CAP_USD", "100.0"))
    rate_limit_per_minute_ask: int = int(os.getenv("RATE_LIMIT_PER_MINUTE_ASK", "30"))
    rate_limit_per_minute_upload: int = int(os.getenv("RATE_LIMIT_PER_MINUTE_UPLOAD", "10"))

    @classmethod
    def load(cls) -> Settings:
        """Create settings instance reading current environment."""
        return cls()


def get_git_commit() -> str:
    """Determine current git commit SHA, with fallback to env or unknown."""
    env_sha = os.getenv("GIT_COMMIT")
    if env_sha:
        return env_sha.strip()

    try:
        import subprocess

        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
