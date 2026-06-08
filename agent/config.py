"""Configuration helpers for ChulkHarness."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    project_root: Path
    skills_dir: Path
    store_path: Path
    traces_dir: Path
    model: str
    max_tool_calls_per_turn: int = 5
    shell_timeout_seconds: int = 10


def load_config() -> Config:
    """Load local development configuration."""
    default_root = Path(__file__).resolve().parent.parent
    project_root = Path(os.getenv("CHULK_PROJECT_ROOT", default_root)).resolve()

    return Config(
        project_root=project_root,
        skills_dir=project_root / "agent" / "skills",
        store_path=project_root / "agent" / "store.sqlite",
        traces_dir=project_root / "traces",
        model=os.getenv("CHULK_MODEL", "gpt-4.1-mini"),
    )
