"""Workspace-owned downloads; model binaries remain outside Git."""
from pathlib import Path


def model_directory(workspace: str | Path | None = None) -> Path:
    directory = (Path(workspace) if workspace is not None else Path.cwd()) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()
