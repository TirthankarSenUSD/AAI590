"""Project-root discovery. Every path in the pipeline is resolved from the root."""
from __future__ import annotations

from pathlib import Path

ROOT_MARKERS = ("config.yaml", "CLAUDE.md")


def find_root(start: str | Path | None = None) -> Path:
    """Walk up from `start` (default: this file) to the folder holding the root markers."""
    here = Path(start or __file__).resolve()
    for folder in [here, *here.parents]:
        if all((folder / m).is_file() for m in ROOT_MARKERS):
            return folder
    raise FileNotFoundError(f"No project root (with {ROOT_MARKERS}) above {here}")
