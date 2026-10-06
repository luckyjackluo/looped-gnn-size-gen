"""Shared configuration and path utilities for all pipelines and scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import yaml


def load_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    """Return the UnifiedLearning repository root (parent of the ``unified_learning`` package)."""
    return Path(__file__).resolve().parents[2]


def resolve_paths(data_dir: Path, entries: Optional[Iterable[str]]) -> List[Path]:
    """Resolve a list of file-path strings to concrete ``Path`` objects.

    Resolution rules (applied per entry):
    - Absolute path  → returned as-is.
    - Multi-segment relative path (e.g. ``data/chipgen/.../foo.pt``) → resolved
      under the **repository root** so cross-directory references work.
    - Basename only (e.g. ``foo.pickle``) → resolved under ``data_dir``.
    """
    if not entries:
        return []
    root = repo_root()
    out: List[Path] = []
    for raw in entries:
        p = Path(raw)
        if p.is_absolute():
            out.append(p)
        elif len(p.parts) > 1:
            out.append((root / p).resolve())
        else:
            out.append(data_dir / p)
    return out
