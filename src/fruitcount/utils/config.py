"""YAML configuration loading with dotted-key overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config; ``overrides`` maps dotted keys (``inference.conf``) to values."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
