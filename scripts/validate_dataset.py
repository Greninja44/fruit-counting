#!/usr/bin/env python
"""Validate dataset integrity and print statistics (thin wrapper around fruitcount.data.validate_dataset)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.data.validate_dataset import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
