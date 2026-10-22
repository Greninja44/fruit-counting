#!/usr/bin/env python
"""Paired bootstrap of the per-image absolute count error difference between two evaluated runs.

    python scripts/compare_runs.py detect_none segment_none
Positive mean = the second run has the lower error. If the 95% CI contains 0 the difference is not distinguishable from noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    rep = resolve_path(load_config()["paths"]["reports"])
    a, b = (pd.read_csv(rep / f"per_image_{r}.csv").set_index("image") for r in sys.argv[1:3])
    diff = (a.abs_error - b.loc[a.index].abs_error).to_numpy()
    rng = np.random.default_rng(0)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"mean |err|({sys.argv[1]}) - |err|({sys.argv[2]}) = {diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] over {len(diff)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
