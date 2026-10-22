#!/usr/bin/env python
"""Compute heuristic condition tags and write a manual-annotation template.

    python scripts/tag_conditions.py                    # writes auto tags + template
Edit data/annotations/condition_tags.csv (1/0 per tag; empty = keep auto) to override. See docs/evaluation.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.data.dataset import FruitDataset  # noqa: E402
from fruitcount.evaluation.tags import TAGS, auto_tags, image_stats  # noqa: E402
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402
from fruitcount.utils.io import read_image  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("tag_conditions")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    split = a.split or cfg["evaluation"]["split"]
    root = resolve_path(cfg["evaluation"]["dataset_root"])
    ds = FruitDataset(root, split, cfg["classes"])
    rows = []
    for s in ds:
        img = read_image(s.image_path)
        rows.append({"image": s.image_path.name, **auto_tags(img, s), **image_stats(img)})
    df = pd.DataFrame(rows)
    ann = resolve_path("data/annotations")
    ann.mkdir(parents=True, exist_ok=True)
    df.to_csv(ann / f"condition_tags_auto_{root.name}_{split}.csv", index=False)
    tpl = ann / "condition_tags.csv"
    if not tpl.exists():
        t = pd.DataFrame({"image": df.image, **{k: "" for k in TAGS}})
        t.to_csv(tpl, index=False)
        log.info("wrote manual template %s (fill 1/0; empty keeps the auto value)", tpl)
    summary = df[TAGS].sum().rename("images_tagged")
    log.info("auto tag counts over %d images:\n%s", len(df), summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
