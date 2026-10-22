"""Dataset integrity checks and summary statistics (CLI: ``python -m fruitcount.data.validate_dataset``)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fruitcount.data.dataset import SPLITS, AnnotationError, FruitDataset
from fruitcount.utils.config import load_config, resolve_path
from fruitcount.utils.log import get_logger

log = get_logger("validate")


def validate_split(ds: FruitDataset, check_images: bool = True) -> dict:
    """Check one split. Returns issue lists plus statistics."""
    issues: list[str] = []
    per_class: Counter = Counter()
    sizes: Counter = Counter()
    counts, areas_rel = [], []
    if check_images:
        for p in ds.image_paths:
            if cv2.imread(str(p)) is None:
                issues.append(f"{p.name}: unreadable/corrupt image")
    for i, p in enumerate(ds.image_paths):
        lab = ds.root / "labels" / ds.split / f"{p.stem}.txt"
        if not lab.is_file():
            issues.append(f"{p.name}: missing label file (treated as zero fruits)")
        try:
            s = ds.get(i)
        except (AnnotationError, FileNotFoundError) as e:
            issues.append(f"{p.name}: {e}")
            continue
        sizes[(s.width, s.height)] += 1
        counts.append(s.count())
        seen = set()
        for g in s.instances:
            per_class[ds.classes[g.cls]] += 1
            w, h = g.box[2] - g.box[0], g.box[3] - g.box[1]
            if w < 1 or h < 1:
                issues.append(f"{p.name}: degenerate box ({w:.1f}x{h:.1f}px)")
            areas_rel.append(g.area / (s.width * s.height))
            key = tuple(round(v, 1) for v in g.box) + (g.cls,)
            if key in seen:
                issues.append(f"{p.name}: duplicate annotation {key}")
            seen.add(key)
    orphan = [f.name for f in (ds.root / "labels" / ds.split).glob("*.txt") if not any(q.stem == f.stem for q in ds.image_paths)] \
        if (ds.root / "labels" / ds.split).is_dir() else []
    issues += [f"{n}: label without image" for n in orphan]
    ar = np.asarray(areas_rel) if areas_rel else np.zeros(1)
    return {
        "split": ds.split,
        "images": len(ds),
        "instances": int(sum(counts)),
        "class_distribution": dict(per_class),
        "image_sizes_top5": [{"size": f"{w}x{h}", "n": n} for (w, h), n in sizes.most_common(5)],
        "fruit_per_image": {
            "mean": float(np.mean(counts)) if counts else 0.0,
            "max": int(max(counts)) if counts else 0,
            "zero_images": int(sum(c == 0 for c in counts)),
        },
        "relative_box_area": {k: float(v) for k, v in zip(["min", "p10", "median", "p90", "max"], np.percentile(ar, [0, 10, 50, 90, 100]))},
        "issues": issues,
    }


def check_leakage(root: Path) -> list[str]:
    """Flag any ``source_id`` (video / sequence / photo group) present in more than one split."""
    meta_file = root / "meta.csv"
    if not meta_file.is_file():
        return ["meta.csv missing: cannot check train/test leakage"]
    meta = pd.read_csv(meta_file)
    if "source_id" not in meta or "split" not in meta:
        return ["meta.csv lacks source_id/split columns: cannot check leakage"]
    g = meta.groupby("source_id")["split"].nunique()
    return [f"source_id {sid} appears in multiple splits" for sid in g[g > 1].index]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None, help="dataset root (default: evaluation.dataset_root)")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    ap.add_argument("--skip-image-decode", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = resolve_path(a.root or cfg["evaluation"]["dataset_root"])
    summary = {"root": str(root), "splits": [], "leakage": check_leakage(root)}
    for sp in SPLITS:
        if not (root / "images" / sp).is_dir():
            continue
        r = validate_split(FruitDataset(root, sp, cfg["classes"]), not a.skip_image_decode)
        summary["splits"].append(r)
        log.info("%s: %d images, %d instances, %d issues", sp, r["images"], r["instances"], len(r["issues"]))
    for msg in summary["leakage"]:
        log.warning(msg)
    print(json.dumps({k: v for k, v in summary.items()}, indent=2, default=str)[:6000])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    bad = sum(len(s["issues"]) for s in summary["splits"]) + len(summary["leakage"])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
