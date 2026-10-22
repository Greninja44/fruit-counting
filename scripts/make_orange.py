#!/usr/bin/env python
"""Build an orange-only dataset (class 0 = orange) from the multi-class fruit data; every other fruit label is dropped.

Positives are images with at least one orange. A small share of images with *no* orange (and no look-alike citrus/round fruit)
are kept as empty-label negatives so the detector learns not to fire on apples, peaches and so on. The COCO val2017 test images
become the held-out ``test_coco`` split (COCO labels oranges exhaustively, so images without oranges are true zero-count tests).
Images are symlinked; only label files are written.

    python scripts/make_orange.py [--neg-ratio 0.15]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.utils.config import resolve_path  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("make_orange")
LOOKALIKE = ["mandarin", "grapefruit", "lemon", "lime", "melon", "peach", "apricot", "persimmon", "cantaloupe"]


def orange_rows(label_file: Path, cls_id: int) -> list[str]:
    rows = []
    for ln in label_file.read_text().splitlines():
        p = ln.split()
        if p and int(p[0]) == cls_id:
            rows.append(" ".join(["0"] + p[1:]))
    return rows


def build_split(src: Path, cls_id: int, names: list[str], split: str, out: Path, neg_ratio: float, rng, keep_all_negatives=False, meta=None):
    img_dir, lab_dir = out / "images" / split, out / "labels" / split
    for d in (img_dir, lab_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    pos, neg = [], []
    for f in sorted((src / "labels" / split).glob("*.txt")):
        rows = orange_rows(f, cls_id)
        (pos if rows else neg).append((f, rows))
    if meta is not None:                          # drop negatives that contain look-alike fruit
        bad = set(meta.loc[meta[[f"n_{c}" for c in LOOKALIKE if f"n_{c}" in meta]].fillna(0).sum(axis=1) > 0, "image"])
        neg = [(f, r) for f, r in neg if f.with_suffix(".jpg").name not in bad]
    if not keep_all_negatives:
        k = min(len(neg), int(round(neg_ratio * len(pos))))
        neg = [neg[i] for i in rng.choice(len(neg), k, replace=False)] if k else []
    for f, rows in pos + neg:
        jpg = next((src / "images" / split / (f.stem + e) for e in (".jpg", ".jpeg", ".png") if (src / "images" / split / (f.stem + e)).exists()), None)
        if jpg is None:
            continue
        (img_dir / jpg.name).symlink_to(jpg.resolve())
        (lab_dir / f.name).write_text("\n".join(rows) + ("\n" if rows else ""))
    n_boxes = sum(len(r) for _, r in pos)
    log.info("%s: %d orange images (%d boxes) + %d negatives", split, len(pos), n_boxes, len(neg))
    return len(pos), n_boxes, len(neg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/processed/openimages_fruit")
    ap.add_argument("--coco-test", default="data/processed/coco_fruit")
    ap.add_argument("--out", default="data/processed/orange")
    ap.add_argument("--neg-ratio", type=float, default=0.15)
    a = ap.parse_args()
    src, coco, out = resolve_path(a.src), resolve_path(a.coco_test), resolve_path(a.out)
    rng = np.random.default_rng(42)
    names = lambda p: {int(k): v for k, v in yaml.safe_load((p / "dataset.yaml").read_text())["names"].items()}  # noqa: E731
    oi_id = next(k for k, v in names(src).items() if v == "orange")
    co_id = next(k for k, v in names(coco).items() if v == "orange")
    meta = pd.read_csv(src / "meta.csv")
    for sp in ("train", "val", "test"):
        build_split(src, oi_id, [], sp, out, a.neg_ratio, rng, keep_all_negatives=(sp == "test"), meta=meta)
    build_split(coco, co_id, [], "test", out.parent / "_tmp_coco_orange", 0, rng, keep_all_negatives=True)
    tmp = out.parent / "_tmp_coco_orange"
    for kind in ("images", "labels"):
        dst = out / kind / "test_coco"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(tmp / kind / "test"), str(dst))
    shutil.rmtree(tmp)
    cm = pd.read_csv(coco / "meta.csv")
    cm = cm[cm.split == "test"].assign(split="test_coco")
    allm = pd.concat([meta, cm])
    keep = []
    for sp in ("train", "val", "test", "test_coco"):
        n = {f.name.rsplit(".", 1)[0] + ".jpg": sum(1 for ln in f.read_text().splitlines() if ln.strip()) for f in (out / "labels" / sp).glob("*.txt")}
        m = allm[(allm.split == sp) & allm.image.isin(n)].copy()
        m["n_fruit"] = m.image.map(n)          # = number of orange boxes
        keep.append(m[[c for c in m.columns if not c.startswith("n_") or c in ("n_fruit", "n_crowd_regions", "n_occluded", "n_truncated")]])
    pd.concat(keep).drop_duplicates("image").to_csv(out / "meta.csv", index=False)
    (out / "dataset.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: orange\n")
    cfg = yaml.safe_load((resolve_path("configs") / "default.yaml").read_text())
    cfg["classes"] = ["orange"]
    cfg["pretrained_class_map"] = {"orange": "orange"}
    cfg["evaluation"].update({"dataset_root": str(out.relative_to(resolve_path("."))), "split": "test"})
    (resolve_path("configs") / "orange.yaml").write_text("# Generated by scripts/make_orange.py\n" + yaml.safe_dump(cfg, sort_keys=False))
    log.info("done: %s ; config configs/orange.yaml", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
