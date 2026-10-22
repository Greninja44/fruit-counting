#!/usr/bin/env python
"""Build a large multi-fruit detection dataset from Open Images (boxes, official train/val/test splits).

Source: Open Images bounding boxes (train: V6 file; validation/test: V5 files, the versions served publicly without login) (https://storage.googleapis.com/openimages/web/index.html). Annotations are
CC BY 4.0; the images are listed by Google as CC BY 2.0 (each image's own licence/author is in its Open Images
metadata: check before redistribution). Nothing is committed: data lands in git-ignored ``data/``.

    python scripts/prepare_openimages.py --max-train-per-class 2500          # ~15 fruit classes
    python scripts/prepare_openimages.py --classes Apple Mango --splits validation test   # small trial

Handling of Open Images flags: ``IsGroupOf`` boxes (one box around many fruit) are NOT instances: they are left out
of the labels and the image is flagged ``has_crowd=1`` (excluded from counting metrics, like COCO crowd). Depiction
images (drawings/toys, ``IsDepiction``) are dropped. ``IsOccluded`` / ``IsTruncated`` counts are kept in meta.csv
(``n_occluded``, ``n_truncated``) and can feed the manual ``occluded`` condition tag.
Detection labels only: Open Images has no instance masks for most fruit, so segmentation models cannot be trained/evaluated here.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.utils.config import resolve_path  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("prepare_openimages")
BASE = "https://storage.googleapis.com/openimages"
IMG = "https://open-images-dataset.s3.amazonaws.com"
BBOX_URL = {"train": f"{BASE}/v6/oidv6-train-annotations-bbox.csv",
            "validation": f"{BASE}/v5/validation-annotations-bbox.csv",
            "test": f"{BASE}/v5/test-annotations-bbox.csv"}
CLASS_URL = f"{BASE}/v7/oidv7-class-descriptions-boxable.csv"
OUR_SPLIT = {"train": "train", "validation": "val", "test": "test"}
# Open Images display name -> canonical lowercase class name used in our labels
DEFAULT_CLASSES = {"Apple": "apple", "Banana": "banana", "Orange (fruit)": "orange", "Lemon (plant)": "lemon", "Mango": "mango",
                   "Grape": "grape", "Strawberry": "strawberry", "Pineapple": "pineapple", "Pear": "pear", "Peach": "peach",
                   "Pomegranate": "pomegranate", "Grapefruit": "grapefruit", "Watermelon": "watermelon",
                   "Cantaloupe": "cantaloupe", "Common fig": "fig"}


def download(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def fetch_image(url: str, dest: Path, retries: int = 3) -> tuple[int, int] | None:
    if dest.is_file():
        im = cv2.imread(str(dest))
        if im is not None:
            return im.shape[1], im.shape[0]
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            dest.write_bytes(data)
            return im.shape[1], im.shape[0]
        except Exception:
            pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES), help="Open Images class display names")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"], choices=list(BBOX_URL))
    ap.add_argument("--max-train-per-class", type=int, default=2500, help="images sampled per class in train (0 = all)")
    ap.add_argument("--max-eval-per-class", type=int, default=0, help="images per class in validation/test (0 = all)")
    ap.add_argument("--raw-dir", default="data/raw/openimages")
    ap.add_argument("--out", default="data/processed/openimages_fruit")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    raw, out = resolve_path(a.raw_dir), resolve_path(a.out)
    cls_df = pd.read_csv(download(CLASS_URL, raw / "classes.csv"), header=None, names=["mid", "name"])
    name2mid = dict(zip(cls_df.name, cls_df.mid))
    unknown = [c for c in a.classes if c not in name2mid]
    if unknown:
        log.error("unknown Open Images classes: %s", unknown)
        return 2
    names = [DEFAULT_CLASSES.get(c, c.lower().split(" (")[0]) for c in a.classes]
    mid2idx = {name2mid[c]: i for i, c in enumerate(a.classes)}
    rng = np.random.default_rng(a.seed)
    all_rows = []

    for sp in a.splits:
        csv = download(BBOX_URL[sp], raw / f"bbox_{sp}.csv")
        usecols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction"]
        parts = [c[c.LabelName.isin(mid2idx)] for c in pd.read_csv(csv, usecols=usecols, chunksize=2_000_000)]
        df = pd.concat(parts)
        df = df[df.IsDepiction != 1]
        depicted = set(pd.concat(parts)[lambda d: d.IsDepiction == 1].ImageID) - set(df.ImageID)
        cap = a.max_train_per_class if sp == "train" else a.max_eval_per_class
        keep: set[str] = set()
        for mid in mid2idx:
            ids = np.array(sorted(set(df[df.LabelName == mid].ImageID)))
            if cap and len(ids) > cap:
                ids = rng.choice(ids, cap, replace=False)
            keep |= set(ids)
            log.info("%s %-14s %6d images selected", sp, a.classes[mid2idx[mid]], len(ids))
        df = df[df.ImageID.isin(keep)]
        ids = sorted(keep)
        split = OUR_SPLIT[sp]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        log.info("%s: downloading %d images (%d depictions dropped)", sp, len(ids), len(depicted))

        def job(i):
            return i, fetch_image(f"{IMG}/{sp}/{i}.jpg", out / "images" / split / f"{i}.jpg")

        with cf.ThreadPoolExecutor(a.workers) as ex:
            sizes = dict(ex.map(job, ids))
        failed = [i for i, s in sizes.items() if s is None]
        log.info("%s: %d ok, %d failed", sp, len(ids) - len(failed), len(failed))
        for i, g in df.groupby("ImageID"):
            if sizes.get(i) is None:
                continue
            w, h = sizes[i]
            inst, crowd = g[g.IsGroupOf != 1], int((g.IsGroupOf == 1).sum())
            lines = []
            for r in inst.itertuples():
                x1, x2, y1, y2 = r.XMin, r.XMax, r.YMin, r.YMax
                if (x2 - x1) * w < 1 or (y2 - y1) * h < 1:
                    continue
                lines.append(f"{mid2idx[r.LabelName]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            (out / "labels" / split / f"{i}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            row = {"image": f"{i}.jpg", "split": split, "source_id": f"oi_{i}", "width": w, "height": h, "has_crowd": int(crowd > 0),
                   "n_crowd_regions": crowd, "n_fruit": len(lines), "n_occluded": int((inst.IsOccluded == 1).sum()),
                   "n_truncated": int((inst.IsTruncated == 1).sum()), "license": "CC BY 2.0 (image) / CC BY 4.0 (annotations)"}
            for k, n in enumerate(names):
                row[f"n_{n}"] = int((inst.LabelName.map(mid2idx) == k).sum())
            all_rows.append(row)
        for i in failed:                               # never leave an image without a label file
            (out / "images" / split / f"{i}.jpg").unlink(missing_ok=True)

    meta = pd.DataFrame(all_rows).sort_values("image")
    meta_f = out / "meta.csv"
    if meta_f.is_file():
        old = pd.read_csv(meta_f)
        meta = pd.concat([old[~old.image.isin(meta.image)], meta]).sort_values("image")
    meta.to_csv(meta_f, index=False)
    nm = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (out / "dataset.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{nm}\n")
    cfg = Path(resolve_path("configs")) / "openimages.yaml"
    base = (resolve_path("configs") / "default.yaml").read_text()
    import yaml

    c = yaml.safe_load(base)
    c["classes"] = names
    c["pretrained_class_map"] = {k: k for k in ("apple", "orange", "banana")}
    c["evaluation"]["dataset_root"] = str(out.relative_to(resolve_path(".")))
    cfg.write_text("# Generated by scripts/prepare_openimages.py\n" + yaml.safe_dump(c, sort_keys=False))
    log.info("done: %d images -> %s ; config: %s", len(meta), out, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
