#!/usr/bin/env python
"""Add LVIS fruit categories (boxes + instance polygons) on COCO images to an existing processed dataset.

Source: LVIS v1 (https://www.lvisdataset.org), annotations CC BY 4.0; the images are COCO images (Flickr, per-image licences).
Safeguards:
  * every image of COCO val2017 (our held-out test set) is excluded;
  * images already in the dataset (e.g. the COCO train fruit images) are skipped, so label sources are not mixed on one image.
LVIS is *federated*: a category is only guaranteed to be exhaustively labelled in images where it was verified, so some real
fruit are unlabelled (counted as false positives by detectors; mild label noise for training). Streaming parser: the 1 GB
annotation file is never fully loaded into memory.

    python scripts/prepare_lvis.py --lvis-json data/raw/lvis/lvis_v1_train.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import ijson
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.data.dataset import box_to_yolo  # noqa: E402
from fruitcount.utils.config import resolve_path  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("prepare_lvis")
# LVIS category name -> canonical class name (vegetable-like items such as tomato/cucumber/pepper/pumpkin are not included by default)
LVIS_FRUIT = {"apple": "apple", "apricot": "apricot", "avocado": "avocado", "banana": "banana", "blackberry": "blackberry",
              "blueberry": "blueberry", "cantaloup": "cantaloupe", "cherry": "cherry", "clementine": "mandarin", "coconut": "coconut",
              "date_(fruit)": "date", "fig_(fruit)": "fig", "grape": "grape", "kiwi_fruit": "kiwi", "lemon": "lemon", "lime": "lime",
              "mandarin_orange": "mandarin", "melon": "melon", "orange_(fruit)": "orange", "papaya": "papaya", "peach": "peach",
              "pear": "pear", "persimmon": "persimmon", "pineapple": "pineapple", "raspberry": "raspberry", "strawberry": "strawberry",
              "watermelon": "watermelon"}


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is not None:
                dest.write_bytes(data)
                return True
        except Exception:
            pass
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lvis-json", default="data/raw/lvis/lvis_v1_train.json")
    ap.add_argument("--out", default="data/processed/openimages_fruit")
    ap.add_argument("--config", default="configs/openimages.yaml", help="class list to extend (rewritten in place)")
    ap.add_argument("--coco-val-json", default="data/raw/coco/annotations/instances_val2017.json")
    ap.add_argument("--extra", nargs="*", default=[], help="additional LVIS category names to include (e.g. tomato)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    out, cfg_path = resolve_path(a.out), resolve_path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    classes: list[str] = list(cfg["classes"])
    cmap = dict(LVIS_FRUIT, **{e: e for e in a.extra})
    for canon in cmap.values():
        if canon not in classes:
            classes.append(canon)
    with open(resolve_path(a.lvis_json), "rb") as f:
        cats = {c["id"]: c["name"] for c in ijson.items(f, "categories.item", use_float=True)}
    cid2idx = {cid: classes.index(cmap[n]) for cid, n in cats.items() if n in cmap}
    log.info("%d LVIS categories selected -> %d classes total", len(cid2idx), len(classes))

    test_ids = {im["id"] for im in json.loads(resolve_path(a.coco_val_json).read_text())["images"]}
    meta_f = out / "meta.csv"
    have = set(pd.read_csv(meta_f).source_id) if meta_f.is_file() else set()
    anns: dict[int, list] = {}
    with open(resolve_path(a.lvis_json), "rb") as f:                     # pass 1: fruit annotations only
        for an in ijson.items(f, "annotations.item", use_float=True):
            if an["category_id"] in cid2idx:
                anns.setdefault(an["image_id"], []).append(an)
    n_before = len(anns)
    anns = {i: v for i, v in anns.items() if i not in test_ids and f"coco_{i}" not in have}
    log.info("%d images with fruit; %d after excluding COCO val2017 (test) and images already present", n_before, len(anns))
    images = {}
    with open(resolve_path(a.lvis_json), "rb") as f:                     # pass 2: image records for those ids
        for im in ijson.items(f, "images.item", use_float=True):
            if im["id"] in anns:
                images[im["id"]] = im
    for d in ("images", "labels", "labels_seg"):
        (out / d / "train").mkdir(parents=True, exist_ok=True)

    def job(iid):
        im = images[iid]
        name = f"{iid:012d}.jpg"
        dest = out / "images" / "train" / name
        return iid, (dest.is_file() or fetch(im["coco_url"], dest))

    with cf.ThreadPoolExecutor(a.workers) as ex:
        ok = dict(ex.map(job, sorted(anns)))
    rows = []
    for iid, good in ok.items():
        if not good:
            continue
        im = images[iid]
        w, h = im["width"], im["height"]
        det, seg, n = [], [], [0] * len(classes)
        for an in anns[iid]:
            x, y, bw, bh = an["bbox"]
            polys = [p for p in an.get("segmentation", []) if len(p) >= 6]
            if bw < 1 or bh < 1 or not polys:
                continue
            c = cid2idx[an["category_id"]]
            det.append(f"{c} {box_to_yolo((x, y, x + bw, y + bh), w, h)}")
            pts = np.clip(np.asarray(max(polys, key=len), float).reshape(-1, 2) / [w, h], 0, 1)
            seg.append(f"{c} " + " ".join(f"{v:.6f}" for v in pts.ravel()))
            n[c] += 1
        stem = f"{iid:012d}"
        (out / "labels" / "train" / f"{stem}.txt").write_text("\n".join(det) + ("\n" if det else ""))
        (out / "labels_seg" / "train" / f"{stem}.txt").write_text("\n".join(seg) + ("\n" if seg else ""))
        rows.append({"image": f"{stem}.jpg", "split": "train", "source_id": f"coco_{iid}", "width": w, "height": h, "has_crowd": 0,
                     "n_crowd_regions": 0, "n_fruit": sum(n), "license": "COCO image (Flickr, per-image) / LVIS CC BY 4.0",
                     **{f"n_{c}": k for c, k in zip(classes, n)}})
    meta = pd.DataFrame(rows)
    if meta_f.is_file():
        meta = pd.concat([pd.read_csv(meta_f), meta])
    for c in classes:
        col = f"n_{c}"
        meta[col] = meta[col].fillna(0).astype(int) if col in meta else 0
    meta.sort_values("image").to_csv(meta_f, index=False)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    (out / "dataset.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    cfg["classes"] = classes
    cfg_path.write_text("# Generated by scripts/prepare_openimages.py / prepare_lvis.py\n" + yaml.safe_dump(cfg, sort_keys=False))
    log.info("added %d images (%d download failures); %d classes now in %s", len(rows), sum(not v for v in ok.values()), len(classes), cfg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
