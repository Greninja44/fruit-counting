#!/usr/bin/env python
"""Build the apple/orange/banana subset of COCO 2017 in YOLO detection + segmentation format.

Source: COCO 2017 (https://cocodataset.org). Annotations are CC BY 4.0; images come from Flickr
under various licences (the licence id and Flickr URL of each image are kept in meta.csv).
Only the images that contain at least one of the configured classes are downloaded.

    python scripts/prepare_data.py --coco-split val2017 --target-split test
    python scripts/prepare_data.py --coco-split train2017 --max-images 1500   # -> train/val (seeded)

Crowd regions (``iscrowd=1``) cannot be counted per instance. Images containing them are kept
but flagged ``has_crowd=1`` in meta.csv and excluded from counting metrics by default.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.data.dataset import box_to_yolo, split_by_group  # noqa: E402
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("prepare_data")
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
                continue
            dest.write_bytes(data)
            return True
        except Exception as e:  # network errors are retried then reported
            err = e
    log.warning("download failed %s (%s)", url, err)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--coco-split", choices=["train2017", "val2017"], default="val2017")
    ap.add_argument("--target-split", choices=["train", "val", "test", "auto"], default=None,
                    help="default: test for val2017, 'auto' (seeded train/val 85/15) for train2017")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all matching images")
    ap.add_argument("--raw-dir", default="data/raw/coco")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    cfg = load_config(a.config)
    classes: list[str] = cfg["classes"]
    out = resolve_path(a.out or cfg["evaluation"]["dataset_root"])
    raw = resolve_path(a.raw_dir)
    target = a.target_split or ("test" if a.coco_split == "val2017" else "auto")

    ann_file = raw / "annotations" / f"instances_{a.coco_split}.json"
    if not ann_file.is_file():
        raw.mkdir(parents=True, exist_ok=True)
        zpath = raw / "annotations_trainval2017.zip"
        if not zpath.is_file():
            log.info("downloading COCO annotations (~250 MB)")
            urllib.request.urlretrieve(ANN_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extract(f"annotations/instances_{a.coco_split}.json", raw)
    coco = json.loads(ann_file.read_text())
    cat_ids = {c["id"]: c["name"] for c in coco["categories"]}
    missing = [c for c in classes if c not in cat_ids.values()]
    if len(missing) == len(classes):
        log.error("none of %s exist in COCO; add your own data instead (docs/dataset_setup.md)", classes)
        return 2
    if missing:
        log.warning("classes not in COCO (left unlabelled in COCO images): %s", missing)
    cid2idx = {cid: classes.index(n) for cid, n in cat_ids.items() if n in classes}

    anns: dict[int, list] = {}
    for an in coco["annotations"]:
        if an["category_id"] in cid2idx:
            anns.setdefault(an["image_id"], []).append(an)
    images = {im["id"]: im for im in coco["images"]}
    ids = sorted(anns)
    if a.max_images:
        import random

        random.Random(cfg["seed"]).shuffle(ids)
        ids = sorted(ids[: a.max_images])
    log.info("%d images with %s in %s", len(ids), classes, a.coco_split)

    if target == "auto":
        gmap = split_by_group([str(i) for i in ids], (0.85, 0.15, 0.0), cfg["seed"])
        split_of = {i: gmap[str(i)] for i in ids}
    else:
        split_of = {i: target for i in ids}
    lic = {l["id"]: l["name"] for l in coco["licenses"]}

    for sp in set(split_of.values()):
        for d in ("images", "labels", "labels_seg"):
            (out / d / sp).mkdir(parents=True, exist_ok=True)

    def job(iid: int):
        im, sp = images[iid], split_of[iid]
        dest = out / "images" / sp / im["file_name"]
        ok = dest.is_file() or fetch(im["coco_url"], dest)
        return iid, ok

    with cf.ThreadPoolExecutor(a.workers) as ex:
        ok = dict(ex.map(job, ids))

    rows = []
    for iid in ids:
        if not ok[iid]:
            continue
        im, sp = images[iid], split_of[iid]
        w, h = im["width"], im["height"]
        det, seg, nclass, crowd = [], [], [0] * len(classes), 0
        for an in anns[iid]:
            c = cid2idx[an["category_id"]]
            if an.get("iscrowd"):
                crowd += 1
                continue
            x, y, bw, bh = an["bbox"]
            if bw < 1 or bh < 1:
                continue
            det.append(f"{c} {box_to_yolo((x, y, x + bw, y + bh), w, h)}")
            polys = an["segmentation"]
            if polys:  # use the largest polygon of a multi-part mask
                poly = max(polys, key=len)
                if len(poly) >= 6:
                    pts = np.asarray(poly, float).reshape(-1, 2) / [w, h]
                    seg.append(f"{c} " + " ".join(f"{v:.6f}" for v in np.clip(pts, 0, 1).ravel()))
                else:
                    seg.append(det[-1])  # box-only fallback keeps line alignment
            else:
                seg.append(det[-1])
            nclass[c] += 1
        stem = Path(im["file_name"]).stem
        (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(det) + ("\n" if det else ""))
        (out / "labels_seg" / sp / f"{stem}.txt").write_text("\n".join(seg) + ("\n" if seg else ""))
        rows.append({"image": im["file_name"], "split": sp, "source_id": f"coco_{iid}", "coco_id": iid,
                     "width": w, "height": h, "has_crowd": int(crowd > 0), "n_crowd_regions": crowd,
                     "n_fruit": sum(nclass), **{f"n_{c}": n for c, n in zip(classes, nclass)},
                     "license": lic.get(im.get("license"), ""), "flickr_url": im.get("flickr_url", "")})
    new = pd.DataFrame(rows)
    meta_f = out / "meta.csv"
    if meta_f.is_file():
        old = pd.read_csv(meta_f)
        new = pd.concat([old[~old.image.isin(new.image)], new])
    new.sort_values("image").to_csv(meta_f, index=False)

    # Dataset YAMLs for Ultralytics training (detection and segmentation roots).
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    (out / "dataset.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    seg_root = out.parent / (out.name + "_seg")
    for sp in set(split_of.values()):
        (seg_root / "images").mkdir(parents=True, exist_ok=True)
        (seg_root / "labels").mkdir(parents=True, exist_ok=True)
        for link, tgt in ((seg_root / "images" / sp, out / "images" / sp), (seg_root / "labels" / sp, out / "labels_seg" / sp)):
            if not link.exists():
                link.symlink_to(tgt, target_is_directory=True)
    (seg_root / "dataset.yaml").write_text(f"path: {seg_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    log.info("wrote %d images to %s (failed downloads: %d)", len(new), out, sum(not v for v in ok.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
