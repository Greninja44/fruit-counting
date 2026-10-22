#!/usr/bin/env python
"""Optional sanity check: compare our box AP with pycocotools on the same cached predictions.

    pip install pycocotools
    python scripts/crosscheck_coco_eval.py --run detect_none

Small differences (< ~0.002) are expected (COCOeval uses area ranges / maxDets=100 and its own tie-breaking).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.data.dataset import FruitDataset  # noqa: E402
from fruitcount.evaluation.matching import build_match_data  # noqa: E402
from fruitcount.evaluation.metrics import detection_metrics  # noqa: E402
from fruitcount.evaluation.predictions import load_predictions  # noqa: E402
from fruitcount.evaluation.runner import usable_samples  # noqa: E402
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="detect_none")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools not installed: pip install pycocotools")
        return 2
    cfg = load_config(a.config)
    cl = cfg["classes"]
    ds = FruitDataset(resolve_path(cfg["evaluation"]["dataset_root"]), cfg["evaluation"]["split"], cl)
    samples, _ = usable_samples(ds)
    _, recs = load_predictions(resolve_path(cfg["paths"]["predictions"]) / f"{a.run}.json")
    images, anns, dets, aid = [], [], [], 1
    for i, s in enumerate(samples, 1):
        images.append({"id": i, "width": s.width, "height": s.height})
        for g in s.instances:
            x1, y1, x2, y2 = g.box
            anns.append({"id": aid, "image_id": i, "category_id": g.cls + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            aid += 1
        for d in recs[s.image_path.name].dets:
            x1, y1, x2, y2 = d.box
            dets.append({"image_id": i, "category_id": d.cls + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d.conf})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": k + 1, "name": c} for k, c in enumerate(cl)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        E = COCOeval(gt, gt.loadRes(dets), "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    ours = detection_metrics([build_match_data(s, recs[s.image_path.name].dets) for s in samples], cl, cfg["inference"]["conf"])
    print(f"pycocotools  AP50-95 {E.stats[0]:.4f}  AP50 {E.stats[1]:.4f}")
    print(f"this project AP50-95 {ours['mAP50_95']:.4f}  AP50 {ours['mAP50']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
