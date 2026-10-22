#!/usr/bin/env python
"""Count fruit in one image or a directory of images and save annotated results.

    python scripts/predict_image.py --source path/to/img_or_dir --approach detect
    python scripts/predict_image.py --source img.jpg --approach segment --conf 0.3 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.counting.counter import count_detections  # noqa: E402
from fruitcount.detection.base import ModelLoadError  # noqa: E402
from fruitcount.detection.factory import build_model  # noqa: E402
from fruitcount.detection.predict import run_model  # noqa: E402
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402
from fruitcount.utils.io import InvalidInputError, list_images, read_image  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402
from fruitcount.visualization.draw_results import counts_header, draw_detections  # noqa: E402

log = get_logger("predict_image")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image file or directory")
    ap.add_argument("--approach", choices=["detect", "segment"], default="detect")
    ap.add_argument("--weights", default=None, help="override weights from config (e.g. fine-tuned best.pt)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--preprocess", choices=["none", "clahe", "gamma", "tile"], default=None)
    ap.add_argument("--out", default=None, help="output directory (default: outputs/visualizations)")
    a = ap.parse_args()

    cfg = load_config(a.config, {"inference.conf": a.conf, "inference.iou": a.iou, "inference.imgsz": a.imgsz,
                                 "device": a.device, "inference.preprocess": a.preprocess})
    inf = cfg["inference"]
    out_dir = resolve_path(a.out or cfg["paths"]["visualizations"])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = build_model(cfg, a.approach, a.weights)
        images = list_images(a.source)
    except (ModelLoadError, InvalidInputError, RuntimeError) as e:
        log.error("%s", e)
        return 2
    if not images:
        log.error("no images found in %s", a.source)
        return 2

    rows, bad = [], 0
    for p in images:
        try:
            img = read_image(p)
        except InvalidInputError as e:
            log.warning("skipping: %s", e)
            bad += 1
            continue
        t0 = time.perf_counter()
        dets = run_model(model, img, inf["conf"], inf["iou"], inf["imgsz"], inf["preprocess"], inf["max_det"])
        ms = (time.perf_counter() - t0) * 1e3
        res = count_detections(dets, cfg["classes"], inf["conf"])
        kept = [d for d in dets if d.conf >= inf["conf"]]
        vis = draw_detections(img, kept, counts_header(res.total, res.per_class), index_instances=model.task == "segment")
        dest = out_dir / f"{p.stem}_{a.approach}.jpg"
        cv2.imwrite(str(dest), vis)
        rows.append({"image": p.name, "approach": a.approach, "total": res.total, **res.per_class, "ms": round(ms, 1)})
        log.info("%s: %d fruits %s (%.0f ms) -> %s", p.name, res.total, res.per_class, ms, dest)
    if rows:
        with (out_dir / f"counts_{a.approach}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if bad:
        log.warning("%d invalid files skipped", bad)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
