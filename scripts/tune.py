#!/usr/bin/env python
"""Select inference settings (model size, image size, NMS IoU, confidence) on a *tuning* split.

Settings are chosen on data that is NOT the final test set, then confirmed once on the test set with
scripts/evaluate.py (see the printed command). Objective = counting MAE (ties -> F1). No training is done.

    python scripts/tune.py --root data/processed/openimages_fruit --split val \
        --weights models/yolo11n.pt models/yolo11s.pt models/yolo11m.pt --imgsz 640 960 --nms-iou 0.5 0.7

Only the classes the models can predict are scored (the others are dropped from GT), and crowd/group images are excluded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.data.dataset import FruitDataset  # noqa: E402
from fruitcount.detection.base import ModelLoadError  # noqa: E402
from fruitcount.detection.factory import build_model  # noqa: E402
from fruitcount.evaluation.predictions import load_predictions, save_predictions  # noqa: E402
from fruitcount.evaluation.runner import evaluate_run, predict_dataset, restrict_samples, usable_samples  # noqa: E402
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("tune")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/openimages.yaml", help="class list of the tuning dataset")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--approach", choices=["detect", "segment"], default="detect")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640])
    ap.add_argument("--nms-iou", nargs="+", type=float, default=[0.7])
    ap.add_argument("--conf-grid", nargs="+", type=float, default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    ap.add_argument("--max-images", type=int, default=0, help="subsample the tuning images (seeded); 0 = all")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config, {"device": a.device})
    classes = cfg["classes"]
    root = resolve_path(a.root or cfg["evaluation"]["dataset_root"])
    try:
        ds = FruitDataset(root, a.split, classes, load_polygons=False)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2
    samples, _ = usable_samples(ds)
    first_model = build_model(cfg, a.approach, a.weights[0])
    known = sorted(first_model._id_map.values())
    samples = restrict_samples(samples, known)
    if a.max_images and len(samples) > a.max_images:
        idx = np.random.default_rng(cfg["seed"]).choice(len(samples), a.max_images, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    log.info("tuning on %s/%s: %d images, %d fruits, classes scored: %s", root.name, a.split, len(samples),
             sum(s.count() for s in samples), [classes[k] for k in known])
    cache = resolve_path("outputs/tuning")
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in a.weights:
        try:
            model = build_model(cfg, a.approach, w)
        except ModelLoadError as e:
            log.error("%s", e)
            return 2
        for sz in a.imgsz:
            for nms in a.nms_iou:
                f = cache / f"{root.name}_{a.split}_{Path(w).stem}_{sz}_{nms}.json"
                if f.is_file():
                    info, recs = load_predictions(f)
                else:
                    log.info("predicting %s imgsz=%d nms_iou=%.2f", Path(w).name, sz, nms)
                    records, extra = predict_dataset(model, samples, cfg["inference"]["eval_conf"], nms, sz, timing_conf=0.25, timing_images=100)
                    info = {"weights": w, "imgsz": sz, "nms_iou": nms, **extra,
                            "ms": float(np.mean([r.ms for r in records[:100]]))}
                    save_predictions(f, records, info)
                    recs = {r.image: r for r in records}
                for conf in a.conf_grid:
                    ev = evaluate_run(samples, recs, classes, conf)
                    c, d = ev["counting"], ev["detection"]
                    rows.append({"weights": Path(w).name, "imgsz": sz, "nms_iou": nms, "conf": conf, "MAE": c["MAE"], "RMSE": c["RMSE"],
                                 "exact": c["exact_count_accuracy"], "signed_err": c["mean_signed_error"], "precision": d["precision"],
                                 "recall": d["recall"], "F1": d["F1"], "mAP50": d["mAP50"], "ms_per_image": info["ms"]})
                log.info("  done %s %d %.2f", Path(w).name, sz, nms)
    df = pd.DataFrame(rows)
    out = resolve_path("outputs/reports") / f"tuning_{root.name}_{a.split}_{a.approach}.csv"
    df.to_csv(out, index=False)
    best = df.sort_values(["MAE", "F1"], ascending=[True, False]).groupby(["weights", "imgsz", "nms_iou"], as_index=False).first()
    print(best.sort_values("MAE").round(3).to_string(index=False))
    top = df.sort_values(["MAE", "F1"], ascending=[True, False]).iloc[0]
    print(f"\nbest on tuning set: {top.weights} imgsz={int(top.imgsz)} nms_iou={top.nms_iou} conf={top.conf} (MAE {top.MAE:.3f}, F1 {top.F1:.3f}, {top.ms_per_image:.0f} ms)")
    print(f"confirm once on the test set:\n  python scripts/evaluate.py --approaches {a.approach} --weights models/{top.weights} "
          f"--imgsz {int(top.imgsz)} --iou {top.nms_iou} --conf {top.conf} --tag tuned")
    log.info("all combinations: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
