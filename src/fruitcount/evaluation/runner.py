"""Prediction pass and evaluation pass over a dataset split (model-agnostic via the FruitModel protocol)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from fruitcount.data.dataset import FruitDataset, Sample
from fruitcount.detection.base import FruitModel
from fruitcount.detection.predict import run_model
from fruitcount.evaluation import metrics as M
from fruitcount.evaluation.matching import build_match_data
from fruitcount.evaluation.predictions import PredRecord
from fruitcount.utils.io import read_image
from fruitcount.utils.log import get_logger

log = get_logger(__name__)
EVAL_MAX_DET = 100   # COCO standard; no test image has anywhere near this many fruits


def usable_samples(ds: FruitDataset, exclude_crowd: bool = True) -> tuple[list[Sample], int]:
    """GT samples used for evaluation. Images with COCO crowd regions cannot be counted per instance."""
    keep, skipped = [], 0
    for s in ds:
        if exclude_crowd and int(s.meta.get("has_crowd", 0)):
            skipped += 1
            continue
        keep.append(s)
    return keep, skipped


def predict_dataset(model: FruitModel, samples: Sequence[Sample], conf: float, iou: float, imgsz: int,
                    preprocess: str = "none", warmup: int = 3, timing_conf: float | None = None,
                    timing_images: int | None = None) -> tuple[list[PredRecord], dict]:
    """Run the model on every sample at a *low* confidence so AP curves can be computed later.

    Latency is *not* taken from that pass (at conf 0.001 a segmentation model post-processes hundreds of
    masks per image, which is not what a user pays). A second pass at ``timing_conf`` (the operating
    threshold) measures end-to-end latency per image (preprocess + inference + post-processing, excluding
    file decode) after ``warmup`` throw-away runs with CUDA synchronised; peak GPU memory comes from that pass.
    """
    import torch

    cuda = torch.cuda.is_available() and str(getattr(model, "device", "cpu")).startswith("cuda")
    first = read_image(samples[0].image_path)
    for _ in range(warmup):
        run_model(model, first, conf, iou, imgsz, preprocess)
    recs: list[PredRecord] = []
    for k, s in enumerate(samples):
        img = read_image(s.image_path)
        dets = run_model(model, img, conf, iou, imgsz, preprocess, max_det=EVAL_MAX_DET)
        recs.append(PredRecord(s.image_path.name, s.width, s.height, 0.0, dets))
        if (k + 1) % 50 == 0:
            log.info("  %d/%d images", k + 1, len(samples))
    tconf = conf if timing_conf is None else timing_conf
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    for r, s in list(zip(recs, samples))[:timing_images]:
        img = read_image(s.image_path)
        t0 = time.perf_counter()
        run_model(model, img, tconf, iou, imgsz, preprocess)
        if cuda:
            torch.cuda.synchronize()
        r.ms = (time.perf_counter() - t0) * 1e3
    info = {"timing_conf": tconf, "wall_s_timing_pass": time.perf_counter() - t_all,
            "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6 if cuda else None}
    return recs, info


def restrict_samples(samples: Sequence[Sample], keep: Sequence[int]) -> list[Sample]:
    """Keep only instances of the given class ids and drop images left without any (used when a model knows a subset of classes)."""
    out = []
    for s in samples:
        inst = [g for g in s.instances if g.cls in set(keep)]
        if inst:
            out.append(Sample(s.image_path, s.width, s.height, inst, s.meta))
    return out


def evaluate_run(samples: Sequence[Sample], recs: dict[str, PredRecord], classes: Sequence[str], conf: float,
                 match_iou: float = 0.5, mape_policy: str = "skip", masks: bool = False) -> dict:
    """Counting + detection (+ mask) metrics and a per-image table for one run."""
    box_md, mask_md, rows = [], [], []
    for s in samples:
        r = recs[s.image_path.name]
        box_md.append(build_match_data(s, r.dets))
        if masks:
            mask_md.append(build_match_data(s, r.dets, use_masks=True))
        kept = [d for d in r.dets if d.conf >= conf]
        pg, act = M.match_image(box_md[-1], match_iou, conf)
        tp = int((pg >= 0).sum())
        row = {"image": s.image_path.name, "gt_count": s.count(), "pred_count": len(kept),
               "abs_error": abs(len(kept) - s.count()), "signed_error": len(kept) - s.count(),
               "TP": tp, "FP": len(kept) - tp, "FN": s.count() - tp, "ms": r.ms}
        for k, c in enumerate(classes):
            row[f"gt_{c}"] = s.count(k)
            row[f"pred_{c}"] = sum(d.cls == k for d in kept)
        rows.append(row)
    per_image = pd.DataFrame(rows)
    out = {
        "counting": M.counting_metrics(per_image.pred_count, per_image.gt_count, mape_policy),
        "detection": M.detection_metrics(box_md, classes, conf, match_iou),
        "per_image": per_image,
        "latency_ms_mean": float(per_image.ms.mean()),
    }
    if masks:
        det = M.detection_metrics(mask_md, classes, conf, match_iou)
        det["mean_matched_mask_iou"] = M.mean_matched_iou(mask_md, conf, match_iou)
        out["mask"] = det
    return out


def evaluate_subsets(per_image: pd.DataFrame, tags: pd.DataFrame, tag_names: Sequence[str], min_images: int = 5) -> pd.DataFrame:
    """Counting and detection-count statistics per condition subset (same predictions, filtered by tag)."""
    df = per_image.merge(tags[["image", *[t for t in tag_names if t in tags]]], on="image", how="left")
    rows = []
    for t in tag_names:
        if t not in df:
            continue
        sub = df[df[t] == 1]
        row = {"condition": t, "n_images": len(sub), "gt_fruits": int(sub.gt_count.sum())}
        if len(sub) < min_images:
            row["note"] = f"fewer than {min_images} images: not reported"
        else:
            cm = M.counting_metrics(sub.pred_count, sub.gt_count)
            tp, fp, fn = int(sub.TP.sum()), int(sub.FP.sum()), int(sub.FN.sum())
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            row.update({"MAE": cm["MAE"], "RMSE": cm["RMSE"], "exact_acc": cm["exact_count_accuracy"],
                        "mean_signed_error": cm["mean_signed_error"], "precision": p, "recall": r,
                        "F1": 2 * p * r / max(p + r, 1e-9)})
        rows.append(row)
    return pd.DataFrame(rows)
