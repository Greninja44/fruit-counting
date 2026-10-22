"""Counting, detection and mask metrics. Pure numpy: no model code, fully unit-testable.

Matching rule (documented, used for P/R/F1, AP and error analysis):
  predictions are processed in descending confidence; each is matched to the *unmatched* ground-truth
  instance of the same class with the highest IoU, if IoU >= threshold. Unmatched predictions are false
  positives, unmatched ground truths are false negatives. AP uses 101-point interpolation (COCO style),
  computed per class and averaged over classes that have ground truth; mAP@0.5:0.95 averages the ten
  thresholds 0.50, 0.55, ..., 0.95.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)


# ----------------------------------------------------------------------------- counting
def counting_metrics(pred: Sequence[int], gt: Sequence[int], mape_zero_policy: str = "skip", eps: float = 1.0) -> dict:
    """MAE, RMSE, MAPE, exact-count accuracy and within-one accuracy.

    MAPE policy for images with zero ground-truth fruit (division by zero):
      ``skip``    - excluded from MAPE (their count of excluded images is reported);
      ``epsilon`` - denominator is ``max(gt, eps)`` so a false detection on an empty image costs ``|pred|/eps``.
    """
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    if p.shape != g.shape:
        raise ValueError(f"length mismatch: {p.shape} vs {g.shape}")
    if len(p) == 0:
        raise ValueError("no images to evaluate")
    if mape_zero_policy not in ("skip", "epsilon"):
        raise ValueError("mape_zero_policy must be 'skip' or 'epsilon'")
    err = p - g
    ae = np.abs(err)
    if mape_zero_policy == "skip":
        m = g > 0
        mape = float(np.mean(ae[m] / g[m]) * 100) if m.any() else float("nan")
        excluded = int((~m).sum())
    else:
        mape = float(np.mean(ae / np.maximum(g, eps)) * 100)
        excluded = 0
    return {
        "n_images": int(len(p)),
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE_percent": mape,
        "MAPE_policy": mape_zero_policy,
        "MAPE_excluded_images": excluded,
        "exact_count_accuracy": float((ae == 0).mean()),
        "within_one_accuracy": float((ae <= 1).mean()),
        "mean_signed_error": float(err.mean()),   # >0 over-counting, <0 under-counting
    }


# ----------------------------------------------------------------------------- matching
@dataclass
class ImageMatchData:
    """Everything needed to match one image: prediction scores/classes, GT classes, and IoU matrix (P, G)."""

    name: str
    pred_conf: np.ndarray
    pred_cls: np.ndarray
    gt_cls: np.ndarray
    iou: np.ndarray


def match_image(d: ImageMatchData, thr: float, conf_thr: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Greedy class-aware matching. Returns (pred_gt_index with -1 for unmatched, pred_active mask)."""
    P = len(d.pred_conf)
    active = d.pred_conf >= conf_thr
    pred_gt = np.full(P, -1, int)
    used = np.zeros(len(d.gt_cls), bool)
    for i in np.argsort(-d.pred_conf, kind="stable"):
        if not active[i] or len(d.gt_cls) == 0:
            continue
        cand = np.where((d.gt_cls == d.pred_cls[i]) & ~used & (d.iou[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(d.iou[i, cand])]
            pred_gt[i] = j
            used[j] = True
    return pred_gt, active


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO)."""
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, x, side="left")
    return float(np.mean(mpre[np.minimum(idx, len(mpre) - 1)] * (idx < len(mpre))))


def _ap_at(images: Sequence[ImageMatchData], cls: int, thr: float) -> float | None:
    n_gt = int(sum((d.gt_cls == cls).sum() for d in images))
    if n_gt == 0:
        return None
    confs, tps = [], []
    for d in images:
        sel = d.pred_cls == cls
        if not sel.any():
            continue
        sub = ImageMatchData(d.name, d.pred_conf[sel], d.pred_cls[sel], d.gt_cls, d.iou[sel])
        pg, _ = match_image(sub, thr)
        confs.append(sub.pred_conf)
        tps.append(pg >= 0)
    if not confs:
        return 0.0
    c, t = np.concatenate(confs), np.concatenate(tps)
    order = np.argsort(-c, kind="stable")
    t = t[order]
    tp = np.cumsum(t)
    fp = np.cumsum(~t)
    return average_precision(tp / n_gt, tp / np.maximum(tp + fp, 1e-9))


def detection_metrics(images: Sequence[ImageMatchData], classes: Sequence[str], conf_thr: float, match_iou: float = 0.5) -> dict:
    """Precision/recall/F1 at ``conf_thr`` & ``match_iou`` plus mAP@0.5 and mAP@0.5:0.95 over all predictions."""
    tp = fp = fn = 0
    per = {c: [0, 0, 0] for c in classes}
    for d in images:
        pg, act = match_image(d, match_iou, conf_thr)
        matched = pg[pg >= 0]
        n_tp = len(matched)
        tp += n_tp
        fp += int(act.sum()) - n_tp
        fn += len(d.gt_cls) - n_tp
        for k, c in enumerate(classes):
            m = act & (d.pred_cls == k)
            t = int(((pg >= 0) & m).sum())
            per[c][0] += t
            per[c][1] += int(m.sum()) - t
            per[c][2] += int((d.gt_cls == k).sum()) - t
    def prf(t, f, n):
        p = t / (t + f) if t + f else 0.0
        r = t / (t + n) if t + n else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    p, r, f1 = prf(tp, fp, fn)
    aps = {c: [a for a in (_ap_at(images, k, t) for t in IOU_THRS)] for k, c in enumerate(classes)}
    ap50 = {c: v[0] for c, v in aps.items() if v[0] is not None}
    ap_all = {c: float(np.mean(v)) for c, v in aps.items() if v[0] is not None}
    return {
        "conf_threshold": conf_thr, "match_iou": match_iou,
        "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "F1": f1,
        "mAP50": float(np.mean(list(ap50.values()))) if ap50 else float("nan"),
        "mAP50_95": float(np.mean(list(ap_all.values()))) if ap_all else float("nan"),
        "AP50_per_class": ap50,
        "per_class_prf": {c: dict(zip(("precision", "recall", "F1"), prf(*v))) for c, v in per.items()},
    }


def mean_matched_iou(images: Sequence[ImageMatchData], conf_thr: float, match_iou: float = 0.5) -> float:
    """Mean IoU (of the supplied IoU matrices, e.g. masks) over matched TP pairs."""
    vals = []
    for d in images:
        pg, _ = match_image(d, match_iou, conf_thr)
        vals += [d.iou[i, j] for i, j in enumerate(pg) if j >= 0]
    return float(np.mean(vals)) if vals else float("nan")
