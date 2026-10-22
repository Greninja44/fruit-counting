"""Automated error analysis against human ground truth, plus qualitative flags when GT is absent.

Categories (all determined with the matching rule in :mod:`metrics` at the operating confidence):

Prediction-side
  false_positive      unmatched prediction that overlaps no GT fruit (IoU < 0.1): background/non-fruit
  duplicate           unmatched prediction with IoU >= match_iou to a GT already matched by a better prediction
  merged              unmatched prediction covering >= 2 distinct GT fruits (each >= 50% inside it)
  wrong_class         unmatched prediction with IoU >= match_iou to a GT of a *different* class
  poor_localization   unmatched prediction with 0.1 <= IoU < match_iou to a same-class GT
Ground-truth-side
  missed_small        unmatched GT with box area < 32x32 px
  missed_touching     unmatched GT whose box overlaps another GT (IoU > 0.1) - poor separation
  missed_other        any other unmatched GT
Image-level
  right_total_wrong_objects   predicted count == GT count but FP > 0 and FN > 0
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from fruitcount.data.dataset import Sample
from fruitcount.detection.base import Detection
from fruitcount.evaluation.metrics import match_image
from fruitcount.evaluation.matching import build_match_data
from fruitcount.evaluation.tags import SMALL_AREA
from fruitcount.utils.geometry import box_intersection_matrix, box_iou_matrix

PRED_CATS = ["false_positive", "duplicate", "merged", "wrong_class", "poor_localization"]
GT_CATS = ["missed_small", "missed_touching", "missed_other"]
IMAGE_CATS = ["right_total_wrong_objects"]
ALL_CATS = PRED_CATS + GT_CATS + IMAGE_CATS


@dataclass
class ImageErrors:
    name: str
    gt_count: int
    pred_count: int
    pred_cat: dict[int, str] = field(default_factory=dict)   # pred index (in filtered list) -> category
    gt_cat: dict[int, str] = field(default_factory=dict)     # gt index -> category (only unmatched GT)
    matched_pred: set[int] = field(default_factory=set)
    matched_gt: set[int] = field(default_factory=set)
    image_cats: list[str] = field(default_factory=list)

    @property
    def categories(self) -> set[str]:
        return set(self.pred_cat.values()) | set(self.gt_cat.values()) | set(self.image_cats)


def analyse_image(sample: Sample, dets: Sequence[Detection], conf: float, match_iou: float = 0.5) -> tuple[ImageErrors, list[Detection]]:
    """Categorise every error in one image. Returns the errors and the filtered prediction list they index."""
    kept = [d for d in dets if d.conf >= conf]
    md = build_match_data(sample, kept)
    pg, _ = match_image(md, match_iou)
    err = ImageErrors(sample.name, len(sample.instances), len(kept))
    gb = np.array([g.box for g in sample.instances]).reshape(-1, 4)
    pb = np.array([d.box for d in kept]).reshape(-1, 4)
    g_area = np.array([g.area for g in sample.instances])
    inter = box_intersection_matrix(pb, gb)
    iou = md.iou
    matched_gt = {int(j) for j in pg if j >= 0}
    err.matched_gt, err.matched_pred = matched_gt, {i for i, j in enumerate(pg) if j >= 0}
    for i in range(len(kept)):
        if pg[i] >= 0:
            continue
        same = md.gt_cls == md.pred_cls[i]
        covered = np.where((inter[i] / np.maximum(g_area, 1e-9) >= 0.5))[0] if len(g_area) else np.array([], int)
        if len(iou[i]) and (iou[i][same] >= match_iou).any():
            cat = "duplicate"
        elif len(iou[i]) and (iou[i][~same] >= match_iou).any():
            cat = "wrong_class"
        elif len(covered) >= 2:
            cat = "merged"
        elif len(iou[i]) and (iou[i][same] >= 0.1).any():
            cat = "poor_localization"
        else:
            cat = "false_positive"
        err.pred_cat[i] = cat
    gt_iou = box_iou_matrix(gb, gb)
    np.fill_diagonal(gt_iou, 0)
    merged_covered = {j for i, c in err.pred_cat.items() if c == "merged" for j in np.where(inter[i] / np.maximum(g_area, 1e-9) >= 0.5)[0]}
    for j in range(len(gb)):
        if j in matched_gt:
            continue
        if g_area[j] < SMALL_AREA:
            err.gt_cat[j] = "missed_small"
        elif len(gb) > 1 and gt_iou[j].max() > 0.1 or j in merged_covered:
            err.gt_cat[j] = "missed_touching"
        else:
            err.gt_cat[j] = "missed_other"
    n_fp, n_fn = len(kept) - len(err.matched_pred), len(gb) - len(matched_gt)
    if len(kept) == len(gb) and n_fp > 0 and n_fn > 0:
        err.image_cats.append("right_total_wrong_objects")
    return err, kept


def summarize(errors: Sequence[ImageErrors]) -> dict:
    """Totals per category and the number of images affected."""
    inst, imgs = Counter(), Counter()
    for e in errors:
        for c in list(e.pred_cat.values()) + list(e.gt_cat.values()) + e.image_cats:
            inst[c] += 1
        for c in e.categories:
            imgs[c] += 1
    return {c: {"instances": inst[c], "images": imgs[c]} for c in ALL_CATS}


def qualitative_flags(dets: Sequence[Detection], conf: float, low_conf_band: float = 0.15) -> dict[str, int]:
    """Observations for images *without* ground truth. These are hints, never confirmed FP/FN.

    * possible_duplicates: same-class pairs of kept boxes with IoU > 0.7
    * possible_merged: a kept box that contains the centres of >= 2 other kept boxes
    * uncertain: detections within ``low_conf_band`` above the threshold
    """
    kept = [d for d in dets if d.conf >= conf]
    out = {"possible_duplicates": 0, "possible_merged": 0, "uncertain": sum(d.conf < conf + low_conf_band for d in kept)}
    if len(kept) > 1:
        b = np.array([d.box for d in kept])
        iou = box_iou_matrix(b, b)
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if kept[i].cls == kept[j].cls and iou[i, j] > 0.7:
                    out["possible_duplicates"] += 1
        c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        for i in range(len(kept)):
            inside = (c[:, 0] > b[i, 0]) & (c[:, 0] < b[i, 2]) & (c[:, 1] > b[i, 1]) & (c[:, 1] < b[i, 3])
            inside[i] = False
            out["possible_merged"] += int(inside.sum() >= 2)
    return out
