"""Condition tags for robustness subsets.

Two sources, with manual labels always taking precedence:

* **auto** - heuristics from image statistics and GT geometry (thresholds below). These are *proxies*,
  not human judgements, and are reported as such.
* **manual** - ``data/annotations/condition_tags.csv`` (template: ``scripts/tag_conditions.py --template``).
  Columns: ``image`` plus any of TAGS with values 1 / 0 / empty (empty = keep the auto value).

``occluded`` and ``background_clutter`` / ``viewpoint`` have no reliable automatic signal: ``occluded_proxy``
is only a geometric hint, and the other two are manual-only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fruitcount.data.dataset import Sample
from fruitcount.utils.geometry import box_intersection_matrix, box_iou_matrix

AUTO_TAGS = ["small_fruit", "large_fruit", "overlapping", "occluded_proxy", "dense_cluster",
             "bright_sunlight", "uneven_lighting", "low_light", "blurry"]
MANUAL_ONLY_TAGS = ["occluded", "cluttered_background", "unusual_viewpoint"]
TAGS = AUTO_TAGS + MANUAL_ONLY_TAGS

SMALL_AREA, LARGE_AREA = 32**2, 96**2     # COCO absolute-pixel size convention, on median GT box area
DENSE_MIN = 10                             # >= this many GT fruits
OVERLAP_IOU = 0.10                         # any GT pair with box IoU above this
OCCLUDED_COVER = 0.30                      # a GT box >30% covered by other GT boxes
LOW_LIGHT_MEAN = 70                        # mean gray level (0-255)
BRIGHT_MEAN, BRIGHT_SAT = 165, 0.08        # mean gray level, or fraction of near-saturated pixels
UNEVEN_STD = 40                            # std of 4x4 block-mean gray levels
BLUR_LAPVAR = 100                          # variance of Laplacian on a 640-px-long-side image


def image_stats(img: np.ndarray) -> dict[str, float]:
    h, w = img.shape[:2]
    s = 640 / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blocks = cv2.resize(g, (4, 4), interpolation=cv2.INTER_AREA)
    return {"mean_gray": float(g.mean()), "saturated_frac": float((g > 245).mean()),
            "block_std": float(blocks.std()), "lap_var": float(cv2.Laplacian(g, cv2.CV_64F).var())}


def auto_tags(img: np.ndarray, sample: Sample) -> dict[str, int]:
    st = image_stats(img)
    boxes = np.array([g.box for g in sample.instances]).reshape(-1, 4)
    areas = np.array([g.area for g in sample.instances])
    tags = dict.fromkeys(TAGS, 0)
    n = len(boxes)
    if n:
        med = float(np.median(areas))
        tags["small_fruit"] = int(med < SMALL_AREA)
        tags["large_fruit"] = int(med > LARGE_AREA)
        tags["dense_cluster"] = int(n >= DENSE_MIN)
        if n > 1:
            iou = box_iou_matrix(boxes, boxes)
            np.fill_diagonal(iou, 0)
            tags["overlapping"] = int(iou.max() > OVERLAP_IOU)
            inter = box_intersection_matrix(boxes, boxes)
            np.fill_diagonal(inter, 0)
            cover = inter.max(axis=1) / np.maximum(areas, 1e-9)   # worst single-neighbour cover (lower bound)
            tags["occluded_proxy"] = int((cover > OCCLUDED_COVER).any())
    tags["low_light"] = int(st["mean_gray"] < LOW_LIGHT_MEAN)
    tags["bright_sunlight"] = int(st["mean_gray"] > BRIGHT_MEAN or st["saturated_frac"] > BRIGHT_SAT)
    tags["uneven_lighting"] = int(st["block_std"] > UNEVEN_STD)
    tags["blurry"] = int(st["lap_var"] < BLUR_LAPVAR)
    return tags


def apply_manual_overrides(auto: pd.DataFrame, manual_csv: str | Path | None) -> pd.DataFrame:
    """Overlay manual labels (non-empty cells) on the auto tags; manual-only tags default to 0 if never set."""
    df = auto.copy()
    for t in TAGS:
        if t not in df:
            df[t] = 0
    if manual_csv and Path(manual_csv).is_file():
        man = pd.read_csv(manual_csv).set_index("image")
        for t in TAGS:
            if t in man:
                for img, v in man[t].dropna().items():
                    df.loc[df.image == img, t] = int(v)
    return df
