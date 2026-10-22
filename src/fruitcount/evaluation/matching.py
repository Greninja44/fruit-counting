"""Glue between ground-truth samples, cached predictions and the pure metric functions."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from fruitcount.data.dataset import Sample
from fruitcount.detection.base import Detection
from fruitcount.evaluation.metrics import ImageMatchData
from fruitcount.utils.geometry import box_iou_matrix, mask_iou, polygon_mask


def build_match_data(sample: Sample, dets: Sequence[Detection], use_masks: bool = False) -> ImageMatchData:
    """IoU matrix (boxes, or rasterised masks for instances that have polygons) between predictions and GT."""
    pb = np.array([d.box for d in dets]).reshape(-1, 4)
    gb = np.array([g.box for g in sample.instances]).reshape(-1, 4)
    iou = box_iou_matrix(pb, gb)
    if use_masks and len(dets) and len(sample.instances):
        iou = np.zeros_like(iou)
        gt_masks = [polygon_mask(g.polygon, sample.height, sample.width) if g.polygon is not None else None for g in sample.instances]
        for i, d in enumerate(dets):
            if d.polygon is None:
                continue
            pm = polygon_mask(d.polygon, sample.height, sample.width)
            for j, gm in enumerate(gt_masks):
                # Only rasterise pairs whose boxes overlap; the rest have IoU 0.
                if gm is not None and box_iou_matrix(pb[i : i + 1], gb[j : j + 1])[0, 0] > 0:
                    iou[i, j] = mask_iou(pm, gm)
    return ImageMatchData(
        sample.name,
        np.array([d.conf for d in dets], float),
        np.array([d.cls for d in dets], int),
        np.array([g.cls for g in sample.instances], int),
        iou,
    )
