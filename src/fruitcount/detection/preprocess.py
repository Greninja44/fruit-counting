"""Optional, documented image preprocessing for robustness experiments.

The stored test images are never modified: enhancement is applied in memory at inference time.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from fruitcount.detection.base import Detection
from fruitcount.utils.geometry import nms

MODES = ("none", "clahe", "gamma", "tile")


def enhance_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel (LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_gamma(img: np.ndarray, target_mean: float = 0.45) -> np.ndarray:
    """Adaptive gamma correction that brightens dark images toward ``target_mean`` mean luminance.

    Images already at or above the target are returned unchanged (no darkening).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0
    if gray >= target_mean or gray <= 1e-3:
        return img
    gamma = np.log(target_mean) / np.log(gray)           # <1 brightens
    lut = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def tile_predict(
    predict_fn: Callable[[np.ndarray], list[Detection]],
    img: np.ndarray,
    tile: int = 640,
    overlap: float = 0.25,
    merge_iou: float = 0.5,
) -> list[Detection]:
    """Sliced inference for small objects: run on overlapping tiles plus the full image, merge with NMS.

    Tile polygons are shifted back to full-image coordinates. Detections cut by tile borders may be
    duplicated; the class-aware NMS removes most of them (documented trade-off: more latency).
    """
    h, w = img.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    dets = list(predict_fn(img))
    if w > tile or h > tile:
        for y in ys:
            for x in xs:
                for d in predict_fn(img[y : y + tile, x : x + tile]):
                    poly = d.polygon + [x, y] if d.polygon is not None else None
                    b = d.box
                    dets.append(Detection((b[0] + x, b[1] + y, b[2] + x, b[3] + y), d.conf, d.cls, d.label, poly))
    if not dets:
        return dets
    keep = nms(np.array([d.box for d in dets]), np.array([d.conf for d in dets]), np.array([d.cls for d in dets]), merge_iou)
    return [dets[i] for i in keep]
