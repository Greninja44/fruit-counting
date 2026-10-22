"""Box / mask geometry shared by inference post-processing and evaluation."""
from __future__ import annotations

import cv2
import numpy as np


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def box_intersection_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection *area* between (N,4) and (M,4) xyxy arrays."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return ix * iy


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float) -> list[int]:
    """Class-aware greedy NMS. Returns kept indices, highest score first."""
    order = list(np.argsort(-np.asarray(scores)))
    keep: list[int] = []
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        ious = box_iou_matrix(boxes[i : i + 1], boxes[order])[0]
        order = [j for j, v in zip(order, ious) if classes[j] != classes[i] or v <= iou_thr]
    return keep


def polygon_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterise an (N,2) pixel polygon to a boolean mask."""
    m = np.zeros((height, width), np.uint8)
    if polygon is not None and len(polygon) >= 3:
        cv2.fillPoly(m, [np.round(polygon).astype(np.int32)], 1)
    return m.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
