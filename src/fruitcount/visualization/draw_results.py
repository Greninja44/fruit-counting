"""Draw boxes, instance masks, track IDs and count summaries."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from fruitcount.detection.base import Detection

_PALETTE = [(60, 76, 231), (0, 165, 255), (0, 220, 255), (113, 204, 46), (219, 152, 52), (182, 89, 155), (156, 188, 26)]


def class_color(cls: int) -> tuple[int, int, int]:
    return _PALETTE[cls % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    dets: Sequence[Detection],
    counts_text: Sequence[str] | None = None,
    show_conf: bool = True,
    mask_alpha: float = 0.4,
    index_instances: bool = False,
) -> np.ndarray:
    """Return an annotated copy. Masks are filled when polygons exist; ``counts_text`` is a header panel."""
    out = image.copy()
    overlay = out.copy()
    for d in dets:
        if d.polygon is not None and len(d.polygon) >= 3:
            cv2.fillPoly(overlay, [np.round(d.polygon).astype(np.int32)], class_color(d.cls))
    if any(d.polygon is not None for d in dets):
        out = cv2.addWeighted(overlay, mask_alpha, out, 1 - mask_alpha, 0)
        for d in dets:
            if d.polygon is not None and len(d.polygon) >= 3:
                cv2.polylines(out, [np.round(d.polygon).astype(np.int32)], True, class_color(d.cls), 1, cv2.LINE_AA)
    scale = max(0.4, min(out.shape[:2]) / 900)
    th = max(1, int(round(scale * 2)))
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = map(int, d.box)
        col = class_color(d.cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        tag = d.label
        if index_instances:
            tag = f"#{i} {tag}"
        if d.track_id is not None:
            tag += f" id{d.track_id}"
        if show_conf:
            tag += f" {d.conf:.2f}"
        (tw, tgh), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, 1)
        ty = max(y1, tgh + 4)
        cv2.rectangle(out, (x1, ty - tgh - 4), (x1 + tw + 4, ty), col, -1)
        lum = 0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2]
        txt_col = (20, 20, 20) if lum > 140 else (255, 255, 255)
        cv2.putText(out, tag, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, txt_col, 1, cv2.LINE_AA)
    if counts_text:
        fs = scale * 0.7
        lh = int(28 * fs / 0.7 * 0.8) + 6
        w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0] for t in counts_text) + 16
        cv2.rectangle(out, (0, 0), (w, lh * len(counts_text) + 8), (0, 0, 0), -1)
        for k, t in enumerate(counts_text):
            cv2.putText(out, t, (8, lh * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def counts_header(total: int, per_class: dict[str, int], label: str = "Fruits") -> list[str]:
    """Header lines such as ``Fruits: 5`` / ``apple: 3  orange: 2``."""
    shown = "  ".join(f"{k}: {v}" for k, v in per_class.items() if v)
    return [f"{label}: {total}"] + ([shown] if shown else [])


_ERR_COLORS = {"tp": (80, 200, 80), "false_positive": (0, 0, 255), "duplicate": (255, 0, 255), "merged": (0, 140, 255),
               "wrong_class": (255, 128, 0), "poor_localization": (0, 215, 255)}


def draw_error_overlay(image: np.ndarray, gt_boxes: Sequence, gt_missed: dict[int, str], preds: Sequence[Detection],
                       pred_cat: dict[int, str], header: str = "") -> np.ndarray:
    """GT fruit = thin green outline (missed GT: thick red, labelled); predictions coloured by error category."""
    out = image.copy()
    sc = max(0.4, min(out.shape[:2]) / 900)
    for j, b in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, b)
        if j in gt_missed:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 200), max(2, int(3 * sc)), cv2.LINE_AA)
            cv2.putText(out, "MISSED:" + gt_missed[j].replace("missed_", ""), (x1 + 2, max(12, y1 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, sc * 0.55, (0, 0, 200), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (60, 180, 60), 1, cv2.LINE_AA)
    for i, d in enumerate(preds):
        cat = pred_cat.get(i, "tp")
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), _ERR_COLORS[cat], max(1, int(2 * sc)), cv2.LINE_AA)
        if cat != "tp":
            cv2.putText(out, cat, (x1 + 2, min(out.shape[0] - 4, y2 - 4)), cv2.FONT_HERSHEY_SIMPLEX, sc * 0.55,
                        _ERR_COLORS[cat], 1, cv2.LINE_AA)
    if header:
        cv2.rectangle(out, (0, 0), (int(len(header) * 11 * sc) + 10, int(26 * sc) + 6), (0, 0, 0), -1)
        cv2.putText(out, header, (6, int(20 * sc) + 2), cv2.FONT_HERSHEY_SIMPLEX, sc * 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out
