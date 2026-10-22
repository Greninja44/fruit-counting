"""Shared fixtures: a mock model, synthetic images/videos and a tiny YOLO-format dataset on disk."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fruitcount.detection.base import Detection

CLASSES = ["apple", "orange", "banana"]


def det(x1, y1, x2, y2, conf=0.9, cls=0, polygon=None, track_id=None) -> Detection:
    return Detection((x1, y1, x2, y2), conf, cls, CLASSES[cls], polygon, track_id)


class MockModel:
    """Returns pre-programmed detections; honours ``conf`` like a real model would."""

    name, task, classes = "mock", "detect", CLASSES

    def __init__(self, dets=None, per_call=None):
        self.dets, self.per_call, self.calls = dets or [], per_call, 0

    def predict(self, image, conf=0.25, iou=0.7, imgsz=640, max_det=300):
        out = self.per_call[self.calls % len(self.per_call)] if self.per_call else self.dets
        self.calls += 1
        return [d for d in out if d.conf >= conf]

    def model_size_mb(self):
        return 1.0


@pytest.fixture
def classes():
    return CLASSES


@pytest.fixture
def blank_image():
    return np.full((120, 160, 3), 127, np.uint8)


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    """3 test images: 2 fruits, 0 fruits (background), 1 crowd-flagged image. Includes polygons."""
    root = tmp_path / "ds"
    for d in ("images/test", "labels/test", "labels_seg/test"):
        (root / d).mkdir(parents=True)
    rows = []
    specs = {
        "a.jpg": ["0 0.3 0.4 0.2 0.2", "1 0.7 0.6 0.2 0.3"],
        "b.jpg": [],
        "c.jpg": ["2 0.5 0.5 0.4 0.4"],
    }
    for name, lines in specs.items():
        cv2.imwrite(str(root / "images/test" / name), np.full((100, 200, 3), 90, np.uint8))
        (root / "labels/test" / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        seg = []
        for ln in lines:
            c, cx, cy, w, h = ln.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            seg.append(f"{c} {cx-w/2} {cy-h/2} {cx+w/2} {cy-h/2} {cx+w/2} {cy+h/2} {cx-w/2} {cy+h/2}")
        (root / "labels_seg/test" / f"{Path(name).stem}.txt").write_text("\n".join(seg) + ("\n" if seg else ""))
        rows.append(f"{name},test,src_{name},200,100,{int(name == 'c.jpg')}")
    (root / "meta.csv").write_text("image,split,source_id,width,height,has_crowd\n" + "\n".join(rows) + "\n")
    return root


@pytest.fixture
def synthetic_video(tmp_path: Path):
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(12):
        f = np.full((120, 160, 3), 60, np.uint8)
        cv2.circle(f, (30 + 5 * i, 60), 15, (0, 140, 255), -1)
        w.write(f)
    w.release()
    return p
