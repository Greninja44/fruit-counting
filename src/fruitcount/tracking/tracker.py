"""Multi-object tracking helpers.

* :class:`UniqueTrackCounter` turns track IDs into a unique-object count.
* :class:`IoUTracker` is a dependency-free greedy IoU tracker (used in tests and as a fallback).
* ByteTrack runs inside Ultralytics (``UltralyticsModel.track``); :func:`make_frame_tracker` wires
  either backend behind one ``update(frame) -> detections-with-track_id`` interface.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from fruitcount.detection.base import Detection
from fruitcount.utils.geometry import box_iou_matrix


class UniqueTrackCounter:
    """Counts distinct track IDs that persisted for at least ``min_frames`` frames.

    The minimum-lifetime rule suppresses one- or two-frame ID flicker, a common source of over-counting.
    """

    def __init__(self, min_frames: int = 3):
        self.min_frames = min_frames
        self.frames_seen: dict[int, int] = defaultdict(int)
        self.labels: dict[int, list[str]] = defaultdict(list)

    def update(self, frame_idx: int, dets: Sequence[Detection]) -> None:
        for d in dets:
            if d.track_id is None:
                continue
            self.frames_seen[d.track_id] += 1
            self.labels[d.track_id].append(d.label)

    def _confirmed(self) -> list[int]:
        return [t for t, n in self.frames_seen.items() if n >= self.min_frames]

    def unique_count(self) -> int:
        return len(self._confirmed())

    def unique_by_class(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for t in self._confirmed():
            lab = max(set(self.labels[t]), key=self.labels[t].count)   # majority label over the track
            out[lab] += 1
        return dict(out)


class IoUTracker:
    """Greedy-Hungarian IoU tracker with a short memory (no motion model)."""

    def __init__(self, iou_thr: float = 0.3, max_age: int = 5):
        self.iou_thr, self.max_age = iou_thr, max_age
        self._tracks: dict[int, tuple[tuple, int, int]] = {}   # id -> (box, cls, age)
        self._next = 1

    def update(self, dets: Sequence[Detection]) -> list[Detection]:
        ids = list(self._tracks)
        assigned: dict[int, int] = {}
        if ids and dets:
            iou = box_iou_matrix(np.array([self._tracks[i][0] for i in ids]), np.array([d.box for d in dets]))
            for ti, i in enumerate(ids):                       # only same-class matches allowed
                for di, d in enumerate(dets):
                    if self._tracks[i][1] != d.cls:
                        iou[ti, di] = 0
            r, c = linear_sum_assignment(-iou)
            for ti, di in zip(r, c):
                if iou[ti, di] >= self.iou_thr:
                    assigned[di] = ids[ti]
        new_tracks = {}
        for di, d in enumerate(dets):
            tid = assigned.get(di)
            if tid is None:
                tid, self._next = self._next, self._next + 1
            d.track_id = tid
            new_tracks[tid] = (d.box, d.cls, 0)
        for i in ids:                                          # age unmatched tracks
            if i not in new_tracks:
                box, cls, age = self._tracks[i]
                if age + 1 <= self.max_age:
                    new_tracks[i] = (box, cls, age + 1)
        self._tracks = new_tracks
        return list(dets)


def make_frame_tracker(model, kind: str, conf: float, iou: float, imgsz: int) -> Callable[[np.ndarray], list[Detection]]:
    """Return ``fn(frame) -> detections with track_id`` for ``kind`` in {'bytetrack', 'iou'}."""
    if kind == "bytetrack":
        if not hasattr(model, "track"):
            raise ValueError("ByteTrack needs an Ultralytics-backed model; use tracker='iou' otherwise")
        if hasattr(model, "reset_tracker"):
            model.reset_tracker()
        return lambda frame: model.track(frame, conf, iou, imgsz, tracker="bytetrack.yaml", persist=True)
    if kind == "iou":
        tr = IoUTracker()
        return lambda frame: tr.update(model.predict(frame, conf, iou, imgsz))
    raise ValueError(f"unknown tracker '{kind}' (expected 'bytetrack' or 'iou')")
