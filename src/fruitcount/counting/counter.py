"""Counting logic: confidence filtering, class-wise counts, and video counting.

Two distinct quantities are kept strictly separate for video:

* ``frame_counts``  - fruits *visible in each frame* (instantaneous, per-frame).
* ``unique_tracked`` - distinct fruit *identities* assigned by a tracker over the clip.

The sum of per-frame counts is never reported as a number of unique fruits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from fruitcount.detection.base import Detection
from fruitcount.tracking.tracker import UniqueTrackCounter

UNIQUE_COUNT_CAVEAT = (
    "Unique-fruit count comes from tracker IDs. It assumes a moving camera/scene with limited occlusion: "
    "prolonged occlusion or leaving the view can split one fruit into several IDs (over-count), while ID "
    "switches or merged tracks can under-count. Treat it as an estimate, not ground truth."
)


@dataclass
class CountResult:
    total: int
    per_class: dict[str, int]


def filter_detections(dets: Sequence[Detection], conf: float, min_area: float = 0.0) -> list[Detection]:
    """Keep detections with ``conf >= threshold`` (inclusive) and ``area >= min_area`` pixels."""
    return [d for d in dets if d.conf >= conf and d.area >= min_area]


def count_detections(dets: Sequence[Detection], classes: Sequence[str], conf: float = 0.0) -> CountResult:
    """Total and class-wise counts after confidence filtering. Every canonical class is reported (0 if absent)."""
    kept = filter_detections(dets, conf)
    per = {c: 0 for c in classes}
    for d in kept:
        per[d.label] = per.get(d.label, 0) + 1
    return CountResult(len(kept), per)


@dataclass
class VideoCounter:
    """Accumulates per-frame visible counts and (separately) tracked unique identities."""

    classes: Sequence[str]
    conf: float = 0.25
    min_track_frames: int = 3
    frame_counts: list[int] = field(default_factory=list)
    frame_class_counts: list[dict[str, int]] = field(default_factory=list)
    _tracks: UniqueTrackCounter = field(init=False)

    def __post_init__(self) -> None:
        self._tracks = UniqueTrackCounter(self.min_track_frames)

    def update(self, dets: Sequence[Detection]) -> CountResult:
        res = count_detections(dets, self.classes, self.conf)
        self.frame_counts.append(res.total)
        self.frame_class_counts.append(res.per_class)
        self._tracks.update(self.frame_index, filter_detections(dets, self.conf))
        return res

    @property
    def frame_index(self) -> int:
        return len(self.frame_counts)

    def unique_tracked(self, tracking_enabled: bool = True) -> int | None:
        """Unique tracked identities, or ``None`` when tracking was not used (never a sum of frame counts)."""
        return self._tracks.unique_count() if tracking_enabled else None

    def summary(self, tracking_enabled: bool = True) -> dict:
        fc = self.frame_counts
        out = {
            "frames": len(fc),
            "visible_per_frame": {"mean": sum(fc) / len(fc) if fc else 0.0, "max": max(fc) if fc else 0},
            "unique_tracked": self.unique_tracked(tracking_enabled),
            "unique_tracked_by_class": self._tracks.unique_by_class() if tracking_enabled else None,
            "min_track_frames": self.min_track_frames,
        }
        out["note"] = UNIQUE_COUNT_CAVEAT if tracking_enabled else "Tracking disabled: only per-frame visible counts are valid."
        return out
