"""Video counting: per-frame visible counts, optionally with tracker-based unique counting."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from fruitcount.counting.counter import VideoCounter, filter_detections
from fruitcount.detection.base import Detection
from fruitcount.utils.io import open_video
from fruitcount.utils.log import get_logger
from fruitcount.visualization.draw_results import draw_detections

log = get_logger(__name__)


def process_video(
    src: str | Path,
    frame_fn: Callable[[np.ndarray], list[Detection]],
    classes: Sequence[str],
    conf: float,
    tracking: bool,
    min_track_frames: int = 3,
    out_video: str | Path | None = None,
    out_json: str | Path | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> dict:
    """Run ``frame_fn`` on each frame and count.

    ``frame_fn`` returns detections (with ``track_id`` set when ``tracking`` is on). The result keeps
    ``visible_per_frame`` (what is on screen) separate from ``unique_tracked`` (tracker identities); the latter
    is ``None`` when tracking is off. Raises ``InvalidInputError`` for missing/corrupt video.
    """
    cap = open_video(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_video:
        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
    vc = VideoCounter(classes, conf, min_track_frames)
    n_read, t0 = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:                       # end of stream or a corrupt frame: stop cleanly
                break
            n_read += 1
            if (n_read - 1) % stride:
                continue
            dets = filter_detections(frame_fn(frame), conf)
            res = vc.update(dets)
            if writer is not None:
                lines = [f"Visible now: {res.total}"]
                if tracking:
                    lines.append(f"Unique tracked so far: {vc.unique_tracked()} (estimate)")
                writer.write(draw_detections(frame, dets, lines))
            if max_frames and vc.frame_index >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    summary = vc.summary(tracking)
    summary.update({"source": str(src), "frames_read": n_read, "frames_processed": vc.frame_index, "stride": stride,
                    "tracking": tracking, "processing_seconds": round(time.perf_counter() - t0, 2),
                    "frame_counts": vc.frame_counts})
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(summary, indent=2))
    return summary
