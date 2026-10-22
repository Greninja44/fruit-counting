#!/usr/bin/env python
"""Count fruit in a video.

    python scripts/predict_video.py --source clip.mp4 --mode frame            # visible fruit per frame only
    python scripts/predict_video.py --source clip.mp4 --mode track            # + unique tracked fruit (ByteTrack)

'visible per frame' is how many fruits are on screen in each frame. 'unique tracked' is the number of
distinct tracker identities; it is an estimate that fails under long occlusions / ID switches.
The two numbers are never derived from each other.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.counting.video import process_video  # noqa: E402
from fruitcount.detection.base import ModelLoadError  # noqa: E402
from fruitcount.detection.factory import build_model  # noqa: E402
from fruitcount.tracking.tracker import make_frame_tracker  # noqa: E402
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402
from fruitcount.utils.io import InvalidInputError  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("predict_video")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--mode", choices=["frame", "track"], default="frame")
    ap.add_argument("--approach", choices=["detect", "segment"], default="detect")
    ap.add_argument("--tracker", choices=["bytetrack", "iou"], default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config, {"inference.conf": a.conf, "inference.iou": a.iou, "inference.imgsz": a.imgsz, "device": a.device,
                                 "tracking.tracker": a.tracker})
    inf, tr = cfg["inference"], cfg["tracking"]
    out = resolve_path(a.out_dir or cfg["paths"]["visualizations"])
    stem = Path(a.source).stem
    try:
        model = build_model(cfg, a.approach, a.weights)
        tracking = a.mode == "track"
        if tracking:
            fn = make_frame_tracker(model, tr["tracker"], inf["conf"], inf["iou"], inf["imgsz"])
        else:
            fn = lambda f: model.predict(f, inf["conf"], inf["iou"], inf["imgsz"], inf["max_det"])  # noqa: E731
        s = process_video(a.source, fn, cfg["classes"], inf["conf"], tracking, tr["min_track_frames"],
                          out / f"{stem}_{a.approach}_{a.mode}.mp4", out / f"{stem}_{a.approach}_{a.mode}.json", a.stride, a.max_frames)
    except (ModelLoadError, InvalidInputError, RuntimeError, ValueError) as e:
        log.error("%s", e)
        return 2
    log.info("frames processed: %d | visible per frame: mean %.2f, max %d", s["frames_processed"], s["visible_per_frame"]["mean"], s["visible_per_frame"]["max"])
    if tracking:
        log.info("unique tracked fruit (estimate): %s %s", s["unique_tracked"], s["unique_tracked_by_class"])
        log.info("NOTE: %s", s["note"])
    else:
        log.info("tracking off: no unique count reported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
