"""Gradio demo: upload an image or video, choose model/approach and thresholds, view counts, download results.

    python app.py [--port 7860] [--share]
"""
from __future__ import annotations

import argparse
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from fruitcount.counting.counter import count_detections
from fruitcount.counting.video import process_video
from fruitcount.detection.base import ModelLoadError
from fruitcount.detection.factory import build_model
from fruitcount.detection.predict import run_model
from fruitcount.tracking.tracker import make_frame_tracker
from fruitcount.utils.config import PROJECT_ROOT, load_config, resolve_path
from fruitcount.utils.io import InvalidInputError
from fruitcount.visualization.draw_results import counts_header, draw_detections

CFG = load_config()
OUT = Path(tempfile.mkdtemp(prefix="fruitcount_ui_"))
_WARM: set[int] = set()


def weight_choices(kind: str) -> list[str]:
    """Weights found in models/ and runs/**/best.pt, split into detection / segmentation by filename."""
    found = sorted(p for p in list((PROJECT_ROOT / "models").glob("*.pt")) + list((PROJECT_ROOT / "runs").glob("*/weights/best.pt")))
    seg = lambda p: "seg" in p.name or "segment" in p.parent.parent.name  # noqa: E731
    sel = [p for p in found if seg(p) == (kind == "segment")]
    return [str(p.relative_to(PROJECT_ROOT)) for p in sel] or [CFG["models"]["detection" if kind == "detect" else "segmentation"]["weights"]]


@lru_cache(maxsize=6)
def get_model(kind: str, weights: str, device: str):
    return build_model(load_config(None, {"device": device}), kind, weights)


def _models(choice: str, det_w: str, seg_w: str, device: str):
    kinds = {"Object detection": [("detect", det_w)], "Instance segmentation": [("segment", seg_w)],
             "Compare both": [("detect", det_w), ("segment", seg_w)]}[choice]
    try:
        return [(k, get_model(k, w, device)) for k, w in kinds]
    except (ModelLoadError, RuntimeError) as e:
        raise gr.Error(str(e))


def count_image(img_rgb, choice, det_w, seg_w, conf, iou, imgsz, pre, device):
    if img_rgb is None:
        raise gr.Error("Upload an image first.")
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    outs, rows, files = [], [], []
    for kind, model in _models(choice, det_w, seg_w, device):
        if id(model) not in _WARM:                     # first call per model absorbs CUDA warm-up so timings are fair
            run_model(model, bgr, conf, iou, int(imgsz), pre)
            _WARM.add(id(model))
        t0 = time.perf_counter()
        dets = run_model(model, bgr, conf, iou, int(imgsz), pre, CFG["inference"]["max_det"])
        ms = (time.perf_counter() - t0) * 1e3
        res = count_detections(dets, CFG["classes"], conf)
        vis = draw_detections(bgr, [d for d in dets if d.conf >= conf], counts_header(res.total, res.per_class), index_instances=kind == "segment")
        p = OUT / f"annotated_{kind}.png"
        cv2.imwrite(str(p), vis)
        files.append(str(p))
        outs.append((cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"{model.name}: {res.total} fruits"))
        rows.append({"approach": kind, "model": model.name, "total": res.total, **res.per_class, "inference ms": round(ms, 1)})
    df = pd.DataFrame(rows)
    csv = OUT / "counts.csv"
    df.to_csv(csv, index=False)
    note = "Predictions from the model, not verified ground truth. " + ("; ".join(f"{r['approach']}: {r['total']}" for r in rows))
    return outs, df, note, files + [str(csv)]


def count_video(path, choice, det_w, seg_w, conf, iou, imgsz, mode, tracker, stride, max_frames, device):
    if not path:
        raise gr.Error("Upload a video first.")
    kind, model = _models("Instance segmentation" if choice == "Instance segmentation" else "Object detection", det_w, seg_w, device)[0]
    tracking = mode.startswith("Unique")
    try:
        fn = make_frame_tracker(model, tracker, conf, iou, int(imgsz)) if tracking else \
            (lambda f: model.predict(f, conf, iou, int(imgsz), CFG["inference"]["max_det"]))
        s = process_video(path, fn, CFG["classes"], conf, tracking, CFG["tracking"]["min_track_frames"],
                          OUT / "counted.mp4", OUT / "video_summary.json", int(stride), int(max_frames) or None)
    except (InvalidInputError, ValueError, RuntimeError) as e:
        raise gr.Error(str(e))
    df = pd.DataFrame({"frame": range(len(s["frame_counts"])), "visible fruits": s["frame_counts"]})
    csv = OUT / "frame_counts.csv"
    df.to_csv(csv, index=False)
    uniq = "n/a (tracking off)" if s["unique_tracked"] is None else f"{s['unique_tracked']} (estimate)"
    md = (f"**Frames processed:** {s['frames_processed']}  \n**Visible per frame:** mean {s['visible_per_frame']['mean']:.1f}, max {s['visible_per_frame']['max']}  \n"
          f"**Unique tracked fruit:** {uniq}  \n\n_{s['note']}_")
    return str(OUT / "counted.mp4"), md, df, [str(OUT / "counted.mp4"), str(csv), str(OUT / "video_summary.json")]


def load_reports():
    rep = resolve_path(CFG["paths"]["reports"])
    md = rep / "benchmark.md"
    return md.read_text() if md.is_file() else "No evaluation report yet. Run `python scripts/evaluate.py` (see README)."


def build() -> gr.Blocks:
    det_opts, seg_opts = weight_choices("detect"), weight_choices("segment")
    with gr.Blocks(title="Fruit counting") as demo:
        gr.Markdown("# Fruit counting\nDetection vs instance segmentation, per-frame and tracked video counts.")
        with gr.Accordion("Model & thresholds", open=True):
            with gr.Row():
                choice = gr.Radio(["Object detection", "Instance segmentation", "Compare both"], value="Compare both", label="Approach")
                device = gr.Dropdown(["auto", "cpu", "cuda:0"], value="auto", label="Device")
            with gr.Row():
                det_w = gr.Dropdown(det_opts, value=det_opts[0], label="Detection weights", allow_custom_value=True)
                seg_w = gr.Dropdown(seg_opts, value=seg_opts[0], label="Segmentation weights", allow_custom_value=True)
            with gr.Row():
                conf = gr.Slider(0.05, 0.95, float(CFG["inference"]["conf"]), step=0.05, label="Confidence")
                iou = gr.Slider(0.1, 0.95, float(CFG["inference"]["iou"]), step=0.05, label="NMS IoU")
                imgsz = gr.Dropdown([320, 480, 640, 800, 960, 1280], value=int(CFG["inference"]["imgsz"]), label="Image size")
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    inp = gr.Image(type="numpy", label="Upload image", sources=["upload", "clipboard", "webcam"])
                    gal = gr.Gallery(label="Annotated result(s)", columns=2, height=420)
                pre = gr.Dropdown(["none", "clahe", "gamma", "tile"], value="none", label="Preprocessing (clahe/gamma: lighting, tile: small fruit)")
                btn = gr.Button("Count fruit", variant="primary")
                note = gr.Markdown()
                table = gr.Dataframe(label="Counts (total and per class, inference time)")
                dl = gr.File(label="Downloads: annotated image(s) + counts CSV", file_count="multiple")
                btn.click(count_image, [inp, choice, det_w, seg_w, conf, iou, imgsz, pre, device], [gal, table, note, dl])
            with gr.Tab("Video"):
                gr.Markdown("**Visible per frame** = fruit on screen in each frame. **Unique tracked** = distinct tracker IDs over the clip: an "
                            "estimate that over-counts after long occlusions and under-counts on ID switches. They are reported separately.")
                with gr.Row():
                    vin = gr.Video(label="Upload video")
                    vout = gr.Video(label="Processed video")
                with gr.Row():
                    mode = gr.Radio(["Per-frame counting", "Unique counting (tracking)"], value="Per-frame counting", label="Mode")
                    tracker = gr.Dropdown(["bytetrack", "iou"], value="bytetrack", label="Tracker")
                    stride = gr.Number(1, label="Process every Nth frame", precision=0)
                    maxf = gr.Number(300, label="Max frames (0 = all)", precision=0)
                vbtn = gr.Button("Process video", variant="primary")
                vmd = gr.Markdown()
                vtab = gr.LinePlot(x="frame", y="visible fruits", label="Visible fruit per frame")
                vdl = gr.File(label="Downloads: processed video, frame counts CSV, summary JSON", file_count="multiple")
                vbtn.click(count_video, [vin, choice, det_w, seg_w, conf, iou, imgsz, mode, tracker, stride, maxf, device], [vout, vmd, vtab, vdl])
            with gr.Tab("Evaluation report"):
                rep = gr.Markdown(load_reports())
                gr.Button("Reload").click(load_reports, None, rep)
                csvs = sorted(resolve_path(CFG["paths"]["reports"]).glob("*.csv"))
                gr.File(value=[str(c) for c in csvs] or None, label="CSV reports (download)", file_count="multiple")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build().queue().launch(server_name=a.host, server_port=a.port, share=a.share)
