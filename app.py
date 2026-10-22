"""Gradio app: live orange counting from a webcam attached to this machine (no image/video upload).

    python app.py [--port 7860] [--camera 0]

Shows oranges visible in the current frame and, with tracking on, the number of distinct oranges seen since Start (an estimate).
"""
from __future__ import annotations

import argparse
import threading
import time
from functools import lru_cache

import cv2
import gradio as gr

from fruitcount.counting.counter import VideoCounter, filter_detections
from fruitcount.detection.base import ModelLoadError
from fruitcount.detection.factory import build_model
from fruitcount.tracking.tracker import make_frame_tracker
from fruitcount.utils.config import PROJECT_ROOT, load_config, resolve_path
from fruitcount.visualization.draw_results import draw_detections

CFG = load_config()
CAMERA = 0


def weight_choices(kind: str) -> list[str]:
    """Weights found in models/ and runs/**/best.pt, split into detection / segmentation by filename (fine-tuned first)."""
    found = list((PROJECT_ROOT / "models").glob("*.pt")) + list((PROJECT_ROOT / "runs").glob("*/weights/best.pt"))
    seg = lambda p: "seg" in p.name or "segment" in p.parent.parent.name  # noqa: E731
    sel = sorted((p for p in found if seg(p) == (kind == "segment")), key=lambda p: ("orange" not in p.name and "orange" not in str(p.parent), p.name))
    return [str(p.relative_to(PROJECT_ROOT)) for p in sel] or [CFG["models"]["detection" if kind == "detect" else "segmentation"]["weights"]]


@lru_cache(maxsize=4)
def get_model(kind: str, weights: str, device: str):
    return build_model(load_config(None, {"device": device}), kind, weights)


class LatestFrame:
    """Reads the camera on a background thread and keeps only the newest frame, so slow inference never builds up lag."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap, self.frame, self.alive, self.lock = cap, None, True, threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                self.alive = False
                return
            with self.lock:
                self.frame = f

    def get(self):
        with self.lock:
            f, self.frame = self.frame, None
        return f


def live(kind, weights, conf, iou, imgsz, track, device):
    """Generator: stream annotated frames + stats until Stop is pressed or the camera disappears."""
    try:
        model = get_model("segment" if kind == "Instance segmentation" else "detect", weights, device)
    except (ModelLoadError, RuntimeError) as e:
        raise gr.Error(str(e))
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open camera {CAMERA}. Close any other program using it (e.g. another app or script) and try again.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    reader = LatestFrame(cap)
    classes, min_frames = CFG["classes"], CFG["tracking"]["min_track_frames"]
    vc, imgsz, fps, last = VideoCounter(classes, conf, min_frames), int(imgsz), None, time.perf_counter()
    fn = make_frame_tracker(model, "bytetrack", conf, iou, imgsz) if track else \
        (lambda f: model.predict(f, conf, iou, imgsz, CFG["inference"]["max_det"]))
    try:
        while reader.alive:
            frame = reader.get()
            if frame is None:
                time.sleep(0.005)
                continue
            dets = filter_detections(fn(frame), conf)
            res = vc.update(dets)
            now = time.perf_counter()
            fps = 1 / max(now - last, 1e-6) if fps is None else 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            lines = [f"Oranges in view: {res.total}"] + ([f"Distinct oranges since start: {vc.unique_tracked()} (estimate)"] if track else [])
            vis = draw_detections(frame, dets, lines, index_instances=kind == "Instance segmentation")
            md = f"### Oranges in view: **{res.total}**  \nSpeed: {fps:.1f} FPS"
            if track:
                md += (f"  \n### Distinct oranges since Start: **{vc.unique_tracked()}** _(tracker estimate: can over-count after long occlusions, "
                       "under-count on ID switches; it is not the sum of per-frame counts)_")
            yield cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), md
        if not reader.alive:
            raise gr.Error("Camera stopped delivering frames.")
    finally:
        reader.alive = False
        cap.release()


def load_reports():
    rep = resolve_path(CFG["paths"]["reports"])
    md = rep / "benchmark.md"
    return md.read_text() if md.is_file() else "No evaluation report yet. Run `python scripts/evaluate.py --config configs/default.yaml` (see README)."


def build() -> gr.Blocks:
    det_opts, seg_opts = weight_choices("detect"), weight_choices("segment")
    with gr.Blocks(title="Orange counting") as demo:
        gr.Markdown("# Orange counting (live camera)\nInput is the webcam attached to this machine. Only oranges are counted.")
        with gr.Row():
            kind = gr.Radio(["Object detection", "Instance segmentation"], value="Object detection", label="Approach")
            device = gr.Dropdown(["auto", "cpu", "cuda:0"], value="auto", label="Device")
            track = gr.Checkbox(True, label="Track oranges (distinct count)")
        with gr.Row():
            det_w = gr.Dropdown(det_opts, value=det_opts[0], label="Detection weights", allow_custom_value=True)
            seg_w = gr.Dropdown(seg_opts, value=seg_opts[0], label="Segmentation weights", allow_custom_value=True)
        with gr.Row():
            conf = gr.Slider(0.05, 0.95, float(CFG["inference"]["conf"]), step=0.05, label="Confidence")
            iou = gr.Slider(0.1, 0.95, float(CFG["inference"]["iou"]), step=0.05, label="NMS IoU")
            imgsz = gr.Dropdown([320, 480, 640, 800, 960], value=int(CFG["inference"]["imgsz"]), label="Image size")
        with gr.Row():
            start = gr.Button("Start camera", variant="primary")
            stop = gr.Button("Stop")
        view = gr.Image(label="Live", type="numpy", interactive=False)
        stats = gr.Markdown()
        w = lambda k, d, s: s if k == "Instance segmentation" else d  # noqa: E731
        ev = start.click(lambda k, d, s, *r: live(k, w(k, d, s), *r), [kind, det_w, seg_w, conf, iou, imgsz, track, device], [view, stats])
        stop.click(None, None, None, cancels=[ev])
        with gr.Accordion("Evaluation report", open=False):
            rep = gr.Markdown(load_reports())
            gr.Button("Reload").click(load_reports, None, rep)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    CAMERA = a.camera
    build().queue().launch(server_name=a.host, server_port=a.port, share=a.share)
