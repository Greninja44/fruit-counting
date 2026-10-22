#!/usr/bin/env python
"""Benchmark inference speed (latency, FPS, peak GPU memory) for detection and segmentation models.

    python scripts/benchmark.py --approaches detect segment --batch-sizes 1 4 --runs 50

Uses real images from the configured split (cycled), warm-up runs, and CUDA synchronisation. Writes
outputs/reports/benchmark_speed.{json,md}. Accuracy is *not* measured here (see scripts/evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.data.dataset import FruitDataset  # noqa: E402
from fruitcount.detection.base import ModelLoadError  # noqa: E402
from fruitcount.detection.factory import build_model  # noqa: E402
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402
from fruitcount.utils.io import describe_hardware, read_image, resolve_device  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("benchmark")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approaches", nargs="+", default=["detect", "segment"], choices=["detect", "segment"])
    ap.add_argument("--weights", nargs="*", default=None)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    ap.add_argument("--runs", type=int, default=50, help="timed batches per setting")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--config", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_config(a.config, {"inference.conf": a.conf, "inference.imgsz": a.imgsz, "device": a.device})
    inf = cfg["inference"]
    try:
        device = resolve_device(cfg["device"])
        ds = FruitDataset(resolve_path(cfg["evaluation"]["dataset_root"]), cfg["evaluation"]["split"], cfg["classes"])
    except (RuntimeError, FileNotFoundError) as e:
        log.error("%s", e)
        return 2
    imgs = [read_image(p) for p in ds.image_paths[:32]]
    import torch

    cuda = device.startswith("cuda")
    out = {"hardware": describe_hardware(device), "imgsz": inf["imgsz"], "conf": inf["conf"], "nms_iou": inf["iou"], "results": []}
    for ai, ap_name in enumerate(a.approaches):
        try:
            model = build_model(cfg, ap_name, a.weights[ai] if a.weights and ai < len(a.weights) else None)
        except ModelLoadError as e:
            log.error("%s", e)
            return 2
        for bs in a.batch_sizes:
            def run(i: int) -> None:
                batch = [imgs[(i * bs + k) % len(imgs)] for k in range(bs)]
                if bs == 1:
                    model.predict(batch[0], inf["conf"], inf["iou"], inf["imgsz"])
                else:
                    model._yolo.predict(batch, conf=inf["conf"], iou=inf["iou"], imgsz=inf["imgsz"], **model._kw)

            for i in range(a.warmup):
                run(i)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for i in range(a.runs):
                t0 = time.perf_counter()
                run(i)
                if cuda:
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
            t = np.array(times)
            r = {"approach": ap_name, "model": model.name, "batch": bs, "latency_ms_per_image_mean": float(t.mean() / bs),
                 "latency_ms_per_image_p95": float(np.percentile(t, 95) / bs), "fps": float(1000 * bs / t.mean()),
                 "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6 if cuda else None, "model_size_mb": model.model_size_mb()}
            out["results"].append(r)
            log.info("%s bs=%d: %.1f ms/img (p95 %.1f), %.1f FPS, peak mem %s MB", ap_name, bs, r["latency_ms_per_image_mean"],
                     r["latency_ms_per_image_p95"], r["fps"], "n/a" if r["peak_gpu_mem_mb"] is None else f"{r['peak_gpu_mem_mb']:.0f}")
    rep = resolve_path(cfg["paths"]["reports"])
    rep.mkdir(parents=True, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""
    (rep / f"benchmark_speed{sfx}.json").write_text(json.dumps(out, indent=2))
    lines = ["# Speed benchmark", "", f"Hardware: {out['hardware']}  ", f"imgsz {inf['imgsz']}, conf {inf['conf']}, {a.runs} timed batches after {a.warmup} warm-up; inference + NMS + mask post-processing, no disk decode.", "",
             "| Approach | Model | Batch | ms/image (mean) | ms/image (p95) | FPS | Peak GPU mem (MB) | Size (MB) |", "|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        pm = "Not measurable (CPU)" if r["peak_gpu_mem_mb"] is None else f"{r['peak_gpu_mem_mb']:.0f}"
        lines.append(f"| {r['approach']} | {r['model']} | {r['batch']} | {r['latency_ms_per_image_mean']:.1f} | {r['latency_ms_per_image_p95']:.1f} | {r['fps']:.1f} | {pm} | {r['model_size_mb']:.1f} |")
    (rep / f"benchmark_speed{sfx}.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
