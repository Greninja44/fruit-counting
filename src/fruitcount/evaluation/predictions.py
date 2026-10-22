"""Prediction cache: model output is stored once (at a low confidence) and re-used by every evaluation.

This separates model-specific inference from evaluation logic and guarantees all approaches are
scored on exactly the same images with the same code. Predictions are never mixed with GT labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fruitcount.detection.base import Detection


@dataclass
class PredRecord:
    image: str
    width: int
    height: int
    ms: float
    dets: list[Detection] = field(default_factory=list)


def save_predictions(path: str | Path, records: list[PredRecord], run_info: dict) -> None:
    out = {"run": run_info, "images": [
        {"image": r.image, "width": r.width, "height": r.height, "ms": round(r.ms, 2),
         "dets": [{"box": [round(v, 1) for v in d.box], "conf": round(d.conf, 4), "cls": d.cls, "label": d.label,
                   "polygon": np.round(d.polygon, 1).tolist() if d.polygon is not None else None} for d in r.dets]}
        for r in records]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out))


def load_predictions(path: str | Path) -> tuple[dict, dict[str, PredRecord]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Prediction file not found: {p}. Run scripts/evaluate.py --predict first.")
    raw = json.loads(p.read_text())
    recs = {}
    for r in raw["images"]:
        dets = [Detection(tuple(d["box"]), d["conf"], d["cls"], d["label"],
                          np.asarray(d["polygon"]) if d["polygon"] else None) for d in r["dets"]]
        recs[r["image"]] = PredRecord(r["image"], r["width"], r["height"], r["ms"], dets)
    return raw["run"], recs
