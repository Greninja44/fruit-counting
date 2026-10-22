"""Build a model from config; the one place that knows how to pick weights and class maps."""
from __future__ import annotations

from fruitcount.detection.base import FruitModel
from fruitcount.detection.detector import YoloDetector
from fruitcount.detection.segmenter import YoloSegmenter
from fruitcount.utils.config import resolve_path


def build_model(cfg: dict, approach: str, weights: str | None = None) -> FruitModel:
    """``approach`` is 'detect' or 'segment'. Explicit ``weights`` (fine-tuned) skip the pretrained class map."""
    key = {"detect": "detection", "segment": "segmentation"}.get(approach)
    if key is None:
        raise ValueError(f"unknown approach '{approach}' (expected 'detect' or 'segment')")
    spec = cfg["models"][key]
    cls = YoloDetector if approach == "detect" else YoloSegmenter
    path = resolve_path(weights or spec["weights"])
    # The COCO name map is harmless for fine-tuned models too: their names already equal the canonical ones.
    cmap = {**{c: c for c in cfg["classes"]}, **cfg.get("pretrained_class_map", {})}
    return cls(path, cfg["classes"], cmap, cfg.get("device", "auto"), name=spec["name"] if not weights else path.stem)
