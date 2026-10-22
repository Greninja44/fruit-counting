"""Single entry point that applies the configured preprocessing and runs a model."""
from __future__ import annotations

import numpy as np

from fruitcount.detection.base import Detection, FruitModel
from fruitcount.detection.preprocess import MODES, enhance_clahe, enhance_gamma, tile_predict


def run_model(model: FruitModel, image: np.ndarray, conf: float, iou: float, imgsz: int = 640,
              preprocess: str = "none", max_det: int = 300) -> list[Detection]:
    """Predict with an optional preprocessing mode (``none|clahe|gamma|tile``)."""
    if preprocess not in MODES:
        raise ValueError(f"unknown preprocess '{preprocess}', expected one of {MODES}")
    if preprocess == "clahe":
        image = enhance_clahe(image)
    elif preprocess == "gamma":
        image = enhance_gamma(image)

    def fn(im: np.ndarray) -> list[Detection]:
        return model.predict(im, conf, iou, imgsz, max_det)

    if preprocess == "tile":
        # Tiles are half the inference size so each tile is upsampled 2x: this is what helps small fruit.
        return tile_predict(fn, image, tile=max(imgsz // 2, 160), merge_iou=iou)
    return fn(image)
