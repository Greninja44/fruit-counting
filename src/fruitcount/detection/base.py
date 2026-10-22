"""Model-agnostic detection types and the interface every model backend implements."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Detection:
    """One predicted fruit. Coordinates are absolute pixels in the original image."""

    box: tuple[float, float, float, float]   # x1, y1, x2, y2
    conf: float
    cls: int                                  # index into the canonical class list
    label: str
    polygon: np.ndarray | None = None        # (N, 2) instance outline for segmentation models
    track_id: int | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


class ModelLoadError(RuntimeError):
    """Raised when weights are missing or cannot be loaded."""


class FruitModel(Protocol):
    """Interface used by counting, evaluation and the app; detection and segmentation both satisfy it."""

    name: str
    task: str                       # "detect" | "segment"
    classes: Sequence[str]

    def predict(self, image: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int = 300) -> list[Detection]: ...

    def model_size_mb(self) -> float: ...


def weights_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e6
