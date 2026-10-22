"""Shared Ultralytics-YOLO wrapper used by the detector and the segmenter."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fruitcount.detection.base import Detection, ModelLoadError, weights_size_mb
from fruitcount.utils.io import resolve_device
from fruitcount.utils.log import get_logger

log = get_logger(__name__)


class UltralyticsModel:
    """Wraps an Ultralytics model, mapping its class names onto the canonical fruit classes.

    Predictions of classes that do not map to a canonical fruit (e.g. COCO "person") are dropped,
    so a COCO-pretrained model can be used as a baseline without fine-tuning.
    """

    task = "detect"

    def __init__(
        self,
        weights: str | Path,
        classes: Sequence[str],
        class_map: Mapping[str, str] | None = None,
        device: str = "auto",
        name: str | None = None,
        half: bool | None = None,
    ):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise ModelLoadError(
                f"Model weights not found: {self.weights}. Download pretrained weights into models/ "
                "(see README) or pass --weights pointing at your fine-tuned checkpoint."
            )
        from ultralytics import YOLO

        try:
            self._yolo = YOLO(str(self.weights))
        except Exception as e:  # corrupt file, version mismatch...
            raise ModelLoadError(f"Could not load weights {self.weights}: {e}") from e
        self.name = name or self.weights.stem
        self.classes = list(classes)
        self.device = resolve_device(device)
        self.half = self.device.startswith("cuda") if half is None else half
        cmap = dict(class_map) if class_map is not None else {c: c for c in self.classes}
        self._id_map: dict[int, int] = {}
        for mid, mname in self._yolo.names.items():
            canon = cmap.get(mname)
            if canon in self.classes:
                self._id_map[int(mid)] = self.classes.index(canon)
        if not self._id_map:
            raise ModelLoadError(
                f"None of the model classes {list(self._yolo.names.values())[:10]}... map onto {self.classes}. "
                "Check pretrained_class_map in the config."
            )

    def model_size_mb(self) -> float:
        return weights_size_mb(self.weights)

    @property
    def _kw(self) -> dict:
        return {"device": self.device, "quantize": 16 if self.half else 32, "verbose": False}

    def _to_detections(self, result) -> list[Detection]:
        out: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
        polys = result.masks.xy if getattr(result, "masks", None) is not None else None
        for i in range(len(xyxy)):
            if cls[i] not in self._id_map:
                continue
            c = self._id_map[cls[i]]
            poly = np.asarray(polys[i], float) if polys is not None and len(polys[i]) >= 3 else None
            out.append(Detection(tuple(map(float, xyxy[i])), float(conf[i]), c, self.classes[c], poly,
                                 int(ids[i]) if ids is not None else None))
        return out

    def predict(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640, max_det: int = 300) -> list[Detection]:
        res = self._yolo.predict(image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, **self._kw)[0]
        return self._to_detections(res)

    def track(self, image: np.ndarray, conf: float, iou: float, imgsz: int, tracker: str = "bytetrack.yaml",
              persist: bool = True) -> list[Detection]:
        """Detect + associate with Ultralytics' ByteTrack/BoT-SORT. Call frame by frame with ``persist=True``."""
        res = self._yolo.track(image, conf=conf, iou=iou, imgsz=imgsz, tracker=tracker, persist=persist, **self._kw)[0]
        return self._to_detections(res)

    def reset_tracker(self) -> None:
        """Forget tracker state between videos."""
        self._yolo.predictor = None   # a fresh predictor builds a fresh tracker on the next call
