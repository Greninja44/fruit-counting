"""Approach A: bounding-box object detector."""
from fruitcount.detection.yolo_base import UltralyticsModel


class YoloDetector(UltralyticsModel):
    """YOLO detection model (e.g. yolo11n.pt, or a fine-tuned best.pt)."""

    task = "detect"
