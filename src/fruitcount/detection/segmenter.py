"""Approach B: instance segmentation (one mask polygon per fruit)."""
from fruitcount.detection.yolo_base import UltralyticsModel


class YoloSegmenter(UltralyticsModel):
    """YOLO instance-segmentation model (e.g. yolo11n-seg.pt). Detections carry ``polygon`` masks."""

    task = "segment"
