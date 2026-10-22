"""Input validation helpers: images, videos, devices, hardware description."""
from __future__ import annotations

import platform
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class InvalidInputError(ValueError):
    """Raised for missing, unreadable or corrupt input files."""


def read_image(path: str | Path) -> np.ndarray:
    """Read a BGR image, raising :class:`InvalidInputError` with a clear message on failure."""
    p = Path(path)
    if not p.is_file():
        raise InvalidInputError(f"Image not found: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise InvalidInputError(f"Corrupt or unsupported image file: {p}")
    return img


def list_images(path: str | Path) -> list[Path]:
    """Return a single image or all images (sorted, non-recursive) in a directory."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    raise InvalidInputError(f"Path does not exist: {p}")


def open_video(path: str | Path) -> cv2.VideoCapture:
    """Open a video, validating that it exists and yields at least one frame."""
    p = Path(path)
    if not p.is_file():
        raise InvalidInputError(f"Video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise InvalidInputError(f"Cannot open video (corrupt or unsupported codec): {p}")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise InvalidInputError(f"Video contains no readable frames: {p}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def resolve_device(device: str = "auto") -> str:
    """Map 'auto' to cuda:0 when available, otherwise cpu; validate explicit requests."""
    import torch

    if device in ("auto", "", None):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{device}' requested but CUDA is not available; use --device cpu")
    return device


def describe_hardware(device: str) -> dict[str, str]:
    """Collect hardware/software info to report next to performance numbers."""
    import torch

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
