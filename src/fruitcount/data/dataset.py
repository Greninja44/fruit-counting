"""Dataset loading, YOLO annotation parsing and group-aware splitting.

On-disk layout (produced by ``scripts/prepare_data.py``)::

    <root>/images/<split>/*.jpg
    <root>/labels/<split>/*.txt        YOLO detection:    cls cx cy w h        (normalised)
    <root>/labels_seg/<split>/*.txt    YOLO segmentation: cls x1 y1 x2 y2 ...  (normalised)
    <root>/meta.csv                    per-image metadata (split, source_id, size, flags, ...)

Human ground truth lives only in ``labels*``; model predictions are never written there.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from fruitcount.utils.io import IMAGE_EXTS

SPLITS = ("train", "val", "test")


class AnnotationError(ValueError):
    """Raised when a label file cannot be parsed."""


@dataclass
class GTInstance:
    """One human-annotated fruit instance (absolute pixel coordinates)."""

    cls: int
    box: tuple[float, float, float, float]            # x1, y1, x2, y2
    polygon: np.ndarray | None = None                  # (N, 2) pixels, or None

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


@dataclass
class Sample:
    """An image with its ground-truth instances and metadata."""

    image_path: Path
    width: int
    height: int
    instances: list[GTInstance] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.image_path.stem

    def count(self, cls: int | None = None) -> int:
        return sum(1 for g in self.instances if cls is None or g.cls == cls)


def parse_yolo_label(text: str, width: int, height: int, num_classes: int | None = None) -> list[GTInstance]:
    """Parse YOLO detection (5 columns) or segmentation (>=7 columns, even count) label text.

    Raises :class:`AnnotationError` on malformed rows, out-of-range values or unknown class ids.
    """
    out: list[GTInstance] = []
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError as e:
            raise AnnotationError(f"line {ln}: non-numeric value ({e})") from e
        if num_classes is not None and not 0 <= cls < num_classes:
            raise AnnotationError(f"line {ln}: class id {cls} outside [0, {num_classes})")
        if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
            raise AnnotationError(f"line {ln}: coordinate outside [0, 1]")
        if len(vals) == 4:
            cx, cy, w, h = vals
            box = ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)
            out.append(GTInstance(cls, box))
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            poly = np.asarray(vals, dtype=np.float64).reshape(-1, 2) * np.array([width, height])
            box = (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())
            out.append(GTInstance(cls, box, poly))
        else:
            raise AnnotationError(f"line {ln}: expected 4 box values or an even number >=6 of polygon values, got {len(vals)}")
    return out


def box_to_yolo(box: Sequence[float], width: int, height: int) -> str:
    """Format an absolute xyxy box as normalised ``cx cy w h``."""
    x1, y1, x2, y2 = box
    return f"{(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def split_by_group(
    groups: Sequence[str], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> dict[str, str]:
    """Assign every *group* (video id, orchard row, source sequence...) to one split.

    All items sharing a group key land in the same split, which prevents leakage between
    frames of the same video. Deterministic for a given ``seed``. Returns ``{group: split}``.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    uniq = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    mapping = {}
    for i, g in enumerate(uniq):
        mapping[g] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return mapping


class FruitDataset:
    """A split of a YOLO-format fruit dataset with lazy-validated ground truth."""

    def __init__(self, root: str | Path, split: str, classes: Sequence[str], load_polygons: bool = True):
        self.root = Path(root)
        self.split = split
        self.classes = list(classes)
        img_dir = self.root / "images" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Split '{split}' not found at {img_dir}. Run scripts/prepare_data.py first.")
        self.image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        self.load_polygons = load_polygons
        meta_file = self.root / "meta.csv"
        self.meta = pd.read_csv(meta_file).set_index("image") if meta_file.is_file() else pd.DataFrame()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_labels(self, label_dir: str, stem: str, w: int, h: int) -> list[GTInstance] | None:
        f = self.root / label_dir / self.split / f"{stem}.txt"
        if not f.is_file():
            return None
        return parse_yolo_label(f.read_text(), w, h, len(self.classes))

    def get(self, idx: int) -> Sample:
        p = self.image_paths[idx]
        row = self.meta.loc[p.name].to_dict() if p.name in self.meta.index else {}
        if "width" in row and "height" in row:
            w, h = int(row["width"]), int(row["height"])
        else:  # fall back to reading the header
            import cv2

            img = cv2.imread(str(p))
            if img is None:
                raise AnnotationError(f"Unreadable image {p}")
            h, w = img.shape[:2]
        inst = self._read_labels("labels", p.stem, w, h)
        if inst is None:
            inst = []
        if self.load_polygons:
            seg = self._read_labels("labels_seg", p.stem, w, h)
            if seg is not None and len(seg) == len(inst):
                for g, s in zip(inst, seg):
                    g.polygon = s.polygon
        return Sample(p, w, h, inst, row)

    def __iter__(self) -> Iterable[Sample]:
        for i in range(len(self)):
            yield self.get(i)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for s in self:
            for g in s.instances:
                counts[self.classes[g.cls]] += 1
        return dict(counts)
