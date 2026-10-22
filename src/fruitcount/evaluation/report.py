"""Markdown / table rendering for the benchmark report. Only measured values are rendered."""
from __future__ import annotations

from typing import Mapping

import pandas as pd

NA = "Not evaluated"


def _f(v, nd=3, fmt=None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return NA
    return f"{v:{fmt}}" if fmt else f"{v:.{nd}f}"


def comparison_row(label: str, ev: Mapping, run: Mapping, size_mb: float | None) -> dict:
    """One row of the headline comparison table (columns fixed by the project spec)."""
    d, c = ev["detection"], ev["counting"]
    metric = f"box mAP50 {_f(d['mAP50'])} / mAP50-95 {_f(d['mAP50_95'])}; P {_f(d['precision'])} R {_f(d['recall'])} F1 {_f(d['F1'])}"
    if "mask" in ev:
        m = ev["mask"]
        metric += f"; mask mAP50 {_f(m['mAP50'])} / mAP50-95 {_f(m['mAP50_95'])}, matched mask IoU {_f(m['mean_matched_mask_iou'])}"
    lat = ev["latency_ms_mean"]
    return {
        "Approach": label,
        "Counting MAE": _f(c["MAE"]), "Counting RMSE": _f(c["RMSE"]),
        "Exact-count accuracy": _f(c["exact_count_accuracy"]),
        "Detection/segmentation metric": metric,
        "Latency (ms/image)": _f(lat, fmt=".1f"), "FPS": _f(1000 / lat if lat else None, fmt=".1f"),
        "Model size": f"{size_mb:.1f} MB" if size_mb else NA,
    }


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and v != v) else str(v) for v in r) + " |")
    return "\n".join(lines)
