#!/usr/bin/env python
"""Fine-tune a detection or segmentation model on a YOLO-format fruit dataset.

    python scripts/train.py --approach detect  --data data/processed/coco_fruit/dataset.yaml --epochs 50 --device 0
    python scripts/train.py --approach segment --data data/processed/coco_fruit_seg/dataset.yaml

Checkpoints, logs (results.csv, plots) and the resolved args.yaml are written to runs/<name>/ by Ultralytics;
this script additionally copies the config it ran with. Use the resulting runs/<name>/weights/best.pt with
--weights in the predict/evaluate scripts. (Train on a machine with a capable GPU; see README.)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402
from fruitcount.utils.io import resolve_device  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402

log = get_logger("train")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approach", choices=["detect", "segment"], default="detect")
    ap.add_argument("--data", required=True, help="Ultralytics dataset.yaml")
    ap.add_argument("--weights", default=None, help="starting weights (default: pretrained from config)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="auto | cpu | 0 | cuda:0")
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()

    cfg = load_config(a.config, {"inference.imgsz": a.imgsz, "device": a.device})
    key = "detection" if a.approach == "detect" else "segmentation"
    weights = resolve_path(a.weights or cfg["models"][key]["weights"])
    data = Path(a.data)
    if not weights.is_file():
        log.error("starting weights not found: %s", weights)
        return 2
    if not data.is_file():
        log.error("dataset yaml not found: %s (run scripts/prepare_data.py)", data)
        return 2
    device = resolve_device(cfg["device"])
    from ultralytics import YOLO

    name = a.name or f"fruit_{a.approach}"
    log.info("training %s from %s on %s (epochs=%d imgsz=%d batch=%d seed=%d)", a.approach, weights.name, device, a.epochs, cfg["inference"]["imgsz"], a.batch, cfg["seed"])
    model = YOLO(str(weights))
    model.train(data=str(data), epochs=a.epochs, imgsz=cfg["inference"]["imgsz"], batch=a.batch, device=device.replace("cuda:", ""),
                seed=cfg["seed"], deterministic=True, project=str(resolve_path("runs")), name=name, patience=a.patience, exist_ok=True)
    run_dir = resolve_path("runs") / name
    shutil.copy(a.config or resolve_path("configs/default.yaml"), run_dir / "project_config.yaml")
    log.info("done. best weights: %s", run_dir / "weights" / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
