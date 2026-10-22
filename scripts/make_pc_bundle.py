#!/usr/bin/env python
"""Package the orange-only dataset + a self-contained training script for a GPU workstation (no training happens here).

    python scripts/make_pc_bundle.py --out ~/orange_pc_bundle.tar
Copy the .tar to the PC, extract, then follow the printed commands. Paths inside are relative, so no machine-specific paths leak.
Only train and val are included; the test splits stay on this machine.
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.utils.config import resolve_path  # noqa: E402

TRAIN_PY = '''"""Fine-tune YOLO11 to detect oranges (single class) on the bundled dataset. Run on the GPU PC:
    pip install ultralytics   (RTX 50xx / Blackwell needs torch built for CUDA 12.8:
                               pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128)
    python train_on_pc.py --model yolo11m.pt --epochs 80
Ultralytics downloads the pretrained --model automatically. Result: runs/orange/weights/best.pt -> copy back to the Jetson models/ folder.
"""
import argparse, pathlib
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="orange")
    a = ap.parse_args()
    here = pathlib.Path(__file__).resolve().parent
    yaml = here / "dataset" / "dataset.yaml"
    yaml.write_text(yaml.read_text().replace("path: .", f"path: {(here / 'dataset').as_posix()}"))      # absolute path for this machine
    YOLO(a.model).train(data=str(yaml), epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, workers=a.workers, seed=42, deterministic=True,
                        patience=20, project=str(here / "runs"), name=a.name, exist_ok=True)


if __name__ == "__main__":      # required on Windows: dataloader workers re-import this file
    main()
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/processed/orange")
    ap.add_argument("--out", default="orange_pc_bundle.tar")
    a = ap.parse_args()
    root = resolve_path(a.root)
    if not (root / "dataset.yaml").is_file():
        print(f"not a prepared dataset: {root} (run scripts/make_orange.py)")
        return 2
    keep = [ln for ln in (root / "dataset.yaml").read_text().splitlines() if not ln.startswith(("test:", "path:"))]
    out, top = Path(a.out).expanduser(), "orange_pc_bundle"
    with tarfile.open(out, "w", dereference=True) as tar:      # dereference: images are symlinks here. Uncompressed: JPEGs do not shrink.
        for sp in ("train", "val"):
            tar.add(root / "images" / sp, arcname=f"{top}/dataset/images/{sp}")
            tar.add(root / "labels" / sp, arcname=f"{top}/dataset/labels/{sp}")
        for name, data in (("dataset/dataset.yaml", ("path: .\n" + "\n".join(keep) + "\n").encode()), ("train_on_pc.py", TRAIN_PY.encode())):
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)\n\nOn the PC (PowerShell):\n"
          f"  scp autogenbot@192.168.1.17:{out.resolve()} .\n  tar -xf {out.name}\n  cd {top}\n"
          "  python train_on_pc.py --model yolo11m.pt --epochs 80\n\nThen copy back:\n"
          "  scp runs\\orange\\weights\\best.pt autogenbot@192.168.1.17:/home/autogenbot/fruit-counting/models/orange_yolo11m.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
