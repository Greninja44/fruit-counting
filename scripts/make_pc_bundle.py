#!/usr/bin/env python
"""Package a processed dataset + a self-contained training script for a GPU workstation (no training happens here).

    python scripts/make_pc_bundle.py --root data/processed/openimages_fruit --out ~/fruit_pc_bundle.tar
Copy the .tar to the PC, extract, then follow the printed commands. Paths inside are relative, so no machine-specific paths leak.
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.utils.config import resolve_path  # noqa: E402

TRAIN_PY = '''"""Fine-tune YOLO11 on the bundled fruit dataset. Run on the GPU PC:
    pip install ultralytics   (RTX 50xx / Blackwell needs torch built for CUDA 12.8:
                               pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128)
    python train_on_pc.py --model yolo11m.pt --epochs 80                 # 29 fruit classes
    python train_on_pc.py --model yolo11m.pt --epochs 80 --any-fruit     # single class 'fruit' (counts unseen fruit types too)
Ultralytics downloads the pretrained --model automatically. Results: runs/<name>/weights/best.pt -> copy back to the Jetson models/ folder.
"""
import argparse, os, pathlib, shutil
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--any-fruit", action="store_true", help="train one class 'fruit' instead of the species list")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    here = pathlib.Path(__file__).resolve().parent
    root = here / ("dataset_any" if a.any_fruit else "dataset")
    if a.any_fruit:                       # labels come from the bundle; images are hard-linked from dataset/ (copied if linking fails)
        for sp in ("train", "val"):
            dst = root / "images" / sp
            if dst.is_dir():
                continue
            dst.mkdir(parents=True)
            for f in (here / "dataset" / "images" / sp).iterdir():
                try:
                    os.link(f, dst / f.name)
                except OSError:
                    shutil.copy2(f, dst / f.name)
    yaml = root / "dataset.yaml"
    yaml.write_text(yaml.read_text().replace("path: .", f"path: {root.as_posix()}"))      # absolute path for this machine
    YOLO(a.model).train(data=str(yaml), epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, workers=a.workers, seed=42, deterministic=True,
                        patience=20, project=str(here / "runs"), name=a.name or ("fruit_any" if a.any_fruit else "fruit29"), exist_ok=True)


if __name__ == "__main__":      # required on Windows: dataloader workers re-import this file
    main()
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/processed/openimages_fruit")
    ap.add_argument("--any-root", default="data/processed/any_fruit", help="class-agnostic labels (from make_any_fruit.py)")
    ap.add_argument("--out", default="fruit_pc_bundle.tar")
    a = ap.parse_args()
    root, anyroot = resolve_path(a.root), resolve_path(a.any_root)
    if not (root / "dataset.yaml").is_file():
        print(f"not a prepared dataset: {root}")
        return 2

    def yml(path: Path) -> bytes:
        keep = [ln for ln in path.read_text().splitlines() if not ln.startswith(("test:", "path:"))]
        return ("path: .\n" + "\n".join(keep) + "\n").encode()

    out = Path(a.out).expanduser()
    top = "fruit_pc_bundle"
    with tarfile.open(out, "w") as tar:                 # uncompressed: JPEGs do not compress further. Test split is NOT included.
        for sp in ("train", "val"):
            tar.add(root / "images" / sp, arcname=f"{top}/dataset/images/{sp}")
            tar.add(root / "labels" / sp, arcname=f"{top}/dataset/labels/{sp}")
            if (anyroot / "labels" / sp).is_dir():
                tar.add(anyroot / "labels" / sp, arcname=f"{top}/dataset_any/labels/{sp}")
        tar.add(root / "meta.csv", arcname=f"{top}/dataset/meta.csv")
        files = [("dataset/dataset.yaml", yml(root / "dataset.yaml")), ("train_on_pc.py", TRAIN_PY.encode())]
        if (anyroot / "dataset.yaml").is_file():
            files.append(("dataset_any/dataset.yaml", yml(anyroot / "dataset.yaml")))
        for name, data in files:
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)\n\nOn the PC (PowerShell):\n"
          f"  scp autogenbot@192.168.1.17:{out.resolve()} .\n  tar -xf {out.name}\n  cd fruit_pc_bundle\n"
          "  python train_on_pc.py --model yolo11m.pt --epochs 80              # 29 classes\n  python train_on_pc.py --model yolo11m.pt --epochs 80 --any-fruit  # one class: any fruit\n\nThen copy back:\n"
          "  scp runs\\fruit_any\\weights\\best.pt autogenbot@192.168.1.17:/home/autogenbot/fruit-counting/models/fruit_any_yolo11m.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
