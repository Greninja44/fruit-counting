#!/usr/bin/env python
"""Evaluate detection and/or segmentation on the annotated test split and write the benchmark report.

    python scripts/evaluate.py --approaches detect segment                       # headline comparison
    python scripts/evaluate.py --approaches detect segment --preprocess none clahe gamma tile   # robustness
    python scripts/evaluate.py --approaches detect --weights runs/fruit/weights/best.pt          # fine-tuned model

Predictions are cached in outputs/predictions/ (re-used unless --force) and scored by one shared code path.
Ground truth is read only from data/processed/<dataset>/labels*/; predictions are never written there.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fruitcount.data.dataset import FruitDataset  # noqa: E402
from fruitcount.detection.base import ModelLoadError  # noqa: E402
from fruitcount.detection.factory import build_model  # noqa: E402
from fruitcount.evaluation import error_analysis as EA  # noqa: E402
from fruitcount.evaluation.predictions import load_predictions, save_predictions  # noqa: E402
from fruitcount.evaluation.report import NA, comparison_row, to_markdown  # noqa: E402
from fruitcount.evaluation.runner import evaluate_run, evaluate_subsets, predict_dataset, usable_samples  # noqa: E402
from fruitcount.evaluation.tags import AUTO_TAGS, MANUAL_ONLY_TAGS, apply_manual_overrides  # noqa: E402
from fruitcount.utils.config import load_config, resolve_path  # noqa: E402
from fruitcount.utils.io import describe_hardware, read_image, resolve_device  # noqa: E402
from fruitcount.utils.log import get_logger  # noqa: E402
from fruitcount.visualization.draw_results import counts_header, draw_detections, draw_error_overlay  # noqa: E402

log = get_logger("evaluate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--approaches", nargs="+", default=["detect", "segment"], choices=["detect", "segment"])
    ap.add_argument("--weights", nargs="*", default=None, help="one weights file per approach, same order (optional)")
    ap.add_argument("--preprocess", nargs="+", default=["none"], choices=["none", "clahe", "gamma", "tile"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--conf", type=float, default=None, help="operating confidence for counting / P / R / F1")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--include-crowd-images", action="store_true", help="do not exclude images with COCO crowd regions")
    ap.add_argument("--force", action="store_true", help="re-run inference even if a prediction cache exists")
    ap.add_argument("--examples", type=int, default=6, help="example images per error category / approach")
    ap.add_argument("--tag", default="", help="suffix for output names (e.g. 'finetuned')")
    a = ap.parse_args()

    cfg = load_config(a.config, {"inference.conf": a.conf, "inference.iou": a.iou, "inference.imgsz": a.imgsz,
                                 "device": a.device, "evaluation.split": a.split})
    inf, ev_cfg, classes = cfg["inference"], cfg["evaluation"], cfg["classes"]
    root = resolve_path(ev_cfg["dataset_root"])
    pred_dir, rep_dir, vis_dir = (resolve_path(cfg["paths"][k]) for k in ("predictions", "reports", "visualizations"))
    rep_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""

    try:
        ds = FruitDataset(root, ev_cfg["split"], classes)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2
    samples, n_crowd = usable_samples(ds, not a.include_crowd_images)
    n_gt = sum(s.count() for s in samples)
    if not samples or n_gt == 0:
        msg = "PENDING ANNOTATIONS: the split has no ground-truth fruit labels; no metrics were computed."
        log.error(msg)
        (rep_dir / f"benchmark{sfx}.md").write_text(f"# Benchmark report\n\n{msg}\n")
        return 3
    log.info("split '%s': %d images used (%d excluded for crowd regions), %d GT fruits", ds.split, len(samples), n_crowd, n_gt)

    device = resolve_device(cfg.get("device", "auto"))
    hw = describe_hardware(device)
    tag_file = resolve_path("data/annotations") / f"condition_tags_auto_{root.name}_{ds.split}.csv"
    tags_auto = pd.read_csv(tag_file) if tag_file.is_file() else None
    if tags_auto is None:
        log.warning("no condition tags yet; run scripts/tag_conditions.py for robustness subsets")
    else:
        tags_auto = apply_manual_overrides(tags_auto, resolve_path("data/annotations/condition_tags.csv"))

    results, rows, robust_rows, err_summary = {}, [], [], []
    for ai, approach in enumerate(a.approaches):
        w = a.weights[ai] if a.weights and ai < len(a.weights) else None
        model = None
        for pre in a.preprocess:
            run = f"{approach}_{pre}{sfx}"
            pfile = pred_dir / f"{run}.json"
            if pfile.is_file() and not a.force:
                info, recs = load_predictions(pfile)
                log.info("[%s] using cached predictions %s", run, pfile.name)
            else:
                try:
                    model = model or build_model(cfg, approach, w)
                except (ModelLoadError, RuntimeError) as e:
                    log.error("%s", e)
                    return 2
                log.info("[%s] predicting %d images (conf>=%g, imgsz=%d, preprocess=%s)", run, len(samples), inf["eval_conf"], inf["imgsz"], pre)
                records, extra = predict_dataset(model, samples, inf["eval_conf"], inf["iou"], inf["imgsz"], pre, timing_conf=inf["conf"])
                info = {"run": run, "approach": approach, "model": model.name, "weights": str(model.weights.name),
                        "model_size_mb": model.model_size_mb(), "preprocess": pre, "imgsz": inf["imgsz"], "batch": 1,
                        "nms_iou": inf["iou"], "eval_conf": inf["eval_conf"], "hardware": hw, **extra,
                        "created": datetime.now().isoformat(timespec="seconds")}
                save_predictions(pfile, records, info)
                recs = {r.image: r for r in records}
            ev = evaluate_run(samples, recs, classes, inf["conf"], ev_cfg["match_iou"], ev_cfg["mape_zero_policy"],
                              masks=(approach == "segment"))
            ev["run_info"] = info
            results[run] = ev
            ev["per_image"].to_csv(rep_dir / f"per_image_{run}.csv", index=False)
            label = {"detect": "Object detection", "segment": "Instance segmentation"}[approach] + ("" if pre == "none" else f" + {pre}")
            rows.append(comparison_row(label, ev, info, info.get("model_size_mb")))
            c, d = ev["counting"], ev["detection"]
            log.info("[%s] MAE %.3f RMSE %.3f exact %.3f | P %.3f R %.3f F1 %.3f mAP50 %.3f mAP50-95 %.3f | %.1f ms",
                     run, c["MAE"], c["RMSE"], c["exact_count_accuracy"], d["precision"], d["recall"], d["F1"], d["mAP50"], d["mAP50_95"], ev["latency_ms_mean"])
            if tags_auto is not None:
                sub = evaluate_subsets(ev["per_image"], tags_auto, AUTO_TAGS)
                sub.insert(0, "run", run)
                robust_rows.append(sub)

            if pre == "none":      # error analysis + example images for the unmodified-input runs
                errs, items = [], {}
                for s in samples:
                    e, kept = EA.analyse_image(s, recs[s.image_path.name].dets, inf["conf"], ev_cfg["match_iou"])
                    errs.append(e)
                    items[s.image_path.name] = (s, e, kept)
                summ = EA.summarize(errs)
                err_summary.append({"run": run, "summary": summ})
                _save_error_examples(run, errs, items, vis_dir, a.examples)
                _save_gallery(run, samples, recs, inf["conf"], vis_dir, cfg["seed"], approach)

    comp = pd.DataFrame(rows)
    comp.to_csv(rep_dir / f"benchmark{sfx}.csv", index=False)
    robust = pd.concat(robust_rows) if robust_rows else pd.DataFrame()
    if len(robust):
        robust.to_csv(rep_dir / f"robustness{sfx}.csv", index=False)
    serial = {r: {k: v for k, v in e.items() if k != "per_image"} for r, e in results.items()}
    (rep_dir / f"metrics{sfx}.json").write_text(json.dumps({"hardware": hw, "images_evaluated": len(samples),
        "images_excluded_crowd": n_crowd, "gt_fruits": n_gt, "results": serial, "error_summary": err_summary}, indent=2, default=str))
    _write_markdown(rep_dir / f"benchmark{sfx}.md", cfg, ds.split, len(samples), n_crowd, n_gt, hw, comp, robust, err_summary, results)
    log.info("report: %s", rep_dir / f"benchmark{sfx}.md")
    return 0


def _save_error_examples(run, errs, items, vis_dir: Path, n: int) -> None:
    base = vis_dir / "errors" / run
    for cat in EA.ALL_CATS:
        hit = [e for e in errs if cat in e.categories]
        hit.sort(key=lambda e: -sum(1 for c in list(e.pred_cat.values()) + list(e.gt_cat.values()) if c == cat))
        (base / cat).mkdir(parents=True, exist_ok=True)
        for f in (base / cat).glob("*.jpg"):
            f.unlink()
        for e in hit[:n]:
            s, _, kept = items[e.name + s_ext(items, e.name)]
            img = read_image(s.image_path)
            vis = draw_error_overlay(img, [g.box for g in s.instances], {j: c for j, c in e.gt_cat.items()}, kept, e.pred_cat,
                                     f"GT {e.gt_count} / pred {e.pred_count}  [{cat}]")
            cv2.imwrite(str(base / cat / f"{e.name}.jpg"), vis)


def s_ext(items, stem: str) -> str:
    for k in (".jpg", ".jpeg", ".png"):
        if stem + k in items:
            return k
    raise KeyError(stem)


def _save_gallery(run, samples, recs, conf, vis_dir: Path, seed: int, approach: str, n: int = 6) -> None:
    out = vis_dir / "examples" / run
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(samples), size=min(n, len(samples)), replace=False):
        s = samples[i]
        kept = [d for d in recs[s.image_path.name].dets if d.conf >= conf]
        cnt = {}
        for d in kept:
            cnt[d.label] = cnt.get(d.label, 0) + 1
        vis = draw_detections(read_image(s.image_path), kept, counts_header(len(kept), cnt, f"Pred (GT {s.count()})"),
                              index_instances=approach == "segment")
        cv2.imwrite(str(out / f"{s.name}.jpg"), vis)


def _write_markdown(path, cfg, split, n_img, n_crowd, n_gt, hw, comp, robust, err_summary, results) -> None:
    inf, ev = cfg["inference"], cfg["evaluation"]
    L = ["# Benchmark report", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M}. All numbers below were measured by `scripts/evaluate.py`; nothing is hand-entered._", ""]
    L += ["## Evaluation configuration", "",
          f"- Split: `{split}` of `{ev['dataset_root']}`; {n_img} images, {n_gt} ground-truth fruits "
          f"({n_crowd} images with COCO crowd regions excluded: per-instance counts are undefined for them)",
          f"- Hardware: {', '.join(f'{k}={v}' for k, v in hw.items())}",
          f"- Image size {inf['imgsz']}, batch size 1, operating confidence {inf['conf']}, NMS IoU {inf['iou']}, "
          f"AP computed from predictions cached at conf {inf['eval_conf']}",
          f"- Matching: greedy by confidence, class-aware, IoU >= {ev['match_iou']}; AP 101-point; MAPE zero-GT policy `{ev['mape_zero_policy']}`",
          "- Latency: measured in a separate pass at the operating confidence; end-to-end per image (preprocessing + inference + NMS, excluding disk decode) after 3 warm-up runs, CUDA synchronised; FPS = 1000 / mean latency",
          "- Ground truth: COCO 2017 val human annotations (not re-verified by us; see docs/evaluation.md for caveats)", ""]
    L += ["## Headline comparison", "", to_markdown(comp), ""]
    peaks = {r: e["run_info"].get("peak_gpu_mem_mb") for r, e in results.items()}
    L += ["Peak GPU memory (torch allocator, includes model weights): " + ", ".join(f"{r}: {NA if v is None else f'{v:.0f} MB'}" for r, v in peaks.items()), ""]
    L += ["## Counting detail", "", to_markdown(pd.DataFrame([{
        "Run": r, "MAE": f"{e['counting']['MAE']:.3f}", "RMSE": f"{e['counting']['RMSE']:.3f}",
        f"MAPE % ({e['counting']['MAPE_policy']})": f"{e['counting']['MAPE_percent']:.1f}",
        "Exact": f"{e['counting']['exact_count_accuracy']:.3f}", "Within +-1": f"{e['counting']['within_one_accuracy']:.3f}",
        "Mean signed error": f"{e['counting']['mean_signed_error']:+.3f}"} for r, e in results.items()])), "",
        "Per-image counts and absolute errors: `outputs/reports/per_image_<run>.csv`.", ""]
    if len(robust):
        L += ["## Robustness by condition", "",
              "Condition tags are **heuristic proxies** computed from image statistics and GT geometry (see `fruitcount/evaluation/tags.py`); "
              "`occluded`, `cluttered_background` and `unusual_viewpoint` need manual tags and are *Pending annotations*. "
              "Subsets with < 5 images are not reported.", ""]
        r2 = robust.copy()
        for col in [c for c in ("MAE", "RMSE", "exact_acc", "mean_signed_error", "precision", "recall", "F1") if c in r2]:
            r2[col] = r2[col].map(lambda v: NA if pd.isna(v) else f"{v:.3f}")
        L += [to_markdown(r2.fillna("")), ""]
        L += ["Manual-only conditions: " + ", ".join(f"`{t}`" for t in MANUAL_ONLY_TAGS) + " - Pending annotations.", ""]
    L += ["## Error categories (unmodified input)", ""]
    for es in err_summary:
        L += [f"**{es['run']}**", "", to_markdown(pd.DataFrame([{"category": c, "instances": v["instances"], "images": v["images"]} for c, v in es["summary"].items()])), ""]
    L += ["Example images per category: `outputs/visualizations/errors/<run>/<category>/`; representative predictions: `outputs/visualizations/examples/<run>/`.", ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
