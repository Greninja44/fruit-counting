# Evaluation protocol

Everything is computed by `scripts/evaluate.py` from **cached predictions** (`outputs/predictions/<run>.json`)
and **human ground truth** (`data/processed/<dataset>/labels*`). Detection and segmentation models go through the
same loader, matching code and metric functions (`src/fruitcount/evaluation/`), so differences come from the models.

## Counting metrics (per image, then averaged)

* MAE = mean(|pred - gt|)
* RMSE = sqrt(mean((pred - gt)^2))
* MAPE = mean(|pred - gt| / gt) x 100. Images with **gt = 0** divide by zero; policy `mape_zero_policy` in the config:
  `skip` (default; excluded and the number of excluded images is reported) or `epsilon` (denominator `max(gt, 1)`).
* Exact-count accuracy = fraction of images with pred == gt; within-one accuracy = fraction with |pred - gt| <= 1
* Mean signed error (pred - gt): positive = over-counting, negative = under-counting
* The count is the number of predictions with confidence >= `inference.conf` (default 0.25, inclusive) after NMS.
  Predicted classes that do not map onto the configured fruit classes are discarded.

Per-image counts and absolute errors: `outputs/reports/per_image_<run>.csv`.

## Detection / segmentation metrics

* **Matching rule**: predictions in descending confidence; each is matched to the unmatched ground-truth instance of the
  *same class* with the highest IoU, if IoU >= 0.5 (`evaluation.match_iou`). Unmatched predictions = FP, unmatched GT = FN.
* Precision / recall / F1 are micro-averaged over images at the operating confidence and IoU 0.5.
* mAP@0.5 and mAP@0.5:0.95: class-wise 101-point interpolated AP (COCO style) from predictions cached at conf 0.001
  (`inference.eval_conf`), averaged over classes with ground truth. Cross-checked against pycocotools on the same
  predictions (`scripts/crosscheck_coco_eval.py`; difference < 0.001 when we ran it).
* **Box metrics are computed for both approaches** (segmentation models also output boxes), so the comparison is like for like.
* **Mask metrics** (segmentation only): the same procedure with mask IoU (polygons rasterised at image resolution),
  plus the mean mask IoU of matched pairs. Detectors have no masks, so they are *Not evaluated* for mask metrics.
* Ground-truth masks are COCO polygons (largest part of a multi-part mask); they are coarse, which caps attainable mask IoU.

## Performance metrics

Measured in a separate pass over the test images at the operating confidence, batch size 1, after 3 warm-up runs, with
CUDA synchronised: end-to-end latency per image (preprocess + inference + NMS + mask post-processing; file decoding is
excluded), FPS = 1000 / mean latency, peak GPU memory from `torch.cuda.max_memory_allocated` (includes weights; this is
the PyTorch allocator only, not total device memory), model size = weight file size.
`scripts/benchmark.py` adds batch-size sweeps with p95 latency. Hardware, image size, batch size and thresholds are written
into every report.

## Robustness conditions

`scripts/tag_conditions.py` writes `data/annotations/condition_tags_auto_<dataset>_<split>.csv` and, once, an empty manual template
`data/annotations/condition_tags.csv`. Tags per image (thresholds in `src/fruitcount/evaluation/tags.py`):

| tag | automatic proxy |
|---|---|
| small_fruit / large_fruit | median GT box area < 32^2 px / > 96^2 px |
| overlapping | any GT pair with box IoU > 0.10 |
| occluded_proxy | a GT box > 30% covered by one other GT box (geometric hint only) |
| dense_cluster | >= 10 GT fruits |
| bright_sunlight | mean gray > 165 or > 8% saturated pixels |
| low_light | mean gray < 70 |
| uneven_lighting | std of 4x4 block-mean gray > 40 (shadows) |
| blurry | variance of Laplacian < 100 (640 px long side) |
| occluded, cluttered_background, unusual_viewpoint | **manual only** (Pending annotations until tagged) |

To tag manually: open `condition_tags.csv`, write `1` or `0` in a tag column for an image; empty cells keep the automatic
value; manual-only columns default to 0 (untagged). These are heuristics, not human judgements. Subsets with fewer than 5
images are listed but not scored. Subsets overlap (one image can carry several tags), and they are small, so differences
between conditions are indicative, not statistically established.

Preprocessing variants (`--preprocess none clahe gamma tile`) are applied in memory at inference only; the test images are
never modified:

* `clahe`: CLAHE on the L channel (clip 2.0, 8x8 grid)
* `gamma`: adaptive gamma that brightens images darker than mean gray 0.45; brighter images are unchanged
* `tile`: sliced inference with half-size overlapping tiles (25% overlap) upsampled to the model size, plus the full image, merged by class-wise NMS

## Reading the results

* Ground truth is COCO's crowd-sourced annotation and is **not** exhaustive; unlabelled background fruit shows up as FP.
* The comparison uses one small test set (see `docs/dataset_setup.md`) and one run per configuration (inference is
  deterministic, but there is no confidence interval). Treat gaps of a few hundredths as noise.
* Counting metrics depend on the operating confidence. `evaluate.py --conf` re-scores the same cached predictions
  without re-running the models only if `eval_conf` <= the new value.
