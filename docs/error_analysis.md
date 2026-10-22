# Error analysis (measured on COCO val2017 apple/orange/banana, 187 images / 690 fruits)

Source numbers: `outputs/reports/benchmark.md`, `per_image_*.csv`, `robustness.csv`; example images in
`outputs/visualizations/errors/<run>/<category>/`. Models: COCO-pretrained YOLO11n (detect) and YOLO11n-seg (segment),
imgsz 640, conf 0.25, NMS IoU 0.7, no fine-tuning. Ground truth is COCO's human annotation, so categories below
are *confirmed against that GT* (an "FP" can still be an unlabelled real fruit). Condition tags are heuristic proxies.

## Headline

* Both approaches **under-count heavily**: total predicted 414 (detect) and 398 (segment) vs 690 ground-truth fruits;
  112 / 108 of 187 images are under-counted, 36 / 33 over-counted. Mean signed error -1.5 fruits per image.
* **Segmentation was not reliably better at counting.** MAE 2.44 vs 2.54, exact-count 0.246 vs 0.209, but a paired bootstrap
  of per-image error (`scripts/compare_runs.py detect_none segment_none`) gives a difference of +0.10 with 95% CI [-0.13, +0.31]:
  not distinguishable from zero on 187 images. Box mAP50 is slightly *lower* for the seg model (0.332 vs 0.345).
  Mask quality of matched fruit is good (mean mask IoU 0.85) but mask mAP50-95 (0.214) is lower than box mAP50-95 (0.236),
  partly because GT polygons are coarse.
* Cost of segmentation here: ~30% more latency at batch 1 (44 vs 34 ms), ~2.5x peak GPU memory, 10% larger weights.

## Where the counting error comes from

Missed fruit dominates (the table counts GT fruits that no prediction matched at IoU 0.5; categories are exclusive):

| detect_none | GT instances | images |
|---|---|---|
| missed_small (<32x32 px) | 187 | 62 |
| missed_touching (overlapping another GT, or swallowed by a merged box) | 148 | 55 |
| missed_other | 136 | 76 |

Prediction-side errors are much smaller: false_positive 78, poor_localization 83, duplicate 12, merged 11, wrong_class 11.
Segment results are similar (see report). Duplicates are rare at NMS IoU 0.7, so duplicate detections are not a major
issue. 8 (detect) / 12 (segment) images have the right total for the wrong reasons (`right_total_wrong_objects`): count-only
metrics flatter the model there.

## Which conditions cause the largest errors (detect_none; segment is similar)

| condition (proxy tag) | images | MAE | mean signed error | recall |
|---|---|---|---|---|
| dense_cluster (>=10 fruits) | 16 | 8.44 | -7.69 | 0.24 |
| overlapping | 68 | 4.22 | -3.22 | 0.29 |
| small_fruit | 44 | 3.34 | -3.21 | **0.08** |
| uneven_lighting | 46 | 2.54 | -1.33 | 0.34 |
| low_light | 9 | 2.56 | -0.78 | 0.42 |
| bright_sunlight | 20 | 1.30 | -0.40 | 0.47 |
| large_fruit | 87 | 1.40 | +0.41 | 0.63 |

* Size/crowding dominate: small fruit are almost never found (recall 0.08) and dense piles lose ~8 of ~14 fruits per image.
* Lighting effects are second-order *in this sample*: low-light (9 images) and uneven lighting are no worse than the overall
  MAE, but the subsets are small and heavily confounded with fruit size and scene type. `blurry` has only 4 images and is not scored.
* Occlusion, cluttered background and viewpoint need manual tags and remain **Pending**; `occluded_proxy` behaves like `overlapping`.

## Do the tested remedies help?

* **CLAHE / adaptive gamma**: no consistent gain. CLAHE nudges low-light MAE (2.56 -> 2.11 on 9 images) but lowers recall and
  mAP overall (0.345 -> 0.326); gamma is neutral. Do not enable them by default.
* **Tiled inference** (320 px tiles upsampled 2x + full image): recall rises (0.32 -> 0.43) and small-fruit recall doubles
  (0.075 -> 0.16), but precision halves (0.53 -> 0.24) and the count flips to *over*-counting (+3.1 per image), because
  large fruit get cut by tile borders and are detected as fragments that NMS cannot merge. Latency is 7x. Tiling needs
  large images with uniformly small fruit and a smarter merge (e.g. keep only tile detections away from borders, or
  cluster-merge fragments); it is a net loss on this mixed-size data.

## Suggested improvements (not yet tried)

1. **Fine-tune on target-domain data** (orchard/market images with exhaustive labels) on the GPU workstation: the biggest lever,
   since pretrained COCO weights are tuned for fruit as incidental objects. Include dense piles and small fruit.
2. Train/infer at higher resolution (imgsz 960-1280) or a larger model (yolo11s/m) for small fruit; compare against tiling
   with border-aware merging.
3. For dense clusters, sweep NMS IoU (0.5-0.8) and consider a density-map / count-calibration head for heavy occlusion.
4. Tune the confidence threshold on a validation split for counting (the F1-optimal threshold is lower than 0.25 given the
   recall deficit); report it separately from mAP.
5. Add fruit-free images to measure false positives on background, and get manual tags for occlusion/viewpoint/background.
6. For video, the tracker caveats below apply.

## Video demo (mechanics only, no ground truth)

`pan.mp4` was synthesised by panning a 640x480 window over a 2x-upscaled COCO banana image (60 frames); it is **not** a
real orchard video and no unique-fruit ground truth exists, so accuracy of the unique count is *not evaluated*. Observed
(scratch run, not committed): per-frame mode saw mean 7.3 / max 14 fruits per frame; ByteTrack mode reported 18 unique
tracked fruit with max 11 visible per frame. A count of unique tracks above the maximum simultaneously visible is expected when
fruit enter and leave the window, but it also shows that the number is sensitive to ID fragmentation; the two numbers are
different quantities and are always reported separately.

## Inference-time tuning (no training)

`scripts/tune.py` swept model size {yolo11n, s, m} x imgsz {640, 960} x NMS IoU {0.5, 0.7} x confidence (8 values) on a
*separate* tuning set: the Open Images test split restricted to apple/banana/orange (233 images, 765 fruits; group-box
images excluded). Full grid: `outputs/reports/tuning_openimages_fruit_test_detect.csv`. The winner (by counting MAE,
then F1) was confirmed **once** on the untouched COCO test set (`outputs/reports/benchmark_tuned.md`):

| COCO test (187 images) | MAE | RMSE | exact | recall | box mAP50 | ms/image | peak GPU MB |
|---|---|---|---|---|---|---|---|
| yolo11n, conf 0.25, NMS 0.7 (baseline) | 2.54 | 3.97 | 0.209 | 0.317 | 0.345 | 34 | 51 |
| **yolo11m, conf 0.30, NMS 0.5, 640 px** | **2.18** | 3.71 | 0.321 | 0.381 | 0.423 | 40 | 117 |

* The MAE gain over the baseline is +0.36 with paired-bootstrap 95% CI [+0.06, +0.63] (`scripts/compare_runs.py detect_none detect_none_tuned`),
  i.e. real but modest; the model still under-counts (recall 0.38) - it is a pretrained-weights ceiling, fine-tuning is the next step.
* Larger input (960 px) did not help on the tuning set, and the best confidence differs per model (n: 0.20, s: 0.35, m: 0.30).
* Caveat: tuning used only 233 images, so close settings (e.g. NMS 0.5 vs 0.7) are not distinguishable.
* **Segmentation at medium size is identical to detection for counting.** The released `yolo11m-seg.pt` shares all 649 backbone/box-head
  tensors bit-for-bit with `yolo11m.pt` (verified with `torch.equal`) and only adds a mask head, so boxes, counts and box mAP are
  *identical by construction* (MAE 2.176 both; paired difference exactly 0). What segmentation adds here is masks
  (mask mAP50 0.419) at +15 ms (55 vs 40 ms) and +94 MB peak GPU memory (211 vs 117 MB). The earlier n-size gap between detect and seg
  came from the n checkpoints being trained differently, not from masks helping counting.
