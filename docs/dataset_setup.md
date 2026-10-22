# Dataset setup

## What is used here

| | |
|---|---|
| Source | COCO 2017 (<https://cocodataset.org>), classes `apple`, `orange`, `banana` |
| Annotations | CC BY 4.0 (instance polygons + boxes, human-made) |
| Images | Flickr photos with per-image licences (kept in `data/processed/coco_fruit/meta.csv`, column `license`; mostly CC BY / CC BY-NC / CC BY-SA variants) |
| Used for | **Evaluation / test split only**: the 200 `val2017` images that contain at least one of the three classes |
| Not redistributed | Images and labels are downloaded by `scripts/prepare_data.py` into git-ignored `data/` |

The pre-trained COCO YOLO models were trained on `train2017`, so `val2017` is held out for them. Check the
image licences before using the images commercially or redistributing them; the annotation licence does not
cover the photos.

Limitations of this test set (they matter when reading the results):

* COCO labels are crowd-sourced and not exhaustive: unlabelled background fruit counts as a model "false positive".
* 13 of 200 images contain `iscrowd` regions (a blob covering many fruits). Per-instance counts are undefined
  there, so they are excluded from all metrics (`--include-crowd-images` overrides this).
* Every image contains at least one fruit, so false positives on fruit-free images are not measured.
  Add background images (see below) to measure them.
* It is not an orchard/farm dataset. Viewpoint and background variety is broad but not agricultural.

## Prepare the dataset

```bash
python scripts/prepare_data.py --coco-split val2017                  # -> data/processed/coco_fruit (test split)
python scripts/prepare_data.py --coco-split train2017 --max-images 1500   # optional seeded train/val (85/15)
python scripts/validate_dataset.py                                   # integrity + stats + leakage check
python scripts/tag_conditions.py                                     # condition tags (see evaluation.md)
```

Output layout (YOLO format, so the same folder trains with Ultralytics):

```
data/processed/coco_fruit/
  images/{train,val,test}/*.jpg
  labels/{...}/*.txt          cls cx cy w h              (detection)
  labels_seg/{...}/*.txt      cls x1 y1 x2 y2 ...        (instance polygons; same instance order as labels/)
  meta.csv                    image, split, source_id, size, has_crowd, per-class counts, licence, flickr_url
  dataset.yaml                for detection training
data/processed/coco_fruit_seg/dataset.yaml   for segmentation training (symlinks to the same images)
```

Classes are configured in `configs/default.yaml` (`classes:`). To add a fruit that COCO has (e.g. none other than
the three here) change the list; for fruit COCO lacks (mango, strawberry...) use your own data.

## Using your own dataset

1. Put images in `data/raw/<name>/` and annotate with any tool that exports YOLO detection (and optionally
   segmentation) labels. **Humans review and correct the labels**; keep model-generated pre-labels out of the
   test split unless every one is verified.
2. Arrange it as above under `data/processed/<name>/{images,labels,labels_seg}/<split>/` and write `meta.csv`
   with at least `image,split,source_id,width,height`.
3. **Avoid leakage**: frames from one video, or images from one orchard row/tree, must share a `source_id` and land
   in a single split. `fruitcount.data.dataset.split_by_group(groups, ratios, seed)` does this deterministically
   (tested in `tests/test_dataset.py`), and `validate_dataset.py` fails if a `source_id` spans splits.
4. Point `evaluation.dataset_root` in the config (or `--config`) at it, update `classes:`, and re-run validation.

Other public sources worth considering (check each licence before use): MinneApple (apple orchards, detection and
segmentation), Open Images v7 (apple/orange/banana boxes, CC BY), Roboflow Universe fruit datasets (per-dataset
licences; require an API key, so they are not scripted here).

COCO-format conversions: `scripts/prepare_data.py` is a worked COCO -> YOLO example (boxes and polygons).
Pascal VOC / CSV boxes are converted with `fruitcount.data.dataset.box_to_yolo`.

## Annotation review workflow

1. Run `evaluate.py`; inspect `outputs/visualizations/errors/<run>/<category>/` - green = human GT, red thick = missed GT,
   coloured boxes = predictions by error category.
2. When a "false positive" is really an unlabelled fruit, or a GT box is wrong, correct the label file
   (CVAT / Label Studio / labelImg), keep the change in a dated commit or a copy of `labels/`, and re-evaluate.
3. Never write predictions into `labels*/`; predictions live only in `outputs/predictions/*.json`.

## Multi-fruit data ("detect any fruit")

| Dataset | Script | What it adds | Licence |
|---|---|---|---|
| Open Images (V6 train boxes, V5 val/test) | `scripts/prepare_openimages.py` | 15 fruit classes, official splits | boxes CC BY 4.0, images CC BY 2.0 |
| COCO train2017 | `scripts/prepare_data.py --config configs/openimages.yaml --coco-split train2017 --target-split train --out data/processed/openimages_fruit` | exhaustive apple/orange/banana boxes + polygons | CC BY 4.0 / Flickr per image |
| LVIS v1 train | `scripts/prepare_lvis.py` | 14 more fruit types (apricot, avocado, berries, cherry, coconut, date, kiwi, lime, melon, papaya, persimmon ...) with polygons; COCO val2017 images excluded | CC BY 4.0 / Flickr per image |

Result (`data/processed/openimages_fruit`, 29 classes): 11,661 train / 428 val / 1,356 test images, ~56k train boxes. Class counts are very uneven
(orange/banana/apple/strawberry in the thousands; mango, fig, cantaloupe, date, persimmon in the low hundreds or less).
Known limits: Open Images group boxes (one box around many fruit) are dropped and their images flagged `has_crowd` (44% of the Open
Images test images, so only 761 are used for counting); LVIS/Open Images are not exhaustively labelled for every fruit in every image;
vegetable-like categories (tomato, cucumber, pepper, pumpkin) are excluded by default (`prepare_lvis.py --extra tomato` adds one).

**Class-agnostic view:** `python scripts/make_any_fruit.py` -> `data/processed/any_fruit` (one class `fruit`, duplicate boxes removed, extra
held-out split `test_coco`), config `configs/any_fruit.yaml`. A one-class detector learns "fruit-ness" and is the right model for
counting fruit types that were never labelled; use the 29-class model when you need the species.

**Training bundle (PC only):** `python scripts/make_pc_bundle.py` -> `~/fruit_pc_bundle.tar` (train+val only; the test splits never leave
this machine). `train_on_pc.py --any-fruit` trains the single-class model.

**Measured baseline (pretrained COCO yolo11m, no fine-tuning, `outputs/reports/benchmark_anyfruit_pretrained_*.md`):**
on the Open Images test split recall is 0.63 for its three trained fruit but only 0.29 for the 12 other types (strawberry 0.01,
grape 0.07, pineapple 0.08, watermelon 0.17, vs peach 0.74, pear 0.65). Fine-tuned numbers are *Pending* until training on the PC.
