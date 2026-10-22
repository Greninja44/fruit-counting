import cv2
import numpy as np
import pytest

from fruitcount.data.dataset import AnnotationError, FruitDataset, box_to_yolo, parse_yolo_label, split_by_group
from fruitcount.data.validate_dataset import check_leakage, validate_split
from fruitcount.evaluation.runner import usable_samples
from conftest import CLASSES


def test_parse_detection_label():
    g = parse_yolo_label("1 0.5 0.5 0.2 0.4\n", 200, 100)
    assert len(g) == 1 and g[0].cls == 1
    assert g[0].box == pytest.approx((80, 30, 120, 70))


def test_parse_segmentation_label_gives_polygon_and_box():
    g = parse_yolo_label("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5", 100, 100)
    assert g[0].polygon.shape == (4, 2) and g[0].box == pytest.approx((10, 10, 50, 50))


@pytest.mark.parametrize("bad", ["0 0.5 0.5 0.2", "x 0.1 0.1 0.1 0.1", "0 1.5 0.5 0.2 0.2", "7 0.5 0.5 0.1 0.1", "0 0.1 0.1 0.2 0.2 0.3"])
def test_parse_rejects_malformed(bad):
    with pytest.raises(AnnotationError):
        parse_yolo_label(bad, 100, 100, num_classes=3)


def test_blank_label_file_means_no_fruit():
    assert parse_yolo_label("\n\n", 10, 10) == []


def test_box_roundtrip():
    s = box_to_yolo((80, 30, 120, 70), 200, 100)
    assert parse_yolo_label(f"0 {s}", 200, 100)[0].box == pytest.approx((80, 30, 120, 70), abs=0.01)


def test_dataset_loading(tiny_dataset):
    ds = FruitDataset(tiny_dataset, "test", CLASSES)
    assert len(ds) == 3
    a = ds.get(0)
    assert a.count() == 2 and a.count(0) == 1 and a.instances[0].polygon is not None
    assert ds.get(1).count() == 0                         # background image
    assert ds.class_counts() == {"apple": 1, "orange": 1, "banana": 1}


def test_missing_split_raises_helpful_error(tiny_dataset):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        FruitDataset(tiny_dataset, "val", CLASSES)


def test_crowd_images_excluded_from_evaluation(tiny_dataset):
    ds = FruitDataset(tiny_dataset, "test", CLASSES)
    keep, skipped = usable_samples(ds)
    assert skipped == 1 and [s.name for s in keep] == ["a", "b"]
    assert len(usable_samples(ds, exclude_crowd=False)[0]) == 3


def test_split_by_group_has_no_leakage_and_is_reproducible():
    groups = [f"video{i % 10}" for i in range(100)]     # 10 videos x 10 frames
    m1, m2 = split_by_group(groups, seed=7), split_by_group(groups, seed=7)
    assert m1 == m2 and split_by_group(groups, seed=8) != m1
    frame_split = [m1[g] for g in groups]
    for g in set(groups):
        assert len({s for gg, s in zip(groups, frame_split) if gg == g}) == 1
    assert set(m1.values()) == {"train", "val", "test"}
    with pytest.raises(ValueError):
        split_by_group(groups, (0.5, 0.5, 0.5))


def test_validate_detects_corrupt_image_and_bad_label(tiny_dataset):
    (tiny_dataset / "images/test/bad.jpg").write_bytes(b"not an image")
    (tiny_dataset / "labels/test/bad.txt").write_text("0 2 2 2 2\n")
    r = validate_split(FruitDataset(tiny_dataset, "test", CLASSES))
    joined = " ".join(r["issues"])
    assert "corrupt" in joined and "bad.jpg" in joined


def test_leakage_check(tiny_dataset):
    assert check_leakage(tiny_dataset) == []
    meta = tiny_dataset / "meta.csv"
    meta.write_text(meta.read_text() + "z.jpg,train,src_a.jpg,200,100,0\n")
    assert any("src_a.jpg" in m for m in check_leakage(tiny_dataset))
