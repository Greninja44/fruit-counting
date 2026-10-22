"""End-to-end flow with a mocked model: predict -> count -> draw -> evaluate -> error analysis -> video."""
import cv2
import numpy as np

from fruitcount.counting.counter import count_detections
from fruitcount.counting.video import process_video
from fruitcount.data.dataset import FruitDataset
from fruitcount.detection.predict import run_model
from fruitcount.evaluation import error_analysis as EA
from fruitcount.evaluation.predictions import PredRecord, load_predictions, save_predictions
from fruitcount.evaluation.runner import evaluate_run, evaluate_subsets, predict_dataset, usable_samples
from fruitcount.tracking.tracker import make_frame_tracker
from fruitcount.visualization.draw_results import counts_header, draw_detections
from conftest import CLASSES, MockModel, det


def test_zero_detections_image_end_to_end(blank_image):
    m = MockModel([])
    dets = run_model(m, blank_image, 0.25, 0.7)
    res = count_detections(dets, CLASSES, 0.25)
    vis = draw_detections(blank_image, dets, counts_header(res.total, res.per_class))
    assert res.total == 0 and vis.shape == blank_image.shape


def test_multiple_detections_drawn_and_counted(blank_image):
    poly = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], float)
    m = MockModel([det(10, 10, 30, 30, polygon=poly), det(50, 50, 80, 90, cls=2), det(5, 5, 9, 9, conf=0.1)])
    dets = run_model(m, blank_image, 0.25, 0.7)
    assert count_detections(dets, CLASSES, 0.25).total == 2
    vis = draw_detections(blank_image, dets, ["Fruits: 2"])
    assert (vis != blank_image).any()


def _gt_matching_model(ds):
    """A mock that returns the GT of image 'a' perfectly, plus one hallucination on image 'b'."""
    a = ds.get(0)
    return {
        "a.jpg": [det(*g.box, cls=g.cls) for g in a.instances],
        "b.jpg": [det(10, 10, 50, 50, conf=0.8)],
    }


def test_evaluation_pipeline_with_mock_predictions(tiny_dataset, tmp_path):
    ds = FruitDataset(tiny_dataset, "test", CLASSES)
    samples, _ = usable_samples(ds)
    preds = _gt_matching_model(ds)
    recs = {s.image_path.name: PredRecord(s.image_path.name, s.width, s.height, 5.0, preds[s.image_path.name]) for s in samples}
    ev = evaluate_run(samples, recs, CLASSES, conf=0.25, masks=True)
    pi = ev["per_image"].set_index("image")
    assert pi.loc["a.jpg", "abs_error"] == 0 and pi.loc["b.jpg", "abs_error"] == 1 and pi.loc["b.jpg", "gt_count"] == 0
    assert ev["counting"]["MAE"] == 0.5 and ev["counting"]["exact_count_accuracy"] == 0.5
    assert ev["counting"]["MAPE_excluded_images"] == 1          # zero-GT image skipped for MAPE
    d = ev["detection"]
    assert (d["TP"], d["FP"], d["FN"]) == (2, 1, 0)
    assert ev["mask"]["TP"] == 0 or "mean_matched_mask_iou" in ev["mask"]   # predictions carry no polygons -> mask stats exist
    # persistence round trip keeps predictions separate from GT
    f = tmp_path / "p.json"
    save_predictions(f, list(recs.values()), {"run": "t"})
    info, back = load_predictions(f)
    assert info["run"] == "t" and len(back["a.jpg"].dets) == 2


def test_error_analysis_categories(tiny_dataset):
    ds = FruitDataset(tiny_dataset, "test", CLASSES)
    a = ds.get(0)
    g0, g1 = a.instances
    dets = [
        det(*g0.box, conf=0.9, cls=g0.cls),                       # TP
        det(*g0.box, conf=0.8, cls=g0.cls),                       # duplicate of g0
        det(*g1.box, conf=0.7, cls=(g1.cls + 1) % 3),             # wrong class for g1 -> g1 also missed
        det(0, 80, 10, 90, conf=0.6),                             # background false positive
    ]
    err, kept = EA.analyse_image(a, dets, conf=0.25)
    assert sorted(err.pred_cat.values()) == ["duplicate", "false_positive", "wrong_class"]
    assert list(err.gt_cat.values()) == ["missed_other"]
    assert "right_total_wrong_objects" not in err.image_cats    # pred count 4 != gt count 2
    assert EA.summarize([err])["duplicate"]["instances"] == 1


def test_merged_detection_and_right_total_wrong_objects(tiny_dataset):
    from fruitcount.data.dataset import GTInstance, Sample

    s = Sample(tiny_dataset / "images/test/a.jpg", 200, 100, [GTInstance(0, (10, 10, 60, 60)), GTInstance(0, (65, 10, 115, 60))])
    err, _ = EA.analyse_image(s, [det(5, 5, 120, 65), det(150, 70, 190, 99)], 0.25)
    assert sorted(err.pred_cat.values()) == ["false_positive", "merged"]
    assert "right_total_wrong_objects" in err.image_cats         # 2 predicted == 2 GT, but 0 correct
    assert set(err.gt_cat.values()) == {"missed_touching"}       # both GT fruits were swallowed by one merged box


def test_small_missed_category():
    from fruitcount.data.dataset import GTInstance, Sample
    from pathlib import Path

    s = Sample(Path("x.jpg"), 640, 480, [GTInstance(0, (10, 10, 25, 25))])
    err, _ = EA.analyse_image(s, [], 0.25)
    assert list(err.gt_cat.values()) == ["missed_small"]


def test_qualitative_flags_without_gt():
    f = EA.qualitative_flags([det(0, 0, 50, 50), det(1, 1, 50, 50), det(0, 0, 200, 200, conf=0.3), det(10, 10, 20, 20), det(30, 30, 40, 40)], 0.25)
    assert f["possible_duplicates"] >= 1 and f["possible_merged"] >= 1


def test_subset_reporting_hides_tiny_subsets(tiny_dataset):
    ds = FruitDataset(tiny_dataset, "test", CLASSES)
    samples, _ = usable_samples(ds)
    recs = {s.image_path.name: PredRecord(s.image_path.name, s.width, s.height, 1.0, []) for s in samples}
    ev = evaluate_run(samples, recs, CLASSES, 0.25)
    import pandas as pd
    tags = pd.DataFrame({"image": ["a.jpg", "b.jpg"], "low_light": [1, 1]})
    out = evaluate_subsets(ev["per_image"], tags, ["low_light"], min_images=5)
    assert "not reported" in out.loc[0, "note"] and "MAE" not in out


def test_predict_dataset_uses_mock_and_low_conf(tiny_dataset):
    ds = FruitDataset(tiny_dataset, "test", CLASSES)
    samples, _ = usable_samples(ds)
    m = MockModel([det(0, 0, 10, 10, conf=0.05), det(0, 0, 10, 10, conf=0.9)])
    recs, info = predict_dataset(m, samples, 0.001, 0.7, 640, warmup=1, timing_conf=0.25)
    assert all(len(r.dets) == 2 for r in recs) and all(r.ms >= 0 for r in recs) and "timing_conf" in info


def test_video_counts_with_iou_tracker(synthetic_video, tmp_path):
    centres = [(30 + 5 * i, 60) for i in range(12)]
    m = MockModel(per_call=[[det(cx - 15, cy - 15, cx + 15, cy + 15, cls=1)] for cx, cy in centres])
    fn = make_frame_tracker(m, "iou", 0.25, 0.7, 640)
    out_json = tmp_path / "o" / "s.json"
    s = process_video(synthetic_video, fn, CLASSES, 0.25, True, 3, tmp_path / "o" / "v.mp4", out_json)
    assert s["frames_processed"] == 12 and sum(s["frame_counts"]) == 12
    assert s["unique_tracked"] == 1 and s["unique_tracked_by_class"] == {"orange": 1}   # one fruit, 12 sightings
    assert out_json.is_file() and (tmp_path / "o" / "v.mp4").stat().st_size > 0


def test_video_without_tracking_reports_no_unique_count(synthetic_video):
    m = MockModel([det(0, 0, 10, 10)])
    s = process_video(synthetic_video, lambda f: m.predict(f), CLASSES, 0.25, False)
    assert s["unique_tracked"] is None and s["visible_per_frame"]["max"] == 1 and "disabled" in s["note"]


def test_video_corrupt_raises(tmp_path):
    import pytest
    from fruitcount.utils.io import InvalidInputError

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"123")
    with pytest.raises(InvalidInputError):
        process_video(bad, lambda f: [], CLASSES, 0.25, False)
