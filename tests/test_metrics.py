import math

import numpy as np
import pytest

from fruitcount.evaluation import metrics as M


def test_counting_metrics_definitions():
    pred, gt = [3, 5, 0, 2], [3, 3, 1, 4]
    m = M.counting_metrics(pred, gt)
    err = np.array([0, 2, -1, -2])
    assert m["MAE"] == pytest.approx(np.abs(err).mean())                    # 1.25
    assert m["RMSE"] == pytest.approx(math.sqrt((err**2).mean()))          # sqrt(2.25)
    assert m["exact_count_accuracy"] == 0.25
    assert m["within_one_accuracy"] == 0.5
    assert m["mean_signed_error"] == pytest.approx(-0.25)


def test_mape_zero_gt_policies():
    pred, gt = [2, 1], [0, 2]
    skip = M.counting_metrics(pred, gt, "skip")
    assert skip["MAPE_percent"] == pytest.approx(50.0) and skip["MAPE_excluded_images"] == 1
    eps = M.counting_metrics(pred, gt, "epsilon", eps=1.0)
    assert eps["MAPE_percent"] == pytest.approx((2 / 1 + 1 / 2) / 2 * 100)
    assert math.isnan(M.counting_metrics([1], [0], "skip")["MAPE_percent"])


def test_counting_metrics_validation():
    with pytest.raises(ValueError):
        M.counting_metrics([1, 2], [1])
    with pytest.raises(ValueError):
        M.counting_metrics([], [])
    with pytest.raises(ValueError):
        M.counting_metrics([1], [1], "bogus")


def _img(iou, conf, pcls, gcls, name="x"):
    return M.ImageMatchData(name, np.array(conf, float), np.array(pcls), np.array(gcls), np.array(iou, float).reshape(len(conf), len(gcls)))


def test_matching_is_greedy_by_confidence_and_one_to_one():
    d = _img([[0.9], [0.8]], [0.6, 0.9], [0, 0], [0])      # two preds on one GT: higher conf wins
    pg, _ = M.match_image(d, 0.5)
    assert list(pg) == [-1, 0]


def test_matching_respects_class_and_threshold():
    d = _img([[0.9]], [0.9], [1], [0])
    assert M.match_image(d, 0.5)[0][0] == -1               # wrong class
    d = _img([[0.4]], [0.9], [0], [0])
    assert M.match_image(d, 0.5)[0][0] == -1               # below IoU threshold


def test_perfect_predictions_give_ap_one_and_prf_one():
    d = _img(np.eye(2), [0.9, 0.8], [0, 1], [0, 1])
    r = M.detection_metrics([d], ["a", "b"], 0.5)
    assert (r["precision"], r["recall"], r["F1"]) == (1, 1, 1)
    assert r["mAP50"] == pytest.approx(1.0)


def test_false_positive_and_false_negative_counted():
    d = _img([[0.9, 0.0], [0.0, 0.0]], [0.9, 0.9], [0, 0], [0, 0])   # 1 TP, 1 FP, 1 FN
    r = M.detection_metrics([d], ["a"], 0.5)
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_confidence_threshold_changes_precision_recall_not_map():
    d = _img([[0.9, 0.0], [0.0, 0.9]], [0.9, 0.3], [0, 0], [0, 0])
    lo, hi = M.detection_metrics([d], ["a"], 0.1), M.detection_metrics([d], ["a"], 0.5)
    assert lo["recall"] == 1.0 and hi["recall"] == 0.5
    assert lo["mAP50"] == hi["mAP50"]


def test_average_precision_known_value():
    # recall steps 0.5 at precision 1.0, then 1.0 at precision 0.5 -> area approx 0.75
    ap = M.average_precision(np.array([0.5, 1.0]), np.array([1.0, 0.5]))
    assert ap == pytest.approx(0.75, abs=0.02)


def test_empty_predictions_zero_ap():
    d = _img(np.zeros((0, 1)), [], [], [0])
    assert M.detection_metrics([d], ["a"], 0.25)["mAP50"] == 0.0
