from fruitcount.tracking.tracker import IoUTracker
from conftest import det


def test_iou_tracker_keeps_identity_across_small_motion():
    tr = IoUTracker(iou_thr=0.3)
    ids = []
    for i in range(5):
        out = tr.update([det(10 + 2 * i, 10, 40 + 2 * i, 40)])
        ids.append(out[0].track_id)
    assert len(set(ids)) == 1


def test_iou_tracker_separates_two_fruits_and_new_id_for_new_fruit():
    tr = IoUTracker()
    a = tr.update([det(0, 0, 20, 20), det(100, 0, 120, 20)])
    b = tr.update([det(1, 0, 21, 20), det(101, 0, 121, 20), det(200, 0, 220, 20)])
    assert {a[0].track_id, a[1].track_id} == {b[0].track_id, b[1].track_id}
    assert b[2].track_id not in {a[0].track_id, a[1].track_id}


def test_iou_tracker_does_not_match_different_classes():
    tr = IoUTracker()
    a = tr.update([det(0, 0, 20, 20, cls=0)])
    b = tr.update([det(0, 0, 20, 20, cls=1)])
    assert a[0].track_id != b[0].track_id


def test_short_occlusion_is_bridged_long_one_splits_identity():
    tr = IoUTracker(max_age=2)
    first = tr.update([det(0, 0, 20, 20)])[0].track_id
    tr.update([]); tr.update([])
    assert tr.update([det(0, 0, 20, 20)])[0].track_id == first      # gap of 2 frames: same id
    tr.update([]); tr.update([]); tr.update([])
    assert tr.update([det(0, 0, 20, 20)])[0].track_id != first      # gap of 3 frames: id switch (over-count risk)
