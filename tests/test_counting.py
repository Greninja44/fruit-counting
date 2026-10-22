from fruitcount.counting.counter import VideoCounter, count_detections, filter_detections
from conftest import CLASSES, det


def test_zero_detections_gives_zero_count():
    r = count_detections([], CLASSES, 0.25)
    assert r.total == 0 and r.per_class == {"apple": 0, "orange": 0, "banana": 0}


def test_multiple_detections_counted_per_class():
    dets = [det(0, 0, 10, 10, cls=0), det(20, 0, 30, 10, cls=0), det(40, 0, 50, 10, cls=2)]
    r = count_detections(dets, CLASSES, 0.25)
    assert r.total == 3 and r.per_class == {"apple": 2, "orange": 0, "banana": 1}


def test_confidence_filtering_is_inclusive_threshold():
    dets = [det(0, 0, 9, 9, conf=0.24), det(0, 0, 9, 9, conf=0.25), det(0, 0, 9, 9, conf=0.9)]
    assert len(filter_detections(dets, 0.25)) == 2
    assert count_detections(dets, CLASSES, 0.5).total == 1
    assert count_detections(dets, CLASSES, 0.0).total == 3


def test_min_area_filter():
    assert len(filter_detections([det(0, 0, 2, 2), det(0, 0, 20, 20)], 0.0, min_area=100)) == 1


def test_video_frame_counts_and_unique_tracks_are_separate():
    # The same 2 fruits (track ids 1, 2) visible for 5 frames, then one new fruit (3) for 1 frame.
    vc = VideoCounter(CLASSES, conf=0.25, min_track_frames=3)
    for _ in range(5):
        vc.update([det(0, 0, 9, 9, track_id=1), det(20, 0, 29, 9, cls=1, track_id=2)])
    vc.update([det(0, 0, 9, 9, track_id=1), det(40, 0, 49, 9, track_id=3)])
    s = vc.summary()
    assert vc.frame_counts == [2, 2, 2, 2, 2, 2]
    assert sum(vc.frame_counts) == 12                    # sum of per-frame counts ...
    assert s["unique_tracked"] == 2                      # ... is NOT the unique count (track 3 is flicker)
    assert s["unique_tracked_by_class"] == {"apple": 1, "orange": 1}
    assert "estimate" in s["note"].lower() or "occlusion" in s["note"]


def test_unique_count_is_none_when_tracking_off():
    vc = VideoCounter(CLASSES)
    vc.update([det(0, 0, 9, 9)])
    s = vc.summary(tracking_enabled=False)
    assert s["unique_tracked"] is None and s["visible_per_frame"]["max"] == 1


def test_detections_without_track_ids_never_create_unique_counts():
    vc = VideoCounter(CLASSES, min_track_frames=1)
    for _ in range(4):
        vc.update([det(0, 0, 9, 9)])
    assert vc.unique_tracked() == 0
