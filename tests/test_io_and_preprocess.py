import numpy as np
import pytest

from fruitcount.detection.base import ModelLoadError
from fruitcount.detection.preprocess import enhance_clahe, enhance_gamma, tile_predict
from fruitcount.detection.yolo_base import UltralyticsModel
from fruitcount.utils.io import InvalidInputError, list_images, open_video, read_image
from conftest import CLASSES, MockModel, det


def test_missing_and_corrupt_image(tmp_path):
    with pytest.raises(InvalidInputError, match="not found"):
        read_image(tmp_path / "nope.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"garbage")
    with pytest.raises(InvalidInputError, match="Corrupt"):
        read_image(bad)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(InvalidInputError):
        read_image(empty)


def test_list_images(tiny_dataset):
    assert len(list_images(tiny_dataset / "images/test")) == 3
    with pytest.raises(InvalidInputError):
        list_images(tiny_dataset / "missing")


def test_missing_and_corrupt_video(tmp_path, synthetic_video):
    with pytest.raises(InvalidInputError):
        open_video(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(InvalidInputError):
        open_video(bad)
    open_video(synthetic_video).release()


def test_missing_weights_message(tmp_path):
    with pytest.raises(ModelLoadError, match="weights not found"):
        UltralyticsModel(tmp_path / "none.pt", CLASSES)


def test_gamma_brightens_dark_and_leaves_bright_alone():
    dark, bright = np.full((10, 10, 3), 20, np.uint8), np.full((10, 10, 3), 200, np.uint8)
    assert enhance_gamma(dark).mean() > dark.mean() * 2
    assert np.array_equal(enhance_gamma(bright), bright)


def test_clahe_preserves_shape_and_dtype():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    out = enhance_clahe(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_tiling_offsets_boxes_and_merges_duplicates():
    img = np.zeros((200, 400, 3), np.uint8)

    def fn(im):                       # a "fruit" at the same absolute location is seen by several tiles
        h, w = im.shape[:2]
        return [det(10, 10, 40, 40)] if (h, w) != (200, 400) else []

    out = tile_predict(fn, img, tile=100, overlap=0.0)
    assert len(out) >= 1 and all(d.box[2] - d.box[0] == 30 for d in out)
    xs = sorted(d.box[0] for d in out)
    assert xs[0] == 10 and xs[-1] > 10               # shifted into full-image coordinates
