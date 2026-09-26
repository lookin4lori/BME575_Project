"""Tests for scripts/preprocess.py. Run from the repo root with: python -m pytest -v"""

import cv2
import numpy as np
import pytest

from scripts import preprocess as pp


def make_fake_fundus(h=300, w=400, radius=120):
    """Make a fake fundus photo: an orange circle (the 'retina') on a black background."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), radius, (180, 90, 40), -1)
    return img


# ---------- load_image ----------

def test_load_image_converts_bgr_to_rgb(tmp_path):
    """A pure-blue image saved by OpenCV (BGR) should load with blue in the last RGB channel."""
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[..., 0] = 255                      # channel 0 is blue in OpenCV's BGR order
    path = tmp_path / "blue.png"
    cv2.imwrite(str(path), bgr)

    rgb = pp.load_image(path)
    assert rgb[0, 0].tolist() == [0, 0, 255]


def test_load_image_missing_file_raises(tmp_path):
    """Loading a file that doesn't exist should raise a clear error instead of returning None."""
    with pytest.raises(FileNotFoundError):
        pp.load_image(tmp_path / "does_not_exist.png")


# ---------- crop_to_retina ----------

def test_crop_removes_black_border():
    """The crop should shrink to exactly the circle's size (diameter = 2 * radius + 1 pixels)."""
    cropped = pp.crop_to_retina(make_fake_fundus(radius=120))
    assert cropped.shape[:2] == (241, 241)


def test_crop_ignores_small_bright_artifact():
    """A small bright spot in the corner should not pull the crop away from the retina."""
    img = make_fake_fundus(radius=120)
    img[5:15, 5:15] = 255                  # fake reflection or camera text
    cropped = pp.crop_to_retina(img)
    assert cropped.shape[:2] == (241, 241)


def test_crop_all_black_image_returned_unchanged():
    """An all-black image should come back unchanged instead of crashing."""
    black = np.zeros((50, 60, 3), dtype=np.uint8)
    assert pp.crop_to_retina(black).shape == black.shape


# ---------- resize_image ----------

def test_resize_gives_224_square():
    """Any input size should come out as 224 x 224 x 3, as in the paper."""
    resized = pp.resize_image(make_fake_fundus(h=500, w=700))
    assert resized.shape == (224, 224, 3)


# ---------- ben_graham ----------

def test_ben_graham_uniform_image_becomes_gray():
    """A flat image has no detail, so image - blurred = 0 and every pixel should become 128."""
    flat = np.full((224, 224, 3), 200, dtype=np.uint8)
    assert np.all(pp.ben_graham(flat) == 128)


def test_ben_graham_keeps_shape_and_type():
    """The enhancement should not change the image's size or data type."""
    img = pp.resize_image(make_fake_fundus())
    out = pp.ben_graham(img)
    assert out.shape == img.shape and out.dtype == np.uint8


# ---------- apply_circular_mask ----------

def test_mask_grays_corners_and_keeps_center():
    """Pixels outside the circle should become 128, and the center should be untouched."""
    img = np.full((224, 224, 3), 200, dtype=np.uint8)
    masked = pp.apply_circular_mask(img)
    assert masked[0, 0].tolist() == [128, 128, 128]
    assert masked[112, 112].tolist() == [200, 200, 200]


# ---------- preprocess_image (full pipeline) ----------

def test_full_pipeline_output_format(tmp_path):
    """The full pipeline should return a 224 x 224 x 3 uint8 image, which is what dataset.py expects."""
    path = tmp_path / "fake.png"
    cv2.imwrite(str(path), make_fake_fundus())
    out = pp.preprocess_image(path)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8