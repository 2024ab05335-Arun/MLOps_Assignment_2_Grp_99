from PIL import Image
import numpy as np
from pathlib import Path

from preprocess_petimages import center_crop, resize_shorter_side


def test_center_crop_square():
    # create a 300x200 image (w>h) and crop to 100
    img = Image.new("RGB", (300, 200), color=(255, 0, 0))
    cropped = center_crop(img, 100)
    assert cropped.size == (100, 100)


def test_center_crop_smaller_than_image():
    img = Image.new("RGB", (50, 80), color=(0, 255, 0))
    # requested crop is larger than width; function currently clamps left/top to 0
    cropped = center_crop(img, 40)
    assert cropped.size == (40, 40)


def test_resize_shorter_side_preserves_aspect_ratio():
    img = Image.new("RGB", (400, 200), color=(0, 0, 255))
    out = resize_shorter_side(img, 256)
    w, h = out.size
    # shorter side should be 256
    assert min(w, h) == 256
    # aspect ratio approximately preserved
    assert abs((w / h) - (400 / 200)) < 0.1
