import base64
import io
import numpy as np
from PIL import Image

from inference_service import preprocess_pil_image, decode_base64_image, IMAGE_SIZE


def create_sample_image(color=(123, 222, 100), size=(300, 200)):
    return Image.new("RGB", size, color=color)


def test_preprocess_pil_image_shape_and_range():
    img = create_sample_image()
    arr = preprocess_pil_image(img)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0], 3) or arr.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    # values should be in [0,1]
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_decode_base64_image_roundtrip():
    img = create_sample_image()
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b = base64.b64encode(buf.getvalue()).decode("utf-8")
    decoded = decode_base64_image(b)
    assert isinstance(decoded, Image.Image)
    assert decoded.mode in ("RGB", "RGBA")
