"""Tests for src.utils.image_utils."""

import base64

from PIL import Image

from src.utils.image_utils import image_to_base64, to_image_data_url


def test_image_to_base64_roundtrip():
    image = Image.new("RGB", (2, 2), color="red")

    encoded = image_to_base64(image)

    assert isinstance(encoded, str)
    assert base64.b64decode(encoded)  # decodes without error


def test_to_image_data_url_format():
    url = to_image_data_url("abc123")

    assert url == "data:image/png;base64,abc123"
