"""Tests for src.utils.image_utils."""

import base64
import io

import pytest
from PIL import Image

from src.utils.image_utils import InvalidImageError, image_to_base64, to_image_data_url, validate_uploaded_image


class _FakeUploadedFile(io.BytesIO):
    """Mimics Streamlit's UploadedFile: file-like, plus a ``.size`` attribute."""

    def __init__(self, data: bytes, size: int | None = None):
        super().__init__(data)
        self.size = size if size is not None else len(data)


def _fake_image_file(width: int = 10, height: int = 10, image_format: str = "PNG") -> _FakeUploadedFile:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color="blue").save(buffer, format=image_format)
    return _FakeUploadedFile(buffer.getvalue())


def test_image_to_base64_roundtrip():
    image = Image.new("RGB", (2, 2), color="red")

    encoded = image_to_base64(image)

    assert isinstance(encoded, str)
    assert base64.b64decode(encoded)  # decodes without error


def test_to_image_data_url_format():
    url = to_image_data_url("abc123")

    assert url == "data:image/png;base64,abc123"


def test_validate_uploaded_image_accepts_a_valid_image():
    image = validate_uploaded_image(_fake_image_file(), max_size_mb=10, max_dimension_px=6000)

    assert image.size == (10, 10)


def test_validate_uploaded_image_rejects_oversized_file():
    # Small real content, but a spoofed .size (as Streamlit reports it) over the limit.
    oversized = _fake_image_file()
    oversized.size = 20 * 1024 * 1024  # 20 MB

    with pytest.raises(InvalidImageError, match="MB"):
        validate_uploaded_image(oversized, max_size_mb=10, max_dimension_px=6000)


def test_validate_uploaded_image_rejects_corrupted_file():
    garbage = _FakeUploadedFile(b"this is not an image file at all")

    with pytest.raises(InvalidImageError, match="readable image"):
        validate_uploaded_image(garbage, max_size_mb=10, max_dimension_px=6000)


def test_validate_uploaded_image_rejects_oversized_dimensions():
    huge = _fake_image_file(width=8000, height=100)

    with pytest.raises(InvalidImageError, match="exceed"):
        validate_uploaded_image(huge, max_size_mb=50, max_dimension_px=6000)
