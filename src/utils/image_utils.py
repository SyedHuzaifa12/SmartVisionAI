"""Image encoding and upload-validation helpers."""

from __future__ import annotations

import base64
import io
from typing import BinaryIO, Protocol

from PIL import Image


class InvalidImageError(ValueError):
    """Raised when an uploaded file isn't a valid, safely-sized image.

    A ``ValueError`` subclass rather than a bare exception so callers can
    still catch broadly if they want to, while the app's UI layer catches
    this specific type to show a clear, user-facing reason.
    """


class _SizedFile(Protocol):
    """The minimal shape this module needs from an uploaded file (matches
    Streamlit's ``UploadedFile`` without importing Streamlit here)."""

    size: int

    def read(self, *args, **kwargs) -> bytes: ...


def validate_uploaded_image(
    uploaded_file: _SizedFile | BinaryIO,
    *,
    max_size_mb: float,
    max_dimension_px: int,
) -> Image.Image:
    """Open and validate an uploaded file, raising :class:`InvalidImageError` on any problem.

    Guards against three real failure modes: an oversized upload, a
    corrupted/non-image file, and a pathologically large image that would
    burn excessive memory/latency/API cost downstream. Forces a full decode
    (``.load()``) rather than trusting ``Image.open()`` alone, since PIL only
    validates lazily and a truncated file can otherwise fail later, deep in
    the pipeline, with a far less useful error.
    """
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise InvalidImageError(f"Image is {size_mb:.1f} MB, which exceeds the {max_size_mb:.0f} MB limit.")

    try:
        image = Image.open(uploaded_file)
        image.load()
    except Exception as exc:
        raise InvalidImageError("This file isn't a valid, readable image.") from exc

    if max(image.size) > max_dimension_px:
        raise InvalidImageError(
            f"Image dimensions ({image.size[0]}x{image.size[1]}) exceed the {max_dimension_px}px limit."
        )

    return image


def image_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    """Encode a Pillow image as a base64 string.

    Args:
        image: The image to encode.
        image_format: Output format understood by ``Image.save`` (e.g. "PNG").

    Returns:
        The base64-encoded image data (no ``data:`` URI prefix).
    """
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode()


def to_image_data_url(image_base64: str, mime_type: str = "image/png") -> str:
    """Build a ``data:`` URL from a base64-encoded image."""
    return f"data:{mime_type};base64,{image_base64}"
