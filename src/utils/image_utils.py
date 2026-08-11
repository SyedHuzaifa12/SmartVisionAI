"""Image encoding helpers."""

from __future__ import annotations

import base64
import io

from PIL import Image


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
