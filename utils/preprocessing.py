"""Image-loading and PIL ↔ OpenCV conversion helpers."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

_REQUEST_TIMEOUT_SECONDS = 8


def load_image(image_path: str) -> Image.Image:
    """Open a local image and return it as RGB."""
    return Image.open(image_path).convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
    """Download an image and return it as RGB."""
    response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR ``np.ndarray``."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
