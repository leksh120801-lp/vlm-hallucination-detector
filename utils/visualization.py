"""Heatmap rendering helpers (overlay, show, save).

Used by the experiment scripts. The Streamlit app builds its own overlay
inline so it can colour-map and gamma-correct in lockstep with the rest of
the dashboard styling.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Resize, colourise and blend a heatmap onto the original image (BGR)."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)


def show_heatmap(image: np.ndarray, heatmap: np.ndarray) -> None:
    """Display a heatmap overlay in a Matplotlib window (interactive contexts only)."""
    overlay = overlay_heatmap(image, heatmap)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def save_heatmap(image: np.ndarray, heatmap: np.ndarray, filename: str) -> None:
    """Persist a heatmap overlay to disk."""
    cv2.imwrite(filename, overlay_heatmap(image, heatmap))
