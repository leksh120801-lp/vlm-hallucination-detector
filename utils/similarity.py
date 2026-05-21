"""Cosine-similarity scoring + threshold-based hallucination decision.

This module is the cosine baseline; the richer detection methods live in
:mod:`utils.methods`. All three methods consume the cosine score produced
here as their input feature.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable, List

import torch
import torch.nn.functional as F

from utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_similarity(image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between L2-normalized image and text embeddings.

    Args:
        image_embedding: ``(B_img, D)`` tensor.
        text_embedding:  ``(B_txt, D)`` tensor.

    Returns:
        ``(B_img, B_txt)`` tensor of cosine similarities in ``[-1, 1]``.
    """
    image_embedding = F.normalize(image_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    return torch.matmul(image_embedding, text_embedding.T)


def detect_hallucination(
    score: float,
    threshold: float = 0.25,
    model_name: str | None = None,
    dataset: str | None = None,
) -> str:
    """Threshold a single similarity score into a decision label.

    Returns one of ``"likely correct"`` or ``"possible hallucination"``.

    Threshold resolution order:
      1. If ``model_name`` is given, look up the per-backbone threshold from
         ``configs/thresholds.yaml`` via :func:`utils.config.get_threshold`.
      2. Otherwise, use the explicit ``threshold`` argument.
    """
    if model_name is not None:
        # Lazy import to avoid a circular dependency at module-load time.
        from utils.config import get_threshold
        threshold = get_threshold(model_name, dataset=dataset)

    return "likely correct" if score >= threshold else "possible hallucination"


def save_results(
    image_path: str,
    captions: Iterable[str],
    scores: Iterable[float],
    decisions: Iterable[str],
    output_dir: str = "experiments/results",
) -> str:
    """Persist (caption, score, decision) rows as a timestamped JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    rows: List[dict] = [
        {"caption": caption, "similarity_score": float(score), "decision": decision}
        for caption, score, decision in zip(captions, scores, decisions)
    ]

    payload = {
        "image": image_path,
        "results": rows,
        "timestamp": datetime.now().isoformat(),
    }

    filename = f"{output_dir}/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as fh:
        json.dump(payload, fh, indent=4)

    logger.info("Results saved to %s", filename)
    return filename
