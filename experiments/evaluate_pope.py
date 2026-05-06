"""Evaluate the detector on POPE — the canonical hallucination benchmark.

POPE (Polling-based Object Probing Evaluation) frames hallucination detection
as a binary yes/no question over an image:

    Q: "Is there a {object} in the image?"     A: "yes" or "no"

We adapt our cross-modal-similarity pipeline by treating each question as a
caption ("a photo containing a {object}") and thresholding the similarity:
score above τ → "yes", below → "no". The resulting accuracy / precision /
recall / F1 are directly comparable to the numbers reported in the POPE
paper (https://arxiv.org/abs/2305.10355).

Three POPE splits are usually reported:
    - random      : random sampling of negative objects
    - popular     : negatives drawn from the most popular COCO categories
    - adversarial : negatives drawn from co-occurring categories (hardest)

The default split below is ``adversarial`` because that's the headline
number. Override with ``--split``.

Run::

    python experiments/evaluate_pope.py
    python experiments/evaluate_pope.py --model-name SigLIP --split popular
    python experiments/evaluate_pope.py --threshold 0.20 --sample-size 500

The dataset is downloaded once via ``datasets.load_dataset`` and cached at
``HF_HOME``; subsequent runs are offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_registry import load_model_by_name
from utils.config import get_threshold
from utils.datasets import _safe_load_dataset
from utils.logging_config import get_logger
from utils.metrics import compute_classification_metrics
from utils.similarity import compute_similarity

logger = get_logger(__name__)

POPE_DATASET = "lmms-lab/POPE"   # canonical POPE mirror with embedded images
RESULTS_DIR = ROOT / "experiments" / "results"


def _question_to_caption(question: str) -> str:
    """Turn a POPE question into a short positive caption to score.

    Example: "Is there a dog in the image?" → "a photo containing a dog".
    Falls back to the raw question if parsing fails.
    """
    q = question.strip().rstrip("?").lower()
    for prefix in ("is there a ", "is there an "):
        if q.startswith(prefix):
            obj = q[len(prefix):].strip()
            return f"a photo containing a {obj}"
    return question


def _normalise_label(label: str) -> str:
    return "yes" if str(label).strip().lower().startswith("y") else "no"


def evaluate_pope(
    model_name: str = "CLIP",
    split: str = "adversarial",
    sample_size: Optional[int] = 200,
    threshold: Optional[float] = None,
) -> dict:
    """Score every (image, question) pair and aggregate POPE-style metrics.

    Args:
        model_name: CLIP / BLIP / SigLIP.
        split: ``"random" | "popular" | "adversarial"``.
        sample_size: cap on rows — keep small for a smoke test.
        threshold: cosine threshold; default = per-backbone value from
                   ``configs/thresholds.yaml``.
    """
    if threshold is None:
        threshold = get_threshold(model_name)
    logger.info(
        "Evaluating %s on POPE-%s (n=%s, τ=%.3f)",
        model_name, split, sample_size or "all", threshold,
    )

    # Most POPE mirrors expose a config per split; ``lmms-lab/POPE`` uses
    # the row's ``category`` field instead. Both shapes are handled below.
    try:
        dataset = _safe_load_dataset(POPE_DATASET, split="test", name=split)
    except Exception:
        dataset = _safe_load_dataset(POPE_DATASET, split="test")
        # Filter to the requested split if the loader returned everything.
        dataset = dataset.filter(
            lambda r: split in str(r.get("category", "")).lower()
        )

    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    backbone, processor, device = load_model_by_name(model_name)

    y_true: list[str] = []
    y_pred: list[str] = []

    for row in dataset:
        image = row.get("image")
        question = row.get("question") or row.get("prompt") or ""
        label = row.get("answer") or row.get("label") or ""
        if image is None or not question or not label:
            continue

        try:
            image = image.convert("RGB")
        except Exception:
            continue

        caption = _question_to_caption(question)
        inputs = processor(
            text=[caption], images=image, return_tensors="pt", padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = backbone(**inputs)

        score = float(
            compute_similarity(outputs.image_embeds, outputs.text_embeds)[0][0].item()
        )
        prediction = "yes" if score >= threshold else "no"

        y_true.append(_normalise_label(label))
        y_pred.append(prediction)

    if not y_true:
        raise RuntimeError(
            "POPE evaluation produced no rows. Check the dataset cache and split name."
        )

    # POPE's positive class is "yes" (object IS in the image).
    metrics = compute_classification_metrics(y_true, y_pred, hallucination_label="yes")
    metrics.update({
        "model_name": model_name,
        "split": split,
        "threshold": threshold,
        "n": len(y_true),
        "yes_rate_pred": sum(1 for p in y_pred if p == "yes") / len(y_pred),
        "yes_rate_true": sum(1 for t in y_true if t == "yes") / len(y_true),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="CLIP", choices=["CLIP", "BLIP", "SigLIP"])
    parser.add_argument(
        "--split", default="adversarial",
        choices=["random", "popular", "adversarial"],
    )
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Use 0 for the full split.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the per-backbone threshold from configs/thresholds.yaml.")
    args = parser.parse_args()

    metrics = evaluate_pope(
        model_name=args.model_name,
        split=args.split,
        sample_size=args.sample_size or None,
        threshold=args.threshold,
    )

    print(json.dumps(metrics, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"pope_{args.model_name}_{args.split}_{stamp}.json"
    with out_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved POPE metrics to %s", out_path)


if __name__ == "__main__":
    main()
