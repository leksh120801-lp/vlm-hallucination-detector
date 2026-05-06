"""Evaluate the three detection methods on a dataset and produce a comparison CSV.

Runs Threshold / Consistency / Logistic side-by-side on the same (image, caption,
adversarial-captions) tuples, then computes accuracy / precision / recall / F1
per method. The output CSV lives at
``experiments/results/method_comparison.csv``.

Run::

    python experiments/evaluate_methods.py
    python experiments/evaluate_methods.py --model-name SigLIP --sample-size 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_registry import load_model_by_name
from utils.caption_attack import generate_adversarial_captions
from utils.classifier import MODEL_PATH
from utils.datasets import load_coco_sample, load_flickr_sample
from utils.logging_config import get_logger
from utils.methods import (
    HALLUCINATION,
    LIKELY_CORRECT,
    method_consistency_threshold,
    method_logistic,
    method_threshold,
)
from utils.metrics import compute_classification_metrics
from utils.similarity import compute_similarity

logger = get_logger(__name__)

RESULTS_PATH = ROOT / "experiments" / "results" / "method_comparison.csv"
DATASET_LOADERS = {
    "Flickr30k": load_flickr_sample,
    "COCO":      load_coco_sample,
}


def _load_logistic_with_metadata():
    """Return ``(model, metadata)`` or ``(None, None)`` if no checkpoint exists."""
    if not MODEL_PATH.exists():
        return None, None
    try:
        payload = joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.warning("Could not load logistic model at %s: %s", MODEL_PATH, exc)
        return None, None

    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata")
    return payload, None


def evaluate_methods(
    model_name: str = "CLIP",
    dataset_name: str = "COCO",
    sample_size: int = 10,
    similarity_threshold: float = 0.25,
    consistency_threshold: float = 0.2,
):
    """Score every (image, caption) pair (real + adversarial) under all methods."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    samples = DATASET_LOADERS[dataset_name](sample_size)
    if not samples:
        raise RuntimeError(
            f"No samples loaded from {dataset_name!r}. "
            "Check your network connection or dataset version pin."
        )

    logger.info("Loaded %d samples from %s", len(samples), dataset_name)

    model, processor, device = load_model_by_name(model_name)
    logistic_model, logistic_meta = _load_logistic_with_metadata()

    if logistic_model is None:
        logger.warning(
            "No trained logistic head found at %s. Skipping the Logistic method. "
            "Train one with `python experiments/train_logistic.py`.", MODEL_PATH,
        )
    elif logistic_meta and logistic_meta.get("model_name") not in (None, model_name):
        logger.warning(
            "Logistic head was trained on backbone %r but evaluation uses %r — "
            "score distributions differ across backbones, so the Logistic method's "
            "metrics will be unreliable. Retrain with "
            "`python experiments/train_logistic.py --model-name %s`.",
            logistic_meta["model_name"], model_name, model_name,
        )

    y_true: list[str] = []
    y_pred_threshold: list[str] = []
    y_pred_consistency: list[str] = []
    y_pred_logistic: list[str] = []
    detailed_rows: list[dict] = []

    for sample_id, sample in enumerate(samples):
        captions = [sample["caption"]] + generate_adversarial_captions(sample["caption"])

        inputs = processor(
            text=captions, images=sample["image"], return_tensors="pt", padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        scores = (
            compute_similarity(outputs.image_embeds, outputs.text_embeds)[0]
            .detach().cpu().tolist()
        )

        for caption_index, candidate_caption in enumerate(captions):
            candidate_score = float(scores[caption_index])
            comparison_scores = [
                float(scores[i]) for i in range(len(scores)) if i != caption_index
            ]
            label = LIKELY_CORRECT if caption_index == 0 else HALLUCINATION

            threshold_pred = method_threshold(candidate_score, threshold=similarity_threshold)
            consistency_pred, consistency_score = method_consistency_threshold(
                candidate_score, comparison_scores, threshold=consistency_threshold,
            )
            if logistic_model is not None:
                logistic_pred, logistic_consistency = method_logistic(
                    logistic_model, candidate_score, comparison_scores,
                )
            else:
                logistic_pred = LIKELY_CORRECT
                logistic_consistency = float("nan")

            y_true.append(label)
            y_pred_threshold.append(threshold_pred)
            y_pred_consistency.append(consistency_pred)
            y_pred_logistic.append(logistic_pred)

            detailed_rows.append({
                "sample_id": sample_id,
                "caption_type": "original" if caption_index == 0 else f"attack_{caption_index}",
                "model": model_name,
                "dataset": dataset_name,
                "caption": candidate_caption,
                "score": round(candidate_score, 4),
                "consistency_score": round(consistency_score, 4),
                "logistic_consistency_score": (
                    round(logistic_consistency, 4)
                    if logistic_consistency == logistic_consistency  # i.e. not NaN
                    else None
                ),
                "y_true": label,
                "y_pred_threshold":   threshold_pred,
                "y_pred_consistency": consistency_pred,
                "y_pred_logistic":    logistic_pred,
            })

    metrics_rows = [
        {
            "method": "Threshold", "model": model_name, "dataset": dataset_name,
            "samples": len(y_true),
            **compute_classification_metrics(y_true, y_pred_threshold),
        },
        {
            "method": "Consistency", "model": model_name, "dataset": dataset_name,
            "samples": len(y_true),
            **compute_classification_metrics(y_true, y_pred_consistency),
        },
    ]
    if logistic_model is not None:
        metrics_rows.append({
            "method": "Logistic", "model": model_name, "dataset": dataset_name,
            "samples": len(y_true),
            **compute_classification_metrics(y_true, y_pred_logistic),
        })
    return detailed_rows, metrics_rows


def save_results(metrics_rows, path: Path = RESULTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(path, index=False)
    return path


def _parse_args_and_run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name",  default="CLIP", choices=["CLIP", "BLIP", "SigLIP"])
    parser.add_argument("--dataset-name", default="COCO", choices=list(DATASET_LOADERS))
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--similarity-threshold",  type=float, default=0.25)
    parser.add_argument("--consistency-threshold", type=float, default=0.2)
    args = parser.parse_args()

    _, metrics_rows = evaluate_methods(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        sample_size=args.sample_size,
        similarity_threshold=args.similarity_threshold,
        consistency_threshold=args.consistency_threshold,
    )
    output_path = save_results(metrics_rows)
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    _parse_args_and_run()
