"""Train the logistic-regression detection head.

By default, training runs in **per-caption** mode so the features match what
``utils.methods.method_logistic`` sees at inference time:

    one row per caption (real and adversarial)
    label = 1 for the real caption, 0 for adversaries
    feature dim = 4   (original_score, mean_attack, consistency, variance)

Per-image mode (``--per-image``) is also available; it produces 8-D features
aggregated per image and labels each image by whether the real caption beat
its adversaries. This is useful for image-level scoring but **does not match
the per-caption decision API**, which is why per-caption is the default.

Single-class fallback: if the dataset ends up with only one class (common with
very small ``--sample-size``) the script falls back to ``DummyClassifier`` so
users still get an artifact they can inspect.

Run::

    python experiments/train_logistic.py
    python experiments/train_logistic.py --model-name SigLIP --sample-size 50
    python experiments/train_logistic.py --do-grid-search
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import contextlib

from models.model_registry import load_model_by_name
from utils.caption_attack import generate_adversarial_captions
from utils.classifier import MODEL_PATH, extract_features, extract_image_features
from utils.datasets import load_coco_sample, load_flickr_sample
from utils.logging_config import get_logger
from utils.similarity import compute_similarity

logger = get_logger(__name__)

DATASET_LOADERS = {
    "Flickr30k": load_flickr_sample,
    "COCO": load_coco_sample,
}


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _score_image(model, processor, device, image, captions):
    """Forward pass + cosine similarity for one (image, captions) tuple."""
    inputs = processor(
        text=captions, images=image, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return (
        compute_similarity(outputs.image_embeds, outputs.text_embeds)[0]
        .detach().cpu().tolist()
    )


def build_training_dataset(model_name: str, dataset_name: str, sample_size: int):
    """Per-image features (8-D), one label per image."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    samples = DATASET_LOADERS[dataset_name](sample_size)
    model, processor, device = load_model_by_name(model_name)

    X, y = [], []
    for sample in samples:
        captions = [sample["caption"]] + generate_adversarial_captions(sample["caption"])
        scores = _score_image(model, processor, device, sample["image"], captions)
        original_score = float(scores[0])
        attack_scores = [float(s) for s in scores[1:]]

        X.append(extract_image_features(original_score, attack_scores))
        y.append(1 if (not attack_scores or original_score >= max(attack_scores)) else 0)

    return X, y


def build_training_dataset_per_caption(model_name: str, dataset_name: str, sample_size: int):
    """Per-caption features (4-D), one label per caption."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    samples = DATASET_LOADERS[dataset_name](sample_size)
    model, processor, device = load_model_by_name(model_name)

    X, y = [], []
    for sample in samples:
        captions = [sample["caption"]] + generate_adversarial_captions(sample["caption"])
        scores = _score_image(model, processor, device, sample["image"], captions)

        for idx, score in enumerate(scores):
            comparison_scores = [scores[i] for i in range(len(scores)) if i != idx]
            X.append(extract_features(score, comparison_scores))
            y.append(1 if idx == 0 else 0)

    return X, y


# ---------------------------------------------------------------------------
# Metrics + training
# ---------------------------------------------------------------------------

def evaluate_and_save_metrics(model, X_test, y_test, metrics_path=None):
    """Score on a held-out set and (optionally) persist the metrics."""
    y_pred = model.predict(X_test)
    y_proba = None
    with contextlib.suppress(Exception):
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(set(y_test)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as fh:
            json.dump(metrics, fh, indent=2)

    return metrics


def train_logistic_model(
    *,
    model_name: str = "CLIP",
    dataset_name: str = "COCO",
    sample_size: int = 50,
    test_size: float = 0.2,
    random_seed: int = 42,
    do_grid_search: bool = False,
    cv_folds: int = 3,
    metrics_path=None,
    model_path=None,
    per_caption: bool = True,
):
    """End-to-end: build dataset → split → fit pipeline → evaluate → save."""
    logger.info(
        "Training logistic head (model=%s, dataset=%s, samples=%d, mode=%s)",
        model_name, dataset_name, sample_size,
        "per-caption" if per_caption else "per-image",
    )

    if per_caption:
        X, y = build_training_dataset_per_caption(model_name, dataset_name, sample_size)
    else:
        X, y = build_training_dataset(model_name, dataset_name, sample_size)

    if not X:
        raise RuntimeError(
            f"No training data was collected from dataset {dataset_name!r}. "
            "Check your network connection or run "
            "`pip install 'datasets>=2.14,<3.0'`."
        )

    X = np.array(X)
    y = np.array(y)

    metadata_base = {
        "dataset_name":  dataset_name,
        "model_name":    model_name,
        "sample_size":   int(sample_size),
        "test_size":     float(test_size),
        "random_seed":   int(random_seed),
        "num_examples":  int(len(X)),
        "feature_dim":   int(X.shape[1]),
        "model_type":    "per_caption" if per_caption else "per_image",
    }

    out_model_path = MODEL_PATH if model_path is None else Path(model_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Single-class fallback ----
    if len(np.unique(y)) < 2:
        logger.warning(
            "Only one class present in training data — falling back to DummyClassifier. "
            "Increase --sample-size to obtain both classes."
        )
        dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
        metrics = evaluate_and_save_metrics(dummy, X, y, metrics_path=metrics_path)
        joblib.dump(
            {"model": dummy, "metadata": {**metadata_base, "metrics": metrics, "estimator": "dummy"}},
            out_model_path,
        )
        return {
            "model_path": str(out_model_path),
            "num_examples": int(len(X)),
            "metrics": metrics,
            "grid_search_best_params": None,
        }

    # ---- Reproducibility ----
    random.seed(random_seed)
    np.random.seed(random_seed)
    with contextlib.suppress(Exception):
        torch.manual_seed(random_seed)

    # ---- Train / val split ----
    stratify = y if len(np.unique(y)) > 1 and len(y) >= 4 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )

    # Tiny-set safety net.
    if len(np.unique(y_train)) < 2:
        X_train, y_train = X, y
        X_val, y_val = X, y

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear",
        )),
    ])
    grid_best_params = None
    best_model = pipeline

    if do_grid_search:
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        cv = StratifiedKFold(
            n_splits=max(2, min(cv_folds, len(y_train))),
            shuffle=True, random_state=random_seed,
        )
        grid = GridSearchCV(
            pipeline, param_grid=param_grid, cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        grid_best_params = getattr(grid, "best_params_", None)
    else:
        best_model.fit(X_train, y_train)

    metrics = evaluate_and_save_metrics(
        best_model, X_val, y_val, metrics_path=metrics_path,
    )

    metadata = {
        **metadata_base,
        "metrics": metrics,
        "grid_search_best_params": grid_best_params,
        "estimator": "logistic_regression",
    }
    joblib.dump({"model": best_model, "metadata": metadata}, out_model_path)
    logger.info("Saved model to %s (val F1=%.3f)", out_model_path, metrics["f1"])

    return {
        "model_path": str(out_model_path),
        "num_examples": int(len(X)),
        "metrics": metrics,
        "grid_search_best_params": grid_best_params,
    }


def _parse_args_and_run():
    parser = argparse.ArgumentParser(description="Train the logistic-regression detector.")
    parser.add_argument("--model-name", default="CLIP", choices=["CLIP", "BLIP", "SigLIP"])
    parser.add_argument("--dataset-name", default="COCO", choices=list(DATASET_LOADERS))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--do-grid-search", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--metrics-path",
        default=str(ROOT / "experiments" / "results" / "logistic_metrics.json"),
    )
    parser.add_argument("--model-path", default=None)

    # Mutually-exclusive mode: --per-caption (default) vs --per-image.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--per-caption",
        dest="per_caption",
        action="store_true",
        help="Default. Train one row per caption (real + adversarial).",
    )
    mode.add_argument(
        "--per-image",
        dest="per_caption",
        action="store_false",
        help="Train one row per image with 8-D aggregated features.",
    )
    parser.set_defaults(per_caption=True)

    args = parser.parse_args()
    result = train_logistic_model(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        sample_size=args.sample_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
        do_grid_search=args.do_grid_search,
        cv_folds=args.cv_folds,
        metrics_path=args.metrics_path,
        model_path=args.model_path,
        per_caption=args.per_caption,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _parse_args_and_run()
