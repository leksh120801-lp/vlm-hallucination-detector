"""Feature engineering and persistence for the logistic-regression detector.

Two feature representations are exposed:

  - ``extract_features``       — 4-D, per caption: (original, mean_attack, consistency, variance).
  - ``extract_image_features`` — 8-D, per image:   adds max/min/std/diff_to_max/is_top.

Both work over *similarity scores*, not raw embeddings. This keeps the feature
dimensionality tiny (4 or 8 features) and is far more sample-efficient than
training a head over 2048-D image+text embedding concatenations — important for
the small adversarial datasets typical in this project.

Models are persisted with ``joblib`` so we get fast pickle, sklearn-friendly,
and a place to stash metadata alongside the estimator.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pvariance

import joblib

# Default location for the persisted estimator.
MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "logistic_model.pkl"
)


def extract_features(original_score, attack_scores):
    """Per-caption 4-D feature vector.

    Returns ``[original_score, mean_attack_score, consistency_score, variance]``
    where ``consistency_score = original - mean_attack``.
    """
    attack_values = [float(score) for score in attack_scores]
    mean_attack_score = mean(attack_values) if attack_values else 0.0
    variance_attack_score = pvariance(attack_values) if len(attack_values) > 1 else 0.0
    consistency_score = float(original_score) - mean_attack_score

    return [
        float(original_score),
        mean_attack_score,
        consistency_score,
        variance_attack_score,
    ]


def extract_image_features(original_score, attack_scores):
    """Per-image 8-D feature vector — richer than ``extract_features``.

    Returns (in order):
      - ``original_score``
      - ``mean_attack_score``
      - ``max_attack_score``
      - ``min_attack_score``
      - ``std_attack_score``
      - ``consistency_score``  (original - mean_attack)
      - ``diff_to_max``        (original - max_attack)
      - ``is_original_top``    (1.0 if original >= max_attack else 0.0)

    The 4-D ``extract_features`` is preserved for backwards compatibility.
    """
    import math

    attack_values = [float(score) for score in attack_scores]
    mean_attack = mean(attack_values) if attack_values else 0.0
    max_attack = max(attack_values) if attack_values else 0.0
    min_attack = min(attack_values) if attack_values else 0.0
    std_attack = math.sqrt(pvariance(attack_values)) if len(attack_values) > 1 else 0.0
    consistency = float(original_score) - mean_attack
    diff_to_max = float(original_score) - max_attack
    is_top = 1.0 if float(original_score) >= max_attack else 0.0

    return [
        float(original_score),
        mean_attack,
        max_attack,
        min_attack,
        std_attack,
        consistency,
        diff_to_max,
        is_top,
    ]


def load_logistic_model(model_path=MODEL_PATH):
    """Load the persisted logistic estimator.

    Backwards compatible: if the saved object is a payload dict with keys
    ``{"model", "metadata"}``, returns the inner model; otherwise returns the
    object as-is.
    """
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def load_logistic_metadata(model_path=MODEL_PATH):
    """Load metadata saved alongside the model (or return ``None``)."""
    try:
        obj = joblib.load(model_path)
    except Exception:
        return None

    if isinstance(obj, dict) and "metadata" in obj:
        return obj["metadata"]
    return None
