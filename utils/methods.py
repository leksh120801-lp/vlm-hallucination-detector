"""Three pluggable hallucination-detection methods, on a uniform interface.

Every method below returns the same shape:

    decision_label, optional_signal_score

with ``decision_label`` ∈ ``{"likely correct", "hallucination"}``. This lets
the evaluation harness in ``experiments/evaluate_methods.py`` and the Streamlit
frontend treat the three methods identically and compare them apples-to-apples.

Methods:

  - ``method_threshold``             — classic cosine ≥ τ baseline.
  - ``method_consistency_threshold`` — original score should beat the mean
                                        adversarial score by ``τ``.
  - ``method_logistic``              — learned logistic-regression detector
                                        over similarity-score statistics.

Note on labels: ``utils.similarity.detect_hallucination`` historically returned
either ``"likely correct"`` or ``"possible hallucination"`` (from the original
detector) or ``"hallucination"`` (from the alternate fork that produced this
module). The ``_normalize_decision`` helper accepts either form and folds them
to the canonical pair ``{"likely correct", "hallucination"}`` used here.
"""

from __future__ import annotations

from statistics import mean

from utils.classifier import extract_features, extract_image_features
from utils.similarity import detect_hallucination

LIKELY_CORRECT = "likely correct"
HALLUCINATION = "hallucination"


def _normalize_decision(decision: str) -> str:
    """Fold either label form ('possible hallucination' or 'hallucination') to one of {LIKELY_CORRECT, HALLUCINATION}."""
    return HALLUCINATION if "hallucination" in decision.lower() else LIKELY_CORRECT


# ---------------------------------------------------------------------------
# Method 1 — classic cosine threshold.
# ---------------------------------------------------------------------------

def method_threshold(score, threshold: float = 0.25) -> str:
    """Threshold baseline wrapped around the existing detector."""
    return _normalize_decision(detect_hallucination(score, threshold=threshold))


# ---------------------------------------------------------------------------
# Method 2 — consistency: how much does the real caption beat its adversaries?
# ---------------------------------------------------------------------------

def method_consistency_threshold(original_score, attack_scores, threshold: float = 0.2):
    """Compare the original caption score against the mean adversarial score.

    Intuition: if the model truly understands the image, the real caption
    should score *higher* than caption-perturbed versions. If it doesn't, the
    "real" caption is likely a hallucination.

    Returns ``(decision_label, consistency_score)``.
    """
    attack_values = [float(score) for score in attack_scores]
    mean_attack_score = mean(attack_values) if attack_values else 0.0
    consistency_score = float(original_score) - mean_attack_score

    if consistency_score >= threshold:
        return LIKELY_CORRECT, consistency_score

    return HALLUCINATION, consistency_score


# ---------------------------------------------------------------------------
# Method 3 — learned logistic head over similarity-score statistics.
# ---------------------------------------------------------------------------

def method_logistic(model, original_score, attack_scores):
    """Run the trained logistic detector.

    Auto-detects whether the model was trained on the 4-D or 8-D feature set
    (by inspecting ``n_features_in_`` on the estimator or pipeline) and
    chooses the right feature vector accordingly. If detection fails or the
    model raises a feature-dimension error, falls back to the 4-D vector and
    finally to a consistency-threshold decision.

    Returns ``(decision_label, consistency_signal)``.
    """
    feats_4 = extract_features(original_score, attack_scores)
    feats_8 = extract_image_features(original_score, attack_scores)

    def _expected_n_features(m):
        try:
            if hasattr(m, "n_features_in_"):
                return int(m.n_features_in_)
            if hasattr(m, "named_steps"):
                for step in m.named_steps.values():
                    if hasattr(step, "n_features_in_"):
                        return int(step.n_features_in_)
                last = list(m.named_steps.values())[-1]
                if hasattr(last, "n_features_in_"):
                    return int(last.n_features_in_)
        except Exception:
            pass
        return None

    expected = _expected_n_features(model) if model is not None else None

    if expected == len(feats_8):
        features = feats_8
    elif expected == len(feats_4):
        features = feats_4
    else:
        features = feats_8  # 8-D is richer; prefer it when shape is unknown.

    try:
        prediction = int(model.predict([features])[0])
    except ValueError:
        # Re-try with the alternate feature width.
        alt = feats_4 if features is feats_8 else feats_8
        try:
            prediction = int(model.predict([alt])[0])
            features = alt
        except Exception:
            # Last resort: fall back to the consistency threshold.
            consistency_score = feats_8[5] if len(feats_8) > 5 else feats_4[2]
            if consistency_score >= 0.2:
                return LIKELY_CORRECT, consistency_score
            return HALLUCINATION, consistency_score

    consistency_score = features[5] if len(features) > 5 else features[2]

    if prediction == 1:
        return LIKELY_CORRECT, consistency_score
    return HALLUCINATION, consistency_score


# ---------------------------------------------------------------------------
# Helper used by the eval harness.
# ---------------------------------------------------------------------------

def split_similarity_scores(scores):
    """Split a list of similarity scores into (original_score, attack_scores)."""
    score_values = [float(score) for score in scores]
    if not score_values:
        raise ValueError("Expected at least one similarity score.")

    return score_values[0], score_values[1:]
