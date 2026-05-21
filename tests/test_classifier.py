"""Unit tests for the similarity-statistic feature engineering."""

from __future__ import annotations

import math

from utils.classifier import extract_features, extract_image_features


def test_extract_features_basic():
    orig = 0.9
    attacks = [0.7, 0.6]
    feats = extract_features(orig, attacks)
    assert isinstance(feats, list)
    assert len(feats) == 4
    # consistency = 0.9 - mean(0.7, 0.6) = 0.9 - 0.65 = 0.25
    assert abs(feats[2] - 0.25) < 1e-6


def test_extract_image_features_rich():
    orig = 0.8
    attacks = [0.5, 0.6, 0.7]
    feats = extract_image_features(orig, attacks)
    assert isinstance(feats, list)
    assert len(feats) == 8
    mean_attack = sum(attacks) / len(attacks)
    assert abs(feats[1] - mean_attack) < 1e-6
    # diff_to_max relationship
    assert math.isclose(feats[6], feats[0] - feats[2], rel_tol=1e-6) or feats[6] == feats[0] - feats[2]
    # is_top should be 1.0 if orig >= max(attacks)
    assert feats[7] in (0.0, 1.0)
