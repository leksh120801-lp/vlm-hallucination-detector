"""Sharper tests for ``extract_features`` and ``extract_image_features``."""

from __future__ import annotations

import math

from utils.classifier import extract_features, extract_image_features


def test_extract_features_basic():
    orig = 0.9
    attacks = [0.7, 0.6, 0.8]
    feats = extract_features(orig, attacks)
    # original, mean_attack, consistency, variance
    assert len(feats) == 4
    assert math.isclose(feats[0], orig, rel_tol=1e-6)
    assert math.isclose(feats[1], sum(attacks) / len(attacks), rel_tol=1e-6)
    assert math.isclose(feats[2], orig - feats[1], rel_tol=1e-6)


def test_extract_image_features_basic():
    orig = 0.9
    attacks = [0.7, 0.6, 0.8]
    feats = extract_image_features(orig, attacks)
    # original, mean, max, min, std, consistency, diff_to_max, is_top
    assert len(feats) == 8
    mean_attack = sum(attacks) / len(attacks)
    assert math.isclose(feats[1], mean_attack, rel_tol=1e-6)
    assert math.isclose(feats[2], max(attacks), rel_tol=1e-6)
    assert math.isclose(feats[3], min(attacks), rel_tol=1e-6)
    assert feats[4] >= 0  # std non-negative
    assert feats[5] == feats[0] - feats[1]
    assert feats[6] == feats[0] - feats[2]
    assert feats[7] in (0.0, 1.0)


def test_empty_attacks():
    """Edge case: no adversarial captions — function must not raise."""
    orig = 0.5
    f1 = extract_features(orig, [])
    f2 = extract_image_features(orig, [])
    assert f1[1] == 0.0
    assert f2[1] == 0.0
    assert f2[2] == 0.0
    assert f2[4] == 0.0
