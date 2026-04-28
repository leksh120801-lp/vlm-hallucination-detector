"""Unit tests for the threshold-config loader."""

from __future__ import annotations

from utils.config import get_threshold


def test_clip_default_threshold():
    assert get_threshold("CLIP") == 0.25


def test_siglip_default_threshold_is_lower():
    """SigLIP's score scale is tighter; its default threshold should reflect that."""
    assert get_threshold("SigLIP") < get_threshold("CLIP")


def test_unknown_model_falls_back_to_safe_default():
    """Unknown backbone names must not raise; they fall back to 0.25."""
    assert get_threshold("not_a_real_model") == 0.25
