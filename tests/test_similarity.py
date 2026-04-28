"""Unit tests for utils.similarity.

These tests do not load any HuggingFace models. They use random / hand-built
PyTorch tensors so the suite is fast and offline-safe.
"""

from __future__ import annotations

import torch

from utils.similarity import compute_similarity, detect_hallucination


def test_cosine_similarity_self_is_one():
    """A vector compared to itself must yield cosine similarity 1.0."""
    v = torch.randn(4, 64)
    sim = compute_similarity(v, v)
    diag = torch.diagonal(sim)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_cosine_similarity_orthogonal_is_zero():
    v = torch.tensor([[1.0, 0.0]])
    t = torch.tensor([[0.0, 1.0]])
    assert compute_similarity(v, t).item() == 0.0


def test_cosine_similarity_antiparallel_is_minus_one():
    v = torch.tensor([[1.0, 0.0]])
    t = torch.tensor([[-1.0, 0.0]])
    assert compute_similarity(v, t).item() == -1.0


def test_cosine_similarity_shape():
    v = torch.randn(3, 32)  # 3 images
    t = torch.randn(5, 32)  # 5 captions
    sim = compute_similarity(v, t)
    assert sim.shape == (3, 5)


def test_cosine_similarity_invariant_to_scale():
    """Cosine similarity must ignore the magnitude of the inputs."""
    v = torch.tensor([[1.0, 2.0, 3.0]])
    t = torch.tensor([[1.0, 2.0, 3.0]])
    base = compute_similarity(v, t).item()
    scaled = compute_similarity(v * 7.5, t * 0.3).item()
    assert abs(base - scaled) < 1e-5


def test_detect_hallucination_above_threshold():
    assert detect_hallucination(0.30, threshold=0.25) == "likely correct"


def test_detect_hallucination_at_threshold_is_correct():
    """The boundary is inclusive: score == threshold counts as correct."""
    assert detect_hallucination(0.25, threshold=0.25) == "likely correct"


def test_detect_hallucination_below_threshold():
    assert detect_hallucination(0.10, threshold=0.25) == "possible hallucination"


def test_detect_hallucination_uses_config_when_model_name_passed():
    """When ``model_name`` is provided, the threshold is read from the YAML config.

    SigLIP's default in the YAML is 0.10, so a score of 0.15 must be classified
    as ``likely correct`` for SigLIP but as a hallucination for CLIP (default 0.25).
    """
    assert detect_hallucination(0.15, model_name="SigLIP") == "likely correct"
    assert detect_hallucination(0.15, model_name="CLIP") == "possible hallucination"
