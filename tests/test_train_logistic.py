"""Smoke test for the logistic-regression training pipeline.

Patches the dataset builders so the test runs offline (no model downloads,
no dataset downloads) and finishes in milliseconds.
"""

from __future__ import annotations

import json


def test_train_logistic_smoke_per_caption(monkeypatch, tmp_path):
    """Default mode (per_caption=True) trains over the 4-D feature builder."""
    from experiments import train_logistic

    # Tiny synthetic per-caption dataset, 4-D features, both classes present.
    X = [
        [0.30, 0.10, 0.20, 0.001],   # original
        [0.05, 0.18, -0.13, 0.001],  # attack
        [0.04, 0.18, -0.14, 0.001],  # attack
        [0.31, 0.11, 0.20, 0.001],   # original
        [0.06, 0.19, -0.13, 0.001],  # attack
        [0.03, 0.19, -0.16, 0.001],  # attack
    ]
    y = [1, 0, 0, 1, 0, 0]

    monkeypatch.setattr(
        train_logistic, "build_training_dataset_per_caption",
        lambda *a, **kw: (X, y),
    )

    out_metrics = tmp_path / "metrics.json"
    out_model = tmp_path / "logistic_model.pkl"
    result = train_logistic.train_logistic_model(
        sample_size=2,
        do_grid_search=False,
        metrics_path=str(out_metrics),
        model_path=str(out_model),
        per_caption=True,
    )

    assert "metrics" in result
    assert result["num_examples"] == 6
    assert out_metrics.exists()
    assert out_model.exists()

    with out_metrics.open("r") as fh:
        metrics = json.load(fh)
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_train_logistic_smoke_per_image(monkeypatch, tmp_path):
    """Per-image mode (--per-image) still works, using the 8-D feature builder."""
    from experiments import train_logistic

    X = [[0.9, 0.1, 0.8, 0.0, 0.1, 1.0, 0.2, 1.0],
         [0.2, 0.3, 0.1, 0.0, 0.15, -0.1, -0.1, 0.0]]
    y = [1, 0]

    monkeypatch.setattr(
        train_logistic, "build_training_dataset",
        lambda *a, **kw: (X, y),
    )

    out_metrics = tmp_path / "metrics.json"
    out_model = tmp_path / "logistic_model.pkl"
    result = train_logistic.train_logistic_model(
        sample_size=2,
        do_grid_search=False,
        metrics_path=str(out_metrics),
        model_path=str(out_model),
        per_caption=False,
    )

    assert result["num_examples"] == 2
    assert out_model.exists()
