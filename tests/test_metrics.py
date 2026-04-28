"""Unit tests for utils.metrics.

These tests pin the post-fix behaviour of ``compute_metrics`` and
``confusion_matrix``. The previous implementation looked for the literal
string ``"hallucination"`` (which never appeared) and silently kept the
hallucination counter at zero. The tests below would have caught that bug.
"""

from __future__ import annotations

from utils.metrics import (
    DECISION_CORRECT,
    DECISION_HALLUCINATION,
    GROUND_TRUTH_CORRECT,
    GROUND_TRUTH_HALLUCINATION,
    compute_metrics,
    confusion_matrix,
)


def _row(decision: str, ground_truth: str) -> dict:
    return {"decision": decision, "ground_truth": ground_truth}


def test_compute_metrics_empty_inputs():
    m = compute_metrics([])
    assert m["total_samples"] == 0
    assert m["accuracy"] == 0.0
    assert m["hallucination_rate"] == 0.0


def test_compute_metrics_counts_both_classes():
    """Pre-fix this counter was always 0 because of a string mismatch."""
    rows = [
        _row(DECISION_CORRECT, GROUND_TRUTH_CORRECT),
        _row(DECISION_CORRECT, GROUND_TRUTH_CORRECT),
        _row(DECISION_HALLUCINATION, GROUND_TRUTH_HALLUCINATION),
        _row(DECISION_HALLUCINATION, GROUND_TRUTH_HALLUCINATION),
    ]
    m = compute_metrics(rows)
    assert m["correct"] == 2
    assert m["hallucinations"] == 2
    assert m["accuracy"] == 0.5
    assert m["hallucination_rate"] == 0.5


def test_compute_metrics_unknown_decision_is_surfaced():
    rows = [
        _row(DECISION_CORRECT, GROUND_TRUTH_CORRECT),
        _row("garbage_label", GROUND_TRUTH_CORRECT),
    ]
    m = compute_metrics(rows)
    assert m["correct"] == 1
    assert m["hallucinations"] == 0
    assert m["unknown_decisions"] == 1


def test_confusion_matrix_balanced_run():
    """Synthetic 8-row run with a known confusion matrix."""
    rows = [
        # 2 TP — gt correct, predicted correct
        _row(DECISION_CORRECT, GROUND_TRUTH_CORRECT),
        _row(DECISION_CORRECT, GROUND_TRUTH_CORRECT),
        # 2 FN — gt correct, predicted hallucination
        _row(DECISION_HALLUCINATION, GROUND_TRUTH_CORRECT),
        _row(DECISION_HALLUCINATION, GROUND_TRUTH_CORRECT),
        # 3 TN — gt hallucination, predicted hallucination
        _row(DECISION_HALLUCINATION, GROUND_TRUTH_HALLUCINATION),
        _row(DECISION_HALLUCINATION, GROUND_TRUTH_HALLUCINATION),
        _row(DECISION_HALLUCINATION, GROUND_TRUTH_HALLUCINATION),
        # 1 FP — gt hallucination, predicted correct
        _row(DECISION_CORRECT, GROUND_TRUTH_HALLUCINATION),
    ]
    cm = confusion_matrix(rows)
    assert (cm["TP"], cm["FP"], cm["TN"], cm["FN"]) == (2, 1, 3, 2)
    assert abs(cm["precision"] - 2 / 3) < 1e-9
    assert abs(cm["recall"] - 0.5) < 1e-9
    assert abs(cm["f1"] - (2 * (2 / 3) * 0.5) / ((2 / 3) + 0.5)) < 1e-9


def test_confusion_matrix_zero_division_is_safe():
    cm = confusion_matrix([])
    assert cm["precision"] == 0.0
    assert cm["recall"] == 0.0
    assert cm["f1"] == 0.0


def test_confusion_matrix_ignores_unknown_labels():
    rows = [
        _row(DECISION_CORRECT, GROUND_TRUTH_CORRECT),
        _row("???", "???"),
    ]
    cm = confusion_matrix(rows)
    assert (cm["TP"], cm["FP"], cm["TN"], cm["FN"]) == (1, 0, 0, 0)
