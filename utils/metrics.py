import numpy as np

# ----------------------------------------------------------------------
# Decision label constants — keep these in sync with
# `utils.similarity.detect_hallucination`, which returns one of these
# two exact strings. Earlier versions of this file looked for the
# literal "hallucination", which never matched and silently kept the
# hallucination counter at zero.
# ----------------------------------------------------------------------
DECISION_CORRECT = "likely correct"
DECISION_HALLUCINATION = "possible hallucination"

GROUND_TRUTH_CORRECT = "correct"
GROUND_TRUTH_HALLUCINATION = "hallucination"


def compute_metrics(results):
    """Aggregate decision-level metrics over a list of result dicts.

    Each entry in ``results`` is expected to contain a ``"decision"`` key whose
    value is one of ``DECISION_CORRECT`` or ``DECISION_HALLUCINATION``.
    Unknown decision strings are ignored (and surfaced via ``unknown`` for
    debugging) instead of being silently dropped.
    """
    correct = 0
    hallucinations = 0
    unknown = 0
    total = len(results)

    for r in results:

        decision = r.get("decision")

        if decision == DECISION_CORRECT:
            correct += 1
        elif decision == DECISION_HALLUCINATION:
            hallucinations += 1
        else:
            unknown += 1

    accuracy = correct / total if total > 0 else 0.0
    hallucination_rate = hallucinations / total if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "total_samples": total,
        "correct": correct,
        "hallucinations": hallucinations,
        "unknown_decisions": unknown,
    }

    return metrics


def confusion_matrix(results):
    """Compute a (TP, FP, TN, FN) confusion matrix.

    Convention used here: the *positive* class is "correct caption". So:
        TP = ground truth correct,         predicted correct
        FN = ground truth correct,         predicted hallucination
        TN = ground truth hallucination,   predicted hallucination
        FP = ground truth hallucination,   predicted correct

    Predictions are matched against ``DECISION_CORRECT`` /
    ``DECISION_HALLUCINATION`` (the strings returned by
    ``detect_hallucination``), and ground truth against
    ``GROUND_TRUTH_CORRECT`` / ``GROUND_TRUTH_HALLUCINATION``.
    """
    TP = FP = TN = FN = 0

    for r in results:

        gt = r.get("ground_truth")
        pred = r.get("decision")

        if gt == GROUND_TRUTH_CORRECT and pred == DECISION_CORRECT:
            TP += 1
        elif gt == GROUND_TRUTH_CORRECT and pred == DECISION_HALLUCINATION:
            FN += 1
        elif gt == GROUND_TRUTH_HALLUCINATION and pred == DECISION_HALLUCINATION:
            TN += 1
        elif gt == GROUND_TRUTH_HALLUCINATION and pred == DECISION_CORRECT:
            FP += 1
        # rows with unknown labels are skipped

    # Convenience derived metrics — guarded against zero-division.
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }