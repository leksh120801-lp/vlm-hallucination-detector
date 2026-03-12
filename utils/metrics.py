import numpy as np


def compute_metrics(results):

    correct = 0
    hallucinations = 0
    total = len(results)

    for r in results:

        if r["decision"] == "likely correct":
            correct += 1

        if r["decision"] == "hallucination":
            hallucinations += 1

    accuracy = correct / total if total > 0 else 0
    hallucination_rate = hallucinations / total if total > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "total_samples": total
    }

    return metrics

def confusion_matrix(results):

    TP = FP = TN = FN = 0

    for r in results:

        gt = r["ground_truth"]
        pred = r["decision"]

        if gt == "correct" and pred == "likely correct":
            TP += 1
        elif gt == "correct" and pred == "hallucination":
            FN += 1
        elif gt == "hallucination" and pred == "hallucination":
            TN += 1
        elif gt == "hallucination" and pred == "likely correct":
            FP += 1

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
    }