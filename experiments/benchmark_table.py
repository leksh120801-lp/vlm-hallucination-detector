"""Generate a markdown performance-benchmark table for the README.

Runs each backbone (CLIP / BLIP / SigLIP) on a small labeled mini-set:
  - "correct" rows = real (image, caption) pairs from the curated fallback list.
  - "hallucinated" rows = the same image paired with an object-swapped caption.

Computes accuracy, precision, recall, F1 per backbone and prints a markdown
table. Saves the same table to ``experiments/results/benchmark_table.md`` and
the raw rows to ``experiments/results/benchmark_table.json``.

Run:

    python experiments/benchmark_table.py
    python experiments/benchmark_table.py --n 30 --threshold-clip 0.25 --threshold-siglip 0.10

This is intentionally small and CPU-friendly so it runs as a smoke check; for
publishable numbers, point it at POPE / CHAIR / AMBER (see IMPROVEMENTS.md §5).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make `from utils...`, `from models...` work whether or not the package
# is installed.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.model_registry import load_model_by_name
from utils.caption_attack import object_swap_attack
from utils.config import get_threshold
from utils.datasets import _materialise_fallback  # type: ignore[attr-defined]
from utils.metrics import (
    DECISION_CORRECT,
    DECISION_HALLUCINATION,
    GROUND_TRUTH_CORRECT,
    GROUND_TRUTH_HALLUCINATION,
    compute_metrics,
    confusion_matrix,
)
from utils.similarity import compute_similarity, detect_hallucination

MODELS = ["CLIP", "BLIP", "SigLIP"]


def _build_eval_set(n: int) -> list[dict]:
    """Build a balanced (image, caption, ground_truth) set."""
    base = _materialise_fallback(n, source="benchmark")
    eval_set: list[dict] = []
    for sample in base:
        # Original caption is by definition correct.
        eval_set.append({
            "image": sample["image"],
            "caption": sample["caption"],
            "ground_truth": GROUND_TRUTH_CORRECT,
        })
        # Object-swap perturbation = hallucinated caption (label is approximate;
        # see IMPROVEMENTS.md for a discussion of this assumption).
        adv = object_swap_attack(sample["caption"])
        if adv != sample["caption"]:  # only include if something actually changed
            eval_set.append({
                "image": sample["image"],
                "caption": adv,
                "ground_truth": GROUND_TRUTH_HALLUCINATION,
            })
    return eval_set


def _evaluate(model_name: str, eval_set: list[dict], threshold: float) -> dict:
    """Run one backbone over the entire eval set and aggregate metrics."""
    model, processor, device = load_model_by_name(model_name)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for sample in eval_set:
        inputs = processor(
            text=[sample["caption"]],
            images=sample["image"],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        sim = compute_similarity(outputs.image_embeds, outputs.text_embeds)
        score = float(sim[0][0].item())
        decision = detect_hallucination(score, threshold=threshold)
        rows.append({
            "caption": sample["caption"],
            "score": score,
            "decision": decision,
            "ground_truth": sample["ground_truth"],
        })

    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(eval_set), 1)

    metrics = compute_metrics(rows)
    cm = confusion_matrix(rows)
    return {
        "model": model_name,
        "threshold": threshold,
        "samples": len(rows),
        "accuracy": metrics["accuracy"],
        "precision": cm["precision"],
        "recall": cm["recall"],
        "f1": cm["f1"],
        "avg_latency_ms": latency_ms,
        "rows": rows,
    }


def _to_markdown_table(results: list[dict]) -> str:
    header = (
        "| Backbone | τ | Samples | Accuracy | Precision | Recall | F1 | Latency / sample |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in results:
        lines.append(
            f"| {r['model']} | {r['threshold']:.2f} | {r['samples']} | "
            f"{r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['avg_latency_ms']:.1f} ms |"
        )
    return header + "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="Number of base images.")
    parser.add_argument("--threshold-clip", type=float, default=None)
    parser.add_argument("--threshold-blip", type=float, default=None)
    parser.add_argument("--threshold-siglip", type=float, default=None)
    args = parser.parse_args()

    eval_set = _build_eval_set(args.n)
    print(f"Built balanced eval set of {len(eval_set)} samples "
          f"({sum(1 for s in eval_set if s['ground_truth']==GROUND_TRUTH_CORRECT)} correct, "
          f"{sum(1 for s in eval_set if s['ground_truth']==GROUND_TRUTH_HALLUCINATION)} adversarial).")

    results = []
    for m in MODELS:
        threshold = (
            getattr(args, f"threshold_{m.lower()}")
            or get_threshold(m)
        )
        print(f"\n→ Evaluating {m} (τ={threshold:.2f}) …")
        results.append(_evaluate(m, eval_set, threshold))

    md = _to_markdown_table(results)
    print("\n--- Benchmark table ---\n")
    print(md)

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = out_dir / f"benchmark_table_{stamp}.md"
    md_path.write_text(md)

    json_path = out_dir / f"benchmark_table_{stamp}.json"
    json_path.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "rows"} for r in results],
        indent=2,
    ))

    print(f"Saved markdown to {md_path}")
    print(f"Saved metrics  to {json_path}")


if __name__ == "__main__":
    main()
