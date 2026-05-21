"""Run a small dataset evaluation: CLIP scores, heatmaps, metrics, JSON dump.

Reads a tiny COCO Karpathy slice, generates LLM-based adversarial captions per
image, scores everything with CLIP, writes per-image attention heatmaps, and
saves a timestamped JSON of the metrics.

Run::

    python experiments/evaluation.py
    python experiments/evaluation.py --size 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from models.clip_model import load_clip_model
from utils.llm_attack import generate_llm_attacks
from utils.logging_config import get_logger
from utils.metrics import compute_metrics, confusion_matrix
from utils.plots import plot_metrics
from utils.preprocessing import load_image_from_url, pil_to_cv2
from utils.real_heatmap import generate_clip_heatmap
from utils.similarity import compute_similarity, detect_hallucination
from utils.visualization import save_heatmap

logger = get_logger(__name__)
RESULTS_DIR = ROOT / "experiments" / "results"


def load_dataset_sample(size: int = 2):
    return load_dataset("yerevann/coco-karpathy", split=f"test[:{size}]")


def run_dataset_evaluation(size: int = 2, threshold: float = 0.25) -> list[dict]:
    model, processor, device = load_clip_model()
    dataset = load_dataset_sample(size=size)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for i, sample in enumerate(dataset):
        caption = sample["sentences"][0]

        try:
            image = load_image_from_url(sample["url"])
        except Exception as exc:
            logger.warning("Skipping sample %d (image fetch failed): %s", i, exc)
            continue

        image_cv = pil_to_cv2(image)

        heatmap = generate_clip_heatmap(model, processor, image, caption)
        if heatmap is not None:
            save_heatmap(image_cv, heatmap, str(RESULTS_DIR / f"heatmap_{i}.jpg"))

        candidate_captions = [caption] + generate_llm_attacks(caption)
        inputs = processor(
            text=candidate_captions, images=image, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        similarity = compute_similarity(outputs.image_embeds, outputs.text_embeds)
        for j, test_caption in enumerate(candidate_captions):
            score = float(similarity[0][j].item())
            decision = detect_hallucination(score, threshold=threshold)
            results.append({
                "original_caption": caption,
                "test_caption": test_caption,
                "score": score,
                "decision": decision,
                "ground_truth": "correct" if j == 0 else "hallucination",
            })

    return results


def save_eval_results(results: list[dict], metrics: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with filename.open("w") as fh:
        json.dump({"metrics": metrics, "results": results}, fh, indent=4)
    logger.info("Saved results to %s", filename)
    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    results = run_dataset_evaluation(size=args.size, threshold=args.threshold)
    metrics = compute_metrics(results)
    matrix = confusion_matrix(results)

    logger.info("Metrics: %s", metrics)
    logger.info("Confusion matrix: %s", matrix)

    plot_metrics(metrics)
    save_eval_results(results, metrics)


if __name__ == "__main__":
    main()
