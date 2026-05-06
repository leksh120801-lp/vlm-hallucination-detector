"""Compare CLIP / BLIP / SigLIP on a small Flickr30k slice (cosine baseline).

Writes ``experiments/results/benchmark.csv`` with one row per backbone.
For the richer cross-method comparison, use ``experiments/evaluate_methods.py``.

Run::

    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --n 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_registry import load_model_by_name
from utils.caption_attack import generate_adversarial_captions
from utils.datasets import load_flickr_sample
from utils.logging_config import get_logger
from utils.metrics import compute_metrics
from utils.similarity import compute_similarity, detect_hallucination

logger = get_logger(__name__)
MODELS = ["CLIP", "BLIP", "SigLIP"]


def evaluate_model(model_name: str, samples: list[dict]) -> dict:
    model, processor, device = load_model_by_name(model_name)
    rows = []

    for sample in samples:
        captions = [sample["caption"]] + generate_adversarial_captions(sample["caption"])
        inputs = processor(
            text=captions, images=sample["image"], return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        sims = compute_similarity(outputs.image_embeds, outputs.text_embeds)[0]
        for i, cap in enumerate(captions):
            score = float(sims[i].item())
            rows.append({
                "caption": cap,
                "score": score,
                "decision": detect_hallucination(score),
                "ground_truth": "correct" if i == 0 else "hallucination",
            })

    return compute_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "results" / "benchmark.csv",
    )
    args = parser.parse_args()

    samples = load_flickr_sample(args.n)
    if not samples:
        logger.error("No samples loaded — aborting benchmark.")
        raise SystemExit(2)

    rows = []
    for model_name in MODELS:
        logger.info("Evaluating %s on %d samples", model_name, len(samples))
        metrics = evaluate_model(model_name, samples)
        metrics["model"] = model_name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("Wrote benchmark to %s", args.out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
