"""Single-image, multi-caption CLI demo.

Usage::

    python main.py
    python main.py --image data/raw/test.jpg --captions "a dog on grass" "a cat on grass"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.clip_model import load_clip_model
from utils.logging_config import get_logger
from utils.preprocessing import load_image
from utils.similarity import compute_similarity, detect_hallucination, save_results
from utils.visualization import show_heatmap  # noqa: F401  (kept for ad-hoc use)

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/raw/test.jpg"),
        help="Path to the input image.",
    )
    parser.add_argument(
        "--captions",
        nargs="+",
        default=[
            "a dog sitting on grass",
            "a cat sitting on grass",
            "a car parked on road",
        ],
        help="Candidate captions to score against the image.",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.image.exists():
        logger.error("Image not found at %s — pass --image with a real path.", args.image)
        raise SystemExit(2)

    model, processor, device = load_clip_model()
    image = load_image(str(args.image))

    inputs = processor(
        text=args.captions, images=image, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    sims = compute_similarity(outputs.image_embeds, outputs.text_embeds)

    decisions: list[str] = []
    print("\nSimilarity scores:\n")
    for caption, score_tensor in zip(args.captions, sims[0]):
        score = float(score_tensor.item())
        decision = detect_hallucination(score, threshold=args.threshold)
        decisions.append(decision)
        print(f"  {caption:40s}  {score: .4f}  → {decision}")

    save_results(
        str(args.image),
        args.captions,
        sims[0].cpu().numpy(),
        decisions,
    )


if __name__ == "__main__":
    main()
