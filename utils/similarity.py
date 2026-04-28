import torch
import torch.nn.functional as F

def compute_similarity(image_embedding, text_embedding):
    image_embedding = F.normalize(image_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)

    similarity = torch.matmul(image_embedding, text_embedding.T)

    return similarity

#This computes cosine similarity between embeddings.

def detect_hallucination(score, threshold=0.25, model_name=None, dataset=None):
    """Return a decision label for a given similarity ``score``.

    Returns one of:
      - ``"likely correct"``         if ``score >= threshold``
      - ``"possible hallucination"`` otherwise

    Threshold resolution:
      - If ``model_name`` is provided, the threshold is looked up from
        ``configs/thresholds.yaml`` via ``utils.config.get_threshold``,
        which knows that CLIP, BLIP and SigLIP use different score scales.
      - Otherwise, the explicit ``threshold`` argument is used.
      - The historical default of ``0.25`` is preserved for backward
        compatibility with callers that pass neither.
    """
    if model_name is not None:
        # Lazy import to avoid a circular dependency at module-load time.
        from utils.config import get_threshold
        threshold = get_threshold(model_name, dataset=dataset)

    if score >= threshold:
        return "likely correct"
    else:
        return "possible hallucination"


import json
import os
from datetime import datetime


def save_results(image_path, captions, scores, decisions):
    os.makedirs("experiments/results", exist_ok=True)

    results = []

    for caption, score, decision in zip(captions, scores, decisions):
        results.append({
            "caption": caption,
            "similarity_score": float(score),
            "decision": decision
        })

    output = {
        "image": image_path,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

    filename = f"experiments/results/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nResults saved to {filename}")

    # This function saves the similarity scores to a JSON file for later analysis.

    