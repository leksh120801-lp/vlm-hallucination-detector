from models.model_registry import load_model_by_name
from utils.datasets import load_flickr_sample
from utils.similarity import compute_similarity, detect_hallucination
from utils.metrics import compute_metrics
from utils.caption_attack import generate_adversarial_captions

import torch
import pandas as pd


MODELS = ["CLIP", "BLIP", "SigLIP"]


def evaluate_model(model_name, samples):

    model, processor, device = load_model_by_name(model_name)

    results = []

    for sample in samples:

        image = sample["image"]
        caption = sample["caption"]

        captions = [caption] + generate_adversarial_captions(caption)

        inputs = processor(
            text=captions,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        similarity = compute_similarity(
            outputs.image_embeds,
            outputs.text_embeds
        )

        for i, cap in enumerate(captions):

            score = similarity[0][i].item()
            decision = detect_hallucination(score)

            ground_truth = "correct" if i == 0 else "hallucination"

            results.append({
                "caption": cap,
                "score": score,
                "decision": decision,
                "ground_truth": ground_truth
            })

    return compute_metrics(results)


def run():

    samples = load_flickr_sample(30)

    all_results = []

    for model_name in MODELS:

        metrics = evaluate_model(model_name, samples)

        metrics["model"] = model_name

        all_results.append(metrics)

    df = pd.DataFrame(all_results)

    print(df)

    df.to_csv("experiments/results/benchmark.csv", index=False)


if __name__ == "__main__":
    run()