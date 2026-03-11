from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from models.clip_model import load_clip_model
from utils.similarity import compute_similarity, detect_hallucination
import torch
from utils.preprocessing import load_image_from_url
import json
from utils.visualization import save_heatmap, show_heatmap, generate_fake_heatmap
from utils.preprocessing import pil_to_cv2





def load_dataset_sample(size=5):

    dataset = load_dataset(
        "yerevann/coco-karpathy",
        split=f"test[:{size}]"
    )

    return dataset

dataset = load_dataset("yerevann/coco-karpathy", split="test[:5]")

print(dataset.column_names)
print(dataset[0])


def run_dataset_evaluation():

    model, processor, device = load_clip_model()

    dataset = load_dataset_sample()

    results = []

    for i, sample in enumerate(dataset):

        caption = sample["sentences"][0]

        image = load_image_from_url(sample["url"])

        image_cv = pil_to_cv2(image)
        heatmap = generate_fake_heatmap()
        show_heatmap(image_cv, heatmap)
        save_heatmap(image_cv, heatmap, f"experiments/results/heatmap_{i}.jpg")

        captions = [
            caption,
            "a spaceship flying in space",
            "a dinosaur eating grass"
        ]

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

        score = similarity[0][0].item()

        decision = detect_hallucination(score)

        print(caption, score, decision)

        results.append({
            "caption": caption,
            "score": score,
            "decision": decision
        })

    return results


def compute_metrics(results):

    total = len(results)
    correct = 0

    for r in results:
        if r["decision"] == "likely correct":
            correct += 1

    accuracy = correct / total if total > 0 else 0

    print("\nEvaluation Metrics")
    print("------------------")
    print("Total:", total)
    print("Correct:", correct)
    print("Accuracy:", accuracy)

    return accuracy


from datetime import datetime

def save_results(results, accuracy):

    output = {
        "dataset": "COCO",
        "total_samples": len(results),
        "accuracy": accuracy,
        "results": results
    }

    filename = f"experiments/results/coco_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(output, f, indent=4)

    print("Saved results to", filename)

if __name__ == "__main__":

    results = run_dataset_evaluation()

    compute_metrics(results)

    save_results(results,accuracy=compute_metrics(results)) 