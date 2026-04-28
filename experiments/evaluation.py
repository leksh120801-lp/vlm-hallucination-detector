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
from utils.visualization import save_heatmap, show_heatmap
from utils.preprocessing import pil_to_cv2
from utils.caption_attack import generate_adversarial_captions
from utils.llm_attack import generate_llm_attacks
from utils.metrics import compute_metrics, confusion_matrix
from utils.plots import plot_metrics
from utils.real_heatmap import generate_clip_heatmap





def load_dataset_sample(size=2):

    dataset = load_dataset(
        "yerevann/coco-karpathy",
        split=f"test[:{size}]"
    )

    return dataset


def run_dataset_evaluation():

    model, processor, device = load_clip_model()

    dataset = load_dataset_sample()

    results = []

    for i, sample in enumerate(dataset):

        caption = sample["sentences"][0]

        image = load_image_from_url(sample["url"])

        image_cv = pil_to_cv2(image)
       

        heatmap = generate_clip_heatmap(
            model,
            processor,
            image,
            caption
        )
        show_heatmap(image_cv, heatmap)
        save_heatmap(image_cv, heatmap, f"experiments/results/heatmap_{i}.jpg")

        captions = [caption] + generate_llm_attacks(caption)

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

        for j, test_caption in enumerate(captions):

            score = similarity[0][j].item()

            decision = detect_hallucination(score)

            ground_truth = "correct" if j == 0 else "hallucination"

            results.append({
                "original_caption": caption,
                "test_caption": test_caption,
                "score": score,
                "decision": decision,
                "ground_truth": ground_truth,
            })

    return results


from datetime import datetime

def save_results(results, metrics):

    import json
    from datetime import datetime

    output = {
        "metrics": metrics,
        "results": results
    }

    filename = f"experiments/results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(output, f, indent=4)

    print("Saved results to", filename)

if __name__ == "__main__":

    results = run_dataset_evaluation()

    metrics = compute_metrics(results)

    matrix = confusion_matrix(results)

    print(metrics)
    print(matrix)

    plot_metrics(metrics)

    save_results(results, metrics)