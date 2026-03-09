from models.clip_model import load_clip_model
from utils.preprocessing import load_image
from utils.similarity import compute_similarity
from utils.similarity import compute_similarity, save_results
from utils.similarity import compute_similarity, save_results, detect_hallucination
from utils.visualization import show_attention_heatmap


import torch


def main():

    # Load model
    model, processor, device = load_clip_model()

    # Load image
    image_path = "data/raw/test.jpg"
    image = load_image(image_path)

    # Example captions
    captions = [
        "a dog sitting on grass",
        "a cat sitting on grass",
        "a car parked on road"
    ]

    # Prepare inputs
    inputs = processor(
        text=captions,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    image_embedding = outputs.image_embeds
    text_embedding = outputs.text_embeds

    similarity = compute_similarity(image_embedding, text_embedding)
    best_score = similarity[0].max().item()

    show_attention_heatmap(image, best_score)

    print("\nSimilarity Scores:\n")
    decisions = []

    for caption, score in zip(captions, similarity[0]):
        s = score.item()
        decision = detect_hallucination(s)
        decisions.append(decision)
        print(f"{caption} : {s:.4f} → {decision}")
    scores = similarity[0].cpu().numpy()

    save_results(image_path, captions, scores, decisions)

if __name__ == "__main__":
    main()