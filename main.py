from models.clip_model import load_clip_model
from utils.preprocessing import load_image
from utils.similarity import compute_similarity
from utils.similarity import compute_similarity, save_results

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

    print("\nSimilarity Scores:\n")

    for caption, score in zip(captions, similarity[0]):
        print(f"{caption} : {score.item():.4f}")
    scores = similarity[0].cpu().numpy()

    save_results(image_path, captions, scores)

if __name__ == "__main__":
    main()