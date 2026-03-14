import torch
from transformers import AutoProcessor, AutoModel


def load_siglip_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        "google/siglip-base-patch16-224"
    )

    model = AutoModel.from_pretrained(
        "google/siglip-base-patch16-224"
    ).to(device)

    model.eval()

    return model, processor, device