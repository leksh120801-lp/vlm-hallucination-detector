import torch
from transformers import AutoProcessor, SiglipModel


def load_siglip_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        "google/siglip-base-patch16-224"
    )

    model = SiglipModel.from_pretrained(
        "google/siglip-base-patch16-224"
    ).to(device)

    model.eval()

    return model, processor, device