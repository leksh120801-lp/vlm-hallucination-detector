import torch
from transformers import BlipProcessor, BlipModel


def load_blip_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipModel.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    model.eval()

    return model, processor, device