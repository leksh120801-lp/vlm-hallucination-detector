from transformers import CLIPProcessor, CLIPModel
import torch

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, processor, device

#load CLIP
#move it to GPU if available