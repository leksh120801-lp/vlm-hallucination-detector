import torch
import numpy as np


def generate_clip_heatmap(model, processor, image, caption):

    inputs = processor(
        text=[caption],
        images=image,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)

    image_embeds = outputs.image_embeds

    heatmap = image_embeds.detach().cpu().numpy()

    heatmap = heatmap.mean(axis=-1)

    heatmap = heatmap / heatmap.max()

    return heatmap