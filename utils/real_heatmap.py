import torch
import numpy as np


def generate_clip_heatmap(model, processor, image, caption):

    inputs = processor(
        text=[caption],
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get last hidden layer of vision transformer
    vision_outputs = outputs.vision_model_output

    patch_embeddings = vision_outputs.last_hidden_state

   # Remove CLS token
    patch_embeddings = patch_embeddings[:, 1:, :]

    heatmap = patch_embeddings.norm(dim=-1)

    # remove batch dimension
    heatmap = heatmap[0]

    # compute grid size
    size = int(np.sqrt(heatmap.shape[0]))

    heatmap = heatmap.reshape(size, size)

    heatmap = heatmap.cpu().numpy()

    heatmap = heatmap / heatmap.max()

    return heatmap

# patch_embeddings → (1, 49, hidden_dim)

# norm(dim=-1)     → (1, 49)

# heatmap[0]       → (49)

# sqrt(49)         → 7

# reshape          → (7,7)