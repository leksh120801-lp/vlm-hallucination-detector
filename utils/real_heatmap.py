"""Unified attention heatmap for CLIP / BLIP / SigLIP.

All three models expose a ``vision_model`` sub-module that's a Vision
Transformer. We extract the last hidden layer, drop the CLS token, take the
L2 norm of each remaining patch vector, and reshape to a square grid. That
norm is a cheap proxy for "how much signal does this patch carry" — a
gradient-free, attention-weight-free interpretability map.

Patch grid sizes (for reference):

    Model                    Input   Patch   Grid
    CLIP   (vit-base-p32)    224     32      7x7   = 49  patches
    BLIP   (vit-base-p16)    384     16      24x24 = 576 patches
    SigLIP (base-p16-224)    224     16      14x14 = 196 patches
"""

from __future__ import annotations

import torch

from utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_heatmap(model, processor, image, caption=None, model_name: str = "CLIP"):
    """Generate an attention-style heatmap for any supported VLM.

    Args:
        model:       Loaded HuggingFace model (CLIP / BLIP / SigLIP).
        processor:   Matching processor.
        image:       PIL.Image.
        caption:     Unused; kept for interface compatibility with prior callers.
        model_name:  ``"CLIP" | "BLIP" | "SigLIP"`` — only used in error messages.

    Returns:
        ``np.ndarray`` of shape ``(H, W)`` normalised to ``[0, 1]``, or
        ``None`` on failure.
    """
    try:
        device = next(model.parameters()).device

        # Encode image only — the vision tower doesn't need text for the heatmap.
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values,
                output_hidden_states=True,
            )

        # Drop the CLS token; per-patch L2 norms are the saliency signal.
        last_hidden = vision_outputs.last_hidden_state
        patch_embeddings = last_hidden[:, 1:, :]
        heatmap = patch_embeddings.norm(dim=-1)[0]

        # Reshape into a square grid.
        num_patches = heatmap.shape[0]
        size = int(num_patches ** 0.5)
        heatmap = heatmap[: size * size].reshape(size, size).cpu().numpy()

        # Normalise + boost contrast for prettier overlays.
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap = heatmap ** 3
        return heatmap

    except Exception as exc:  # pragma: no cover — diagnostic, not a hot path.
        logger.warning("Heatmap[%s] failed: %s", model_name, exc)
        return None


# ---------------------------------------------------------------------------
# Backwards-compatible per-model entry points.
# ---------------------------------------------------------------------------

def generate_clip_heatmap(model, processor, image, caption=None):
    """Original CLIP-only entry point — preserved for older callers."""
    return generate_heatmap(model, processor, image, caption, model_name="CLIP")


def generate_blip_heatmap(model, processor, image, caption=None):
    return generate_heatmap(model, processor, image, caption, model_name="BLIP")


def generate_siglip_heatmap(model, processor, image, caption=None):
    return generate_heatmap(model, processor, image, caption, model_name="SigLIP")
