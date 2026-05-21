"""CLIP backbone loader.

Loads ``openai/clip-vit-base-patch32`` plus the matching processor.
Honours the ``HF_TOKEN`` environment variable for access to gated repos
without breaking when it isn't set.
"""

from __future__ import annotations

import os

import torch
from transformers import CLIPModel, CLIPProcessor


def load_clip_model():
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = "openai/clip-vit-base-patch32"

    # ``use_fast`` picks the Rust-backed image processor — same outputs, less
    # log noise from the slow-tokenizer fallback.
    processor = CLIPProcessor.from_pretrained(
        checkpoint, token=token, use_fast=True,
    )
    model = CLIPModel.from_pretrained(checkpoint, token=token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, processor, device
