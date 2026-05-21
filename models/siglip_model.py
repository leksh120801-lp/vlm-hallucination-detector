"""SigLIP backbone loader.

Loads ``google/siglip-base-patch16-224`` with the modern fast image
processor and optional ``HF_TOKEN`` authentication.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoProcessor, SiglipModel


def load_siglip_model():
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = "google/siglip-base-patch16-224"

    # ``use_fast=True`` silences the SigLIP processor migration warning
    # ("`use_fast=True` will be the default behavior in v5...").
    processor = AutoProcessor.from_pretrained(
        checkpoint, token=token, use_fast=True,
    )
    model = SiglipModel.from_pretrained(checkpoint, token=token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, processor, device
