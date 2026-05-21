"""BLIP backbone loader.

Uses ``BlipForConditionalGeneration`` (the modern, non-deprecated class)
together with ``BlipProcessor``. To preserve the ``model(**inputs)`` →
``outputs.image_embeds / text_embeds`` contract that the rest of the codebase
relies on (similarity scoring, heatmaps), we wrap the underlying model in a
thin adapter that:

  - exposes ``model.vision_model`` so ``utils.real_heatmap`` works unchanged,
  - on ``forward(...)``, returns a ``SimpleNamespace`` with ``image_embeds``
    (vision-model pooler output) and ``text_embeds`` (CLS token of the BLIP
    text encoder used in encoder mode, *not* decoder mode).

This eliminates the "Some weights were not initialized..." warning that
``BlipModel`` produced when loaded from the captioning checkpoint, while
keeping every call site of the form ``model(**inputs).image_embeds`` valid.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import BlipForConditionalGeneration, BlipProcessor


class _BlipSimilarityAdapter(nn.Module):
    """Wraps a ``BlipForConditionalGeneration`` to expose CLIP-style outputs.

    Forward call produces:
      - ``image_embeds``: ``(B, hidden)`` — pooler output of the vision tower.
      - ``text_embeds``:  ``(B, hidden)`` — CLS embedding from the BLIP text
        encoder. Calling ``self.blip.text_decoder.bert`` *without* an
        ``encoder_hidden_states`` argument runs it as a plain bidirectional
        encoder (i.e. ``is_decoder=False``), exactly what we need.
    """

    def __init__(self, blip_model: BlipForConditionalGeneration):
        super().__init__()
        self.blip = blip_model
        # Keep ``vision_model`` reachable so ``utils.real_heatmap`` works unchanged.
        self.vision_model = blip_model.vision_model

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ):
        # Image side — pooler output of the vision transformer.
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output

        # Text side — CLS token of the text encoder, in encoder mode.
        text_outputs = self.blip.text_decoder.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs.last_hidden_state[:, 0, :]

        return SimpleNamespace(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            vision_model_output=vision_outputs,
        )


def load_blip_model():
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(checkpoint, token=token)
    base_model = BlipForConditionalGeneration.from_pretrained(checkpoint, token=token)
    base_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _BlipSimilarityAdapter(base_model).to(device)
    model.eval()

    return model, processor, device
