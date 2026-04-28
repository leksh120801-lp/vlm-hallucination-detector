"""FastAPI service for the VLM hallucination detector.

Endpoints:
    GET  /health             — liveness probe.
    GET  /                   — service metadata.
    POST /v1/score           — score one image against one or more captions.

The chosen backbone (CLIP / BLIP / SigLIP) is loaded lazily on first use and
then cached in process memory, so subsequent requests do not pay the model-load
cost. To swap models hot, restart the worker.

Run locally:

    uvicorn api.app:app --reload --port 8000

Then:

    curl -X POST http://localhost:8000/v1/score \
         -F image=@cat.jpg \
         -F "captions=a cat on grass" \
         -F "captions=a dog on grass" \
         -F model=CLIP
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure the repo root is importable when uvicorn is launched from elsewhere.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from models.model_registry import load_model_by_name
from utils.config import get_threshold
from utils.similarity import compute_similarity, detect_hallucination

# ---------------------------------------------------------------------------
# App + simple in-process model cache.
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VLM Hallucination Detector",
    description="Score image–caption alignment using CLIP, BLIP, or SigLIP.",
    version="0.1.0",
)

_MODEL_CACHE: dict = {}


def _get_cached_model(model_name: str):
    """Lazy-load and cache a backbone keyed by name."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = load_model_by_name(model_name)
    return _MODEL_CACHE[model_name]


# ---------------------------------------------------------------------------
# Response schemas.
# ---------------------------------------------------------------------------

class CaptionResult(BaseModel):
    caption: str
    score: float = Field(..., description="Cosine similarity in [-1, 1].")
    decision: str = Field(..., description='"likely correct" or "possible hallucination".')


class ScoreResponse(BaseModel):
    model: str
    threshold: float
    latency_ms: float
    results: List[CaptionResult]


class HealthResponse(BaseModel):
    status: str
    cached_models: List[str]
    device: str


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "vlm-hallucination-detector",
        "version": "0.1.0",
        "endpoints": ["GET /health", "POST /v1/score"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        cached_models=list(_MODEL_CACHE.keys()),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/v1/score", response_model=ScoreResponse, tags=["score"])
async def score(
    image: UploadFile = File(..., description="Image file (jpg / png)."),
    captions: List[str] = Form(..., description="One or more candidate captions."),
    model: str = Form("CLIP", description="Backbone: CLIP | BLIP | SigLIP."),
    threshold: Optional[float] = Form(
        None,
        description="Override the per-backbone threshold from configs/thresholds.yaml.",
    ),
):
    """Score one image against a list of captions and label each as correct or hallucinated."""

    # 1. Validate model name early so we don't load a 600 MB checkpoint to fail.
    if model not in {"CLIP", "BLIP", "SigLIP"}:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    if not captions:
        raise HTTPException(status_code=400, detail="At least one caption is required.")

    # 2. Decode the uploaded image.
    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # pragma: no cover — guarded by FastAPI's content checks too
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from e

    # 3. Load (or reuse) the chosen backbone.
    backbone, processor, device = _get_cached_model(model)

    # 4. Single forward pass for all captions.
    t0 = time.perf_counter()
    inputs = processor(text=captions, images=pil_img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = backbone(**inputs)

    similarity = compute_similarity(outputs.image_embeds, outputs.text_embeds)
    scores = similarity[0].detach().cpu().tolist()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # 5. Resolve threshold (explicit override > YAML config > fallback).
    tau = float(threshold) if threshold is not None else get_threshold(model)

    results = [
        CaptionResult(
            caption=c,
            score=float(s),
            decision=detect_hallucination(float(s), threshold=tau),
        )
        for c, s in zip(captions, scores)
    ]

    return ScoreResponse(model=model, threshold=tau, latency_ms=latency_ms, results=results)
