"""FastAPI service for the VLM Hallucination Detector.

Endpoints
---------
    GET  /                — service metadata.
    GET  /health          — liveness probe (cached models, device).
    POST /v1/score        — score one image and one or more captions, returning
                            decisions from each requested detection method.

The chosen backbone (CLIP / BLIP / SigLIP) is loaded lazily and cached in
process memory; subsequent requests do not pay the model-load cost.

Run locally::

    uvicorn api.app:app --reload --port 8000

Example request::

    curl -X POST http://localhost:8000/v1/score \\
         -F image=@cat.jpg \\
         -F "captions=a cat on grass" \\
         -F "captions=a dog on grass" \\
         -F model=CLIP \\
         -F method=ALL
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# Ensure repo root is importable when uvicorn launches from elsewhere.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from models.model_registry import load_model_by_name  # noqa: E402
from utils.caption_attack import generate_adversarial_captions  # noqa: E402
from utils.classifier import load_logistic_model  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402
from utils.methods import (  # noqa: E402
    method_consistency_threshold,
    method_logistic,
    method_threshold,
)
from utils.similarity import compute_similarity  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App + caches
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VLM Hallucination Detector",
    description="Score image–caption alignment using CLIP, BLIP, or SigLIP, "
                "with three pluggable detection methods.",
    version="0.2.0",
)

_BACKBONE_CACHE: dict = {}
_LOGISTIC_MODEL_CACHE: dict = {"loaded": False, "model": None}

_VALID_BACKBONES = {"CLIP", "BLIP", "SigLIP"}
_VALID_METHODS = {"Threshold", "Consistency", "Logistic", "ALL"}


def _get_cached_backbone(model_name: str):
    """Lazy-load and cache a backbone keyed by name."""
    if model_name not in _BACKBONE_CACHE:
        logger.info("Loading backbone %s into cache", model_name)
        _BACKBONE_CACHE[model_name] = load_model_by_name(model_name)
    return _BACKBONE_CACHE[model_name]


def _get_logistic_model_lazy():
    """Try to load the trained logistic head exactly once per process."""
    if not _LOGISTIC_MODEL_CACHE["loaded"]:
        try:
            _LOGISTIC_MODEL_CACHE["model"] = load_logistic_model()
        except Exception as exc:
            logger.info("Logistic model not available: %s", exc)
            _LOGISTIC_MODEL_CACHE["model"] = None
        _LOGISTIC_MODEL_CACHE["loaded"] = True
    return _LOGISTIC_MODEL_CACHE["model"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CaptionMethodResult(BaseModel):
    caption: str
    method: str = Field(..., description='"Threshold", "Consistency", or "Logistic".')
    score: float = Field(..., description="Cosine similarity in [-1, 1].")
    consistency: float | None = Field(
        None, description="Original-vs-adversarial gap; only set for Consistency / Logistic."
    )
    decision: str = Field(..., description='"likely correct" or "hallucination".')


class ScoreResponse(BaseModel):
    model: str
    methods: list[str]
    similarity_threshold: float
    consistency_threshold: float
    latency_ms: float
    results: list[CaptionMethodResult]


class HealthResponse(BaseModel):
    status: str
    cached_models: list[str]
    logistic_model_loaded: bool
    device: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "vlm-hallucination-detector",
        "version": app.version,
        "endpoints": ["GET /health", "POST /v1/score"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        cached_models=list(_BACKBONE_CACHE.keys()),
        logistic_model_loaded=_get_logistic_model_lazy() is not None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/v1/score", response_model=ScoreResponse, tags=["score"])
async def score(
    image: UploadFile = File(..., description="Image file (jpg / png)."),
    captions: list[str] = Form(..., description="One or more candidate captions."),
    model: str = Form("CLIP", description="Backbone: CLIP | BLIP | SigLIP."),
    method: str = Form(
        "Threshold",
        description='Detection method: "Threshold", "Consistency", "Logistic", or "ALL".',
    ),
    similarity_threshold: float = Form(0.25),
    consistency_threshold: float = Form(0.20),
):
    """Score one image against captions using one or more detection methods."""
    # ---- Validate inputs ----
    if model not in _VALID_BACKBONES:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    if method not in _VALID_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
    if not captions:
        raise HTTPException(status_code=400, detail="At least one caption is required.")

    # ---- Decode image ----
    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    # ---- Resolve method set ----
    methods_active = (
        {"Threshold", "Consistency", "Logistic"} if method == "ALL" else {method}
    )
    logistic_model = None
    if "Logistic" in methods_active:
        logistic_model = _get_logistic_model_lazy()
        if logistic_model is None:
            methods_active.discard("Logistic")

    # ---- Forward pass through the chosen backbone ----
    backbone, processor, device = _get_cached_backbone(model)
    t0 = time.perf_counter()

    results: list[CaptionMethodResult] = []
    for primary_caption in captions:
        # Each caption is scored against itself + adversaries; this gives
        # Consistency and Logistic the comparison scores they need.
        attack_captions = generate_adversarial_captions(primary_caption)
        all_caps = [primary_caption] + attack_captions

        inputs = processor(
            text=all_caps, images=pil_img, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            outputs = backbone(**inputs)

        sims = compute_similarity(outputs.image_embeds, outputs.text_embeds)
        scores = sims[0].detach().cpu().tolist()
        original_score = float(scores[0])
        attack_scores = [float(s) for s in scores[1:]]

        if "Threshold" in methods_active:
            decision = method_threshold(original_score, threshold=similarity_threshold)
            results.append(CaptionMethodResult(
                caption=primary_caption, method="Threshold",
                score=original_score, consistency=None, decision=decision,
            ))

        if "Consistency" in methods_active:
            decision, consistency = method_consistency_threshold(
                original_score, attack_scores, threshold=consistency_threshold,
            )
            results.append(CaptionMethodResult(
                caption=primary_caption, method="Consistency",
                score=original_score, consistency=consistency, decision=decision,
            ))

        if "Logistic" in methods_active:
            decision, consistency = method_logistic(
                logistic_model, original_score, attack_scores,
            )
            results.append(CaptionMethodResult(
                caption=primary_caption, method="Logistic",
                score=original_score, consistency=consistency, decision=decision,
            ))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return ScoreResponse(
        model=model,
        methods=sorted(methods_active),
        similarity_threshold=similarity_threshold,
        consistency_threshold=consistency_threshold,
        latency_ms=latency_ms,
        results=results,
    )
