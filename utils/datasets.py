"""Dataset loaders for Flickr30k, COCO Karpathy, and Visual Genome.

These wrap HuggingFace ``datasets``. Several of the underlying datasets
(``nlphuji/flickr30k``, ``visual_genome``) are script-based, which the
``datasets`` library deprecated in v3.0. To keep the demo working in the
field we:

  1. Try the HuggingFace loader first (works on ``datasets < 3.0``).
  2. On any failure, fall back to a small curated list of stable
     (image_url, caption) pairs, so the Streamlit dashboard never crashes.

Pin ``datasets < 3.0`` in ``requirements.txt`` for the official path.
"""

from __future__ import annotations

import warnings
from io import BytesIO
from typing import List

import requests
from PIL import Image

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover — datasets is a hard dep, but be defensive.
    load_dataset = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Curated fallback samples — stable public-domain images on Wikimedia Commons.
# Used when the HuggingFace path fails (e.g. on datasets >= 3.0, no network).
# ---------------------------------------------------------------------------

_FALLBACK_SAMPLES: List[dict] = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/640px-Cat_November_2010-1a.jpg",
        "caption": "a tabby cat sitting and looking forward",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/640px-Collage_of_Nine_Dogs.jpg",
        "caption": "a collage of nine different dog breeds",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg",
        "caption": "a black and white cat lying on a wooden floor",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png/640px-Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png",
        "caption": "a wildcat sitting on a rock",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Pioneer_DJ_setup.jpg/640px-Pioneer_DJ_setup.jpg",
        "caption": "a DJ setup with mixing equipment",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/640px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg",
        "caption": "the Eiffel Tower against a clear sky",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg/640px-Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg",
        "caption": "the north face of Mount Everest from base camp",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/640px-Red_Apple.jpg",
        "caption": "a single red apple on a white background",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Juvenile_Ragdoll.jpg/640px-Juvenile_Ragdoll.jpg",
        "caption": "a juvenile ragdoll kitten with blue eyes",
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/American_Eskimo_Dog_1.jpg/640px-American_Eskimo_Dog_1.jpg",
        "caption": "an American Eskimo dog with a fluffy white coat",
    },
]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _fetch_image(url: str, timeout: float = 10.0) -> Image.Image:
    """Download an image and decode it as RGB."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _materialise_fallback(n: int, source: str) -> List[dict]:
    """Materialise the curated fallback list, downloading up to ``n`` images."""
    warnings.warn(
        f"{source}: HuggingFace loader unavailable, falling back to a curated "
        "Wikimedia Commons sample. Pin `datasets<3.0` for the full HF path.",
        RuntimeWarning,
        stacklevel=2,
    )
    out: List[dict] = []
    for sample in _FALLBACK_SAMPLES:
        if len(out) >= n:
            break
        try:
            out.append({"image": _fetch_image(sample["url"]), "caption": sample["caption"]})
        except Exception:
            continue
    return out


def _try_hf_dataset(loader_callable, n: int, url_extractor, caption_extractor) -> List[dict]:
    """Run a HuggingFace loader, materialise up to ``n`` (image, caption) pairs.

    ``url_extractor`` and ``caption_extractor`` are callables ``item -> str``,
    which lets us accept both flat (``item["url"]``) and nested
    (``item["image"]["url"]``) dataset shapes.
    """
    dataset = loader_callable()
    samples: List[dict] = []
    for item in dataset:
        if len(samples) >= n:
            break
        try:
            url = url_extractor(item)
            caption = caption_extractor(item)
            samples.append({"image": _fetch_image(url), "caption": caption})
        except Exception:
            continue
    return samples


# ---------------------------------------------------------------------------
# Public loaders.
# ---------------------------------------------------------------------------

def load_flickr_sample(n: int = 10) -> List[dict]:
    """Load up to ``n`` Flickr30k (image, caption) pairs.

    Falls back to a curated list of public-domain images if the HuggingFace
    loader is not available on the installed ``datasets`` version.
    """
    if load_dataset is None:
        return _materialise_fallback(n, "Flickr30k")

    try:
        return _try_hf_dataset(
            loader_callable=lambda: load_dataset(
                "nlphuji/flickr30k", split="test", trust_remote_code=True
            ),
            n=n,
            url_extractor=lambda item: item["url"],
            caption_extractor=lambda item: item["sentences"][0],
        )
    except Exception:
        return _materialise_fallback(n, "Flickr30k")


def load_coco_sample(n: int = 10) -> List[dict]:
    """Load up to ``n`` COCO Karpathy (image, caption) pairs."""
    if load_dataset is None:
        return _materialise_fallback(n, "COCO Karpathy")

    try:
        return _try_hf_dataset(
            loader_callable=lambda: load_dataset(
                "yerevann/coco-karpathy", split="validation[:50]"
            ),
            n=n,
            url_extractor=lambda item: item["url"],
            caption_extractor=lambda item: item["sentences"][0],
        )
    except Exception:
        return _materialise_fallback(n, "COCO Karpathy")


def load_visual_genome_sample(n: int = 10) -> List[dict]:
    """Load up to ``n`` Visual Genome region-description samples."""
    if load_dataset is None:
        return _materialise_fallback(n, "Visual Genome")

    try:
        return _try_hf_dataset(
            loader_callable=lambda: load_dataset(
                "visual_genome",
                "region_descriptions_v1.2.0",
                split="train[:50]",
                trust_remote_code=True,
            ),
            n=n,
            url_extractor=lambda item: item["image"]["url"],
            caption_extractor=lambda item: item["regions"][0]["phrase"],
        )
    except Exception:
        return _materialise_fallback(n, "Visual Genome")
