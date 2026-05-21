"""Dataset loaders — minimal, cached, schema-tolerant.

Two datasets are supported:

  - ``nlphuji/flickr30k``        — Flickr30k. PIL images are embedded directly
                                   in the rows.
  - ``yerevann/coco-karpathy``   — COCO Karpathy split. The dataset stores
                                   image URLs as strings; we cast that column
                                   to HuggingFace's :class:`datasets.Image`
                                   feature so the library itself lazy-fetches
                                   and caches each image. No manual ``requests``
                                   calls in our code.

All caching is handled by the standard HuggingFace cache. The cache directory
defaults to ``<repo>/.hf_cache`` and is overridable via the ``HF_HOME`` env var.

Failures raise ``RuntimeError`` with a clear, user-actionable message.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

# -----------------------------------------------------------------------------
# Cache configuration — set HF env vars at import time so subsequent
# ``datasets`` / ``transformers`` calls pick them up.
# -----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".hf_cache"

os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(_DEFAULT_CACHE_DIR / "datasets"))

CACHE_DIR = str(Path(os.environ["HF_HOME"]).resolve())


# -----------------------------------------------------------------------------
# Caption extraction — works across both datasets' shapes
# -----------------------------------------------------------------------------

def _get_first_caption(item: dict) -> str:
    """Return the first caption string from a HuggingFace dataset row.

    Schemas observed:

    - ``nlphuji/flickr30k``        → ``item["caption"]`` is ``List[str]``.
    - ``yerevann/coco-karpathy``   → ``item["sentences"]`` is ``List[str]``.
    """
    sentences = item.get("sentences")
    if sentences:
        first = sentences[0]
        if isinstance(first, dict):
            return str(first.get("raw") or first.get("caption") or "")
        if isinstance(first, str):
            return first

    caption = item.get("caption")
    if isinstance(caption, str):
        return caption
    if isinstance(caption, list) and caption:
        first = caption[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return str(first.get("raw") or first.get("caption") or "")

    return ""


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _safe_load_dataset(dataset_id: str, **kwargs):
    """``datasets.load_dataset`` wrapped with a friendly error message."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is not installed. "
            "Install with:  pip install 'datasets>=2.14,<3.0'"
        ) from exc

    try:
        return load_dataset(dataset_id, cache_dir=CACHE_DIR, **kwargs)
    except RuntimeError as exc:
        if "scripts are no longer supported" in str(exc).lower():
            raise RuntimeError(
                f"Cannot load {dataset_id!r}: the installed `datasets` version "
                "removed support for dataset scripts. Pin an older version:\n\n"
                "    pip install 'datasets>=2.14,<3.0'\n"
            ) from exc
        raise RuntimeError(
            f"Failed to load {dataset_id!r}: {exc}\n"
            f"Try clearing the cache at {CACHE_DIR} and re-running."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {dataset_id!r}: {exc}\n"
            "Common causes: missing network connection, missing optional dependency "
            "(e.g. `pip install pyarrow`), or HuggingFace authentication required."
        ) from exc


# -----------------------------------------------------------------------------
# Public loaders
# -----------------------------------------------------------------------------

def load_flickr_sample(n: int = 10) -> List[dict]:
    """Load up to ``n`` (image, caption) pairs from ``nlphuji/flickr30k``."""
    dataset = _safe_load_dataset(
        "nlphuji/flickr30k",
        split=f"test[:{n}]",
        trust_remote_code=True,
    )

    samples: List[dict] = []
    for item in dataset:
        if len(samples) >= n:
            break
        caption = _get_first_caption(item)
        if not caption:
            continue
        try:
            image = item["image"].convert("RGB")
        except Exception:
            continue
        samples.append({"image": image, "caption": caption})

    if not samples:
        raise RuntimeError(
            "No usable samples extracted from nlphuji/flickr30k. "
            f"Try clearing the cache at {CACHE_DIR}."
        )
    return samples


def load_coco_sample(n: int = 10) -> List[dict]:
    """Load up to ``n`` (image, caption) pairs from ``yerevann/coco-karpathy``.

    The dataset's ``url`` column is cast to HuggingFace's :class:`Image` feature
    so HF itself fetches each image lazily through its built-in cache —
    no manual HTTP from this codebase.
    """
    from datasets import Image as DatasetsImage

    dataset = _safe_load_dataset(
        "yerevann/coco-karpathy",
        split=f"validation[:{n}]",
    )
    dataset = dataset.cast_column("url", DatasetsImage())

    samples: List[dict] = []
    for item in dataset:
        if len(samples) >= n:
            break
        caption = _get_first_caption(item)
        if not caption:
            continue
        try:
            image = item["url"].convert("RGB")  # column is now decoded as PIL
        except Exception:
            continue
        samples.append({"image": image, "caption": caption})

    if not samples:
        raise RuntimeError(
            "No usable samples extracted from yerevann/coco-karpathy. "
            f"Try clearing the cache at {CACHE_DIR}."
        )
    return samples
