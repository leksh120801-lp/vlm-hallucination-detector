"""Thread-safe, cached model registry for CLIP / BLIP / SigLIP.

Three layers of caching, top-down — we hit the fastest layer first:

  1. **Process-wide singleton** (in-memory, lock-protected). The first call
     in a process loads the model; every subsequent call in the same process
     returns the same ``(model, processor, device)`` tuple instantly.
  2. **Optional on-disk joblib cache** (``.model_cache/<MODEL>.joblib``).
     Persists across process restarts. Off by default — pass
     ``use_disk_cache=True`` (or set the ``VLMHALL_USE_DISK_CACHE=1`` env var)
     to enable it. HuggingFace's own download cache at ``~/.cache/huggingface``
     is the primary disk cache for *weights*; this layer caches the *fully
     instantiated* model so frequent script restarts skip Python-side init
     time too.
  3. **Cold load** — ``transformers.from_pretrained`` from the HF cache (or
     download if missing).

HuggingFace's chatty initialisation logs are silenced once per process, so
``Some weights of CLIPModel were not initialized…`` no longer floods the
terminal on every reload.

Public API:
    - :func:`load_model_by_name` — primary entry point.
    - :func:`clear_model_cache`  — evict in-process and on-disk caches.
    - :func:`is_loaded`          — non-blocking introspection.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Callable

import joblib

from utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cache directory lives at the repo root so it's easy to find / clear.
DEFAULT_DISK_CACHE_DIR = (
    Path(__file__).resolve().parents[1] / ".model_cache"
)


def _disk_cache_enabled_by_default() -> bool:
    """Allow the user to opt into disk caching via env var (CI / Docker friendly)."""
    return os.environ.get("VLMHALL_USE_DISK_CACHE", "").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Process-wide singleton state (guarded by a single lock).
# ---------------------------------------------------------------------------

_PROCESS_CACHE: dict[str, tuple] = {}
_CACHE_LOCK = threading.Lock()
_LOGGING_QUIETED = False


def _quiet_hf_logging_once() -> None:
    """Silence HuggingFace's per-load INFO / WARNING spam; idempotent.

    Targets three sources of noise we keep seeing in the demo console:

      1. ``transformers``       — "Some weights of … were not initialized…",
                                   model-card download notes, etc.
      2. ``huggingface_hub``    — "you are not authenticated" notices,
                                   "resume_download is deprecated" hints, and
                                   the per-file progress bars.
      3. Generic ``warnings``   — UserWarnings emitted from inside ``transformers``
                                   (e.g. the BLIP slow-tokenizer notice).
    """
    global _LOGGING_QUIETED
    if _LOGGING_QUIETED:
        return

    import logging as _logging
    import warnings as _warnings

    # ``transformers`` — INFO + WARNING off; progress bars off.
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except Exception:
        pass

    # ``huggingface_hub`` — anonymous-request notice + download bars.
    try:
        from huggingface_hub.utils import logging as hub_logging
        hub_logging.set_verbosity_error()
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass

    # Hide the most common module-level deprecation noise without globally
    # masking everything (real exceptions still surface).
    for module in ("transformers", "huggingface_hub", "torch"):
        _warnings.filterwarnings("ignore", category=FutureWarning, module=module)
        _warnings.filterwarnings("ignore", category=UserWarning, module=module)

    # Library-internal loggers some checkpoints touch directly.
    for name in ("transformers", "transformers.modeling_utils",
                 "transformers.configuration_utils", "huggingface_hub"):
        _logging.getLogger(name).setLevel(_logging.ERROR)

    _LOGGING_QUIETED = True


# ---------------------------------------------------------------------------
# Backbone loaders — lazy-imported so importing the registry doesn't pull in
# torch / transformers until someone actually calls a loader.
# ---------------------------------------------------------------------------

def _lazy_clip():
    from models.clip_model import load_clip_model
    return load_clip_model()


def _lazy_blip():
    from models.blip_model import load_blip_model
    return load_blip_model()


def _lazy_siglip():
    from models.siglip_model import load_siglip_model
    return load_siglip_model()


_LOADERS: dict[str, Callable[[], tuple]] = {
    "CLIP":   _lazy_clip,
    "BLIP":   _lazy_blip,
    "SigLIP": _lazy_siglip,
}


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _disk_cache_path(model_name: str, cache_dir: Path) -> Path:
    return cache_dir / f"{model_name}.joblib"


def _load_from_disk(model_name: str, cache_dir: Path):
    path = _disk_cache_path(model_name, cache_dir)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        # A corrupt / mismatched cache shouldn't take down the app.
        logger.warning(
            "Disk cache for %s at %s is unreadable (%s); falling back to fresh load.",
            model_name, path, exc,
        )
        return None


def _save_to_disk(model_name: str, payload, cache_dir: Path) -> None:
    """Persist ``payload`` atomically — tmp file + rename so concurrent writers can't collide."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = _disk_cache_path(model_name, cache_dir)
    tmp_path = final_path.with_suffix(".tmp")
    try:
        joblib.dump(payload, tmp_path)
        tmp_path.replace(final_path)  # atomic on POSIX, near-atomic on Windows.
        logger.info("Cached %s to %s", model_name, final_path)
    except Exception as exc:
        logger.warning(
            "Failed to write disk cache for %s (%s); skipping (singleton cache still active).",
            model_name, exc,
        )
        # Best-effort cleanup of the partial tmp file.
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model_by_name(
    model_name: str,
    *,
    use_disk_cache: bool | None = None,
    cache_dir: Path = DEFAULT_DISK_CACHE_DIR,
) -> tuple:
    """Return ``(model, processor, device)`` for ``model_name``, cached.

    Args:
        model_name: ``"CLIP"``, ``"BLIP"`` or ``"SigLIP"``.
        use_disk_cache: opt into joblib-based disk caching across process restarts.
            Defaults to the value of the ``VLMHALL_USE_DISK_CACHE`` env var,
            which itself defaults to off.
        cache_dir: where to store joblib pickles when disk caching is on.
            Defaults to ``<repo>/.model_cache/``.

    Raises:
        ValueError: if ``model_name`` isn't one of the registered backbones.
    """
    if model_name not in _LOADERS:
        raise ValueError(
            f"Unknown model: {model_name!r}. Choose from {sorted(_LOADERS)}."
        )

    if use_disk_cache is None:
        use_disk_cache = _disk_cache_enabled_by_default()

    _quiet_hf_logging_once()

    # ---- Layer 1: in-process singleton (cheapest) ----
    with _CACHE_LOCK:
        cached = _PROCESS_CACHE.get(model_name)
    if cached is not None:
        return cached

    # ---- Layer 2: disk cache (optional) ----
    if use_disk_cache:
        on_disk = _load_from_disk(model_name, cache_dir)
        if on_disk is not None:
            with _CACHE_LOCK:
                # Race: another thread may have populated the cache while we
                # were reading from disk. The first writer wins.
                _PROCESS_CACHE.setdefault(model_name, on_disk)
                cached = _PROCESS_CACHE[model_name]
            logger.info("Loaded %s from disk cache.", model_name)
            return cached

    # ---- Layer 3: cold load via transformers.from_pretrained ----
    logger.info("Loading %s from HuggingFace…", model_name)
    payload = _LOADERS[model_name]()

    # Persist to disk before publishing in the in-process cache, so concurrent
    # readers either see "no cache yet" or "fully written cache".
    if use_disk_cache:
        _save_to_disk(model_name, payload, cache_dir)

    with _CACHE_LOCK:
        # If a concurrent caller raced past us, keep the existing entry.
        _PROCESS_CACHE.setdefault(model_name, payload)
        return _PROCESS_CACHE[model_name]


def clear_model_cache(
    model_name: str | None = None,
    *,
    clear_disk: bool = True,
    cache_dir: Path = DEFAULT_DISK_CACHE_DIR,
) -> None:
    """Evict cached entries (in-process and, by default, on-disk).

    Args:
        model_name: backbone to evict, or ``None`` to evict every backbone.
        clear_disk: also remove the on-disk joblib cache.
    """
    with _CACHE_LOCK:
        if model_name is None:
            _PROCESS_CACHE.clear()
        else:
            _PROCESS_CACHE.pop(model_name, None)

    if clear_disk and cache_dir.exists():
        if model_name is None:
            for f in cache_dir.glob("*.joblib"):
                with contextlib.suppress(FileNotFoundError):
                    f.unlink()
        else:
            with contextlib.suppress(FileNotFoundError):
                _disk_cache_path(model_name, cache_dir).unlink()


def is_loaded(model_name: str) -> bool:
    """Non-blocking check: is ``model_name`` already in the process cache?"""
    with _CACHE_LOCK:
        return model_name in _PROCESS_CACHE


def cached_models() -> list[str]:
    """Snapshot of currently-cached backbone names."""
    with _CACHE_LOCK:
        return list(_PROCESS_CACHE.keys())
