"""Tiny config loader for per-backbone decision thresholds.

The detector's hallucination threshold used to be hard-coded as 0.25 inside
``utils.similarity.detect_hallucination``. That value is fine for CLIP at base
size but is poorly calibrated for SigLIP, whose score distribution is shifted.
This module reads the YAML config so the thresholds live in one auditable
place and can be tuned without code changes.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Repo root = parent of `utils/`. Resolves correctly whether the package is
# imported from inside the repo or from a `pip install -e .` checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "configs" / "thresholds.yaml"

# Hard-coded fallbacks if the YAML is missing or malformed. Keep these in sync
# with the YAML's `defaults` block.
_FALLBACK_DEFAULTS = {"CLIP": 0.25, "BLIP": 0.25, "SigLIP": 0.10}


def _load_yaml() -> dict:
    if not _CONFIG_PATH.exists():
        return {"defaults": _FALLBACK_DEFAULTS, "overrides": {}}
    try:
        with _CONFIG_PATH.open("r") as f:
            data = yaml.safe_load(f) or {}
        # Ensure the expected top-level keys exist.
        data.setdefault("defaults", _FALLBACK_DEFAULTS)
        data.setdefault("overrides", {})
        return data
    except yaml.YAMLError:
        return {"defaults": _FALLBACK_DEFAULTS, "overrides": {}}


def get_threshold(model_name: str, dataset: str | None = None) -> float:
    """Return the decision threshold for a given backbone (and optional dataset).

    Lookup order:
        1. ``overrides[<dataset>][<model_name>]`` if both are present.
        2. ``defaults[<model_name>]``.
        3. ``_FALLBACK_DEFAULTS[<model_name>]``.
        4. ``0.25`` as a last resort, with the caller's name preserved.
    """
    cfg = _load_yaml()

    if dataset is not None:
        override = cfg.get("overrides", {}).get(dataset, {}).get(model_name)
        if override is not None:
            return float(override)

    default = cfg.get("defaults", {}).get(model_name)
    if default is not None:
        return float(default)

    return float(_FALLBACK_DEFAULTS.get(model_name, 0.25))
