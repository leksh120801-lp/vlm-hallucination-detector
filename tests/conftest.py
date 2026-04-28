"""Shared pytest fixtures.

Tests in this suite avoid downloading model weights from HuggingFace; instead
we synthesise embedding tensors directly. This keeps the CI run fast (< 10 s)
and offline-safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put the repo root on sys.path so `from utils...` and `from models...` work
# when pytest is invoked from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
