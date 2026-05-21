"""Centralised logging configuration.

Use ``utils.logging_config.get_logger(__name__)`` from anywhere in the project
instead of bare ``print()`` or per-module ``logging.basicConfig`` calls. The
first import wires up a single root handler with a sensible format; subsequent
imports are cheap.

Behaviour summary:
  - Default level: ``INFO`` (overridable via the ``VLMHALL_LOG_LEVEL`` env var).
  - Format: ISO timestamp · level · logger name · message.
  - Idempotent: safe to import from many modules without duplicating handlers.
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_LEVEL = os.environ.get("VLMHALL_LOG_LEVEL", "INFO").upper()
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s · %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """Wire up a single stderr handler on the root logger, idempotently."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)

    try:
        root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    except Exception:
        root.setLevel(logging.INFO)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name`` (typically ``__name__``)."""
    _configure_root_logger()
    return logging.getLogger(name)
