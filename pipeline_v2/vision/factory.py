"""
Model registry / factory.

Vision backends are imported lazily so the rest of the pipeline can
run without their dependencies installed.
"""
from __future__ import annotations
from typing import Any

from .base import VisionModel


def make_model(name: str, **opts: Any) -> VisionModel:
    """Construct and return a `VisionModel` by name.

    Supported names:
      * ``"stub"`` — fixed-string reply, for tests
      * ``"gemma4-e2b"`` / ``"gemma4"`` — Gemma 4 E2B via llama.cpp,
        subprocess-isolated, fits in 2 GB RAM
    """
    name = name.lower().strip()

    if name == "stub":
        from .backends.stub import StubVisionModel
        return StubVisionModel(**opts)

    if name in ("gemma4-e2b", "gemma4-e2b-subprocess", "gemma4"):
        from .backends.gemma4_subprocess import Gemma4SubprocessModel
        return Gemma4SubprocessModel(**opts)

    raise ValueError(f"Unknown vision model: {name!r}")
