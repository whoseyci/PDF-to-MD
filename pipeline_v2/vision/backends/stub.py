"""Stub vision model for tests and offline runs."""
from __future__ import annotations
from pathlib import Path
from ..base import VisionModel


class StubVisionModel(VisionModel):
    name = "stub"
    def __init__(self, reply: str = "OK", **_):
        self._reply = reply
    def describe(self, image_path: Path, prompt: str, **_) -> str:
        return self._reply
