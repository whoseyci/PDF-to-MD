"""
Vision-model harness for figure → text/table/diagram conversion.

Public API
----------

* `VisionModel` — abstract base class. Implement one method
  (`describe(image, prompt) -> str`) for a new backend.
* `make_model(name, **opts)` — factory. Supports:
    - ``"stub"`` — fixed-string reply, for tests
    - ``"gemma4-e2b"`` / ``"gemma4"`` — Gemma 4 E2B via llama.cpp
      subprocess (fits in 2 GB RAM, see `backends/gemma4_subprocess.py`)
* `classify_figure(caption, ocr_text) -> FigureKind` — caption-based
  classifier that decides what kind of figure we have so we can route
  it to the right extractor.
* `process_figure(figure_meta, paper_dir, model)` — top-level entry
  point. For each figure: classify → try classical `chart_extract` →
  if chart-shaped, validate the extracted table via the VLM; if
  diagram-shaped, extract a Mermaid graph via `MermaidExtractor`;
  otherwise call the VLM for an alt-text sentence.
"""

from .base import VisionModel, FigureKind, FigureVisionResult
from .classifier import classify_figure
from .factory import make_model
from .runner import process_figure

__all__ = [
    "VisionModel",
    "FigureKind",
    "FigureVisionResult",
    "classify_figure",
    "make_model",
    "process_figure",
]
