"""E5 -- Equation → LaTeX extraction with pix2tex.

Drops alongside chart_extract / mermaid_extract / diagram_extract as a
sibling extractor specialised for equation figures.

Architecture mirrors the chart_extract module:
  * pure-CPU; lazy model load on first call (~110MB weights)
  * graceful degradation: if pix2tex / torch isn't installed, returns
    UNAVAILABLE so the runner can fall back to alt-text
  * wraps output in $$ ... $$ for native GitHub markdown rendering

Usage:
    from pipeline_v2.vision.equation_extract import extract_equation
    result = extract_equation(image_path)
    # result.latex -> "x^2 + y^2 = r^2"
    # result.markdown -> "$$x^2 + y^2 = r^2$$"
    # result.status   -> "ok" | "unavailable" | "error"
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EquationResult:
    extractor: str = "pix2tex"
    status: str = "unavailable"   # ok | unavailable | error
    reason: str = ""
    latex: str = ""
    markdown: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# Cached singleton so we only pay the load cost once.
_MODEL = None
_LOAD_TRIED = False


def _try_load():
    global _MODEL, _LOAD_TRIED
    if _LOAD_TRIED:
        return _MODEL
    _LOAD_TRIED = True
    if os.environ.get("PDF2MD_DISABLE_PIX2TEX") == "1":
        return None
    try:
        # pip install pix2tex
        from pix2tex.cli import LatexOCR  # noqa
        _MODEL = LatexOCR()
        return _MODEL
    except Exception:
        return None


def extract_equation(image_path: Path,
                      *, caption: Optional[str] = None) -> EquationResult:
    t0 = time.time()
    r = EquationResult()
    try:
        from PIL import Image  # noqa
    except ImportError:
        r.reason = "Pillow missing"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r

    model = _try_load()
    if model is None:
        r.reason = ("pix2tex not installed; "
                     "`pip install pix2tex` to enable equation extraction")
        r.status = "unavailable"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r

    try:
        from PIL import Image
        img = Image.open(str(image_path))
        latex = model(img)
        if not latex or not latex.strip():
            r.status = "error"
            r.reason = "empty prediction"
        else:
            r.latex = latex.strip()
            r.markdown = f"$$\n{r.latex}\n$$"
            r.status = "ok"
            r.confidence = 0.7  # pix2tex doesn't expose calibrated conf
    except Exception as e:
        r.status = "error"
        r.reason = f"{type(e).__name__}: {e}"
    r.elapsed_seconds = round(time.time() - t0, 3)
    return r


def available() -> bool:
    return _try_load() is not None
