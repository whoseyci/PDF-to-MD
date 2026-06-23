"""E2 (refactored) -- Gemma-4 OCR validator for low-confidence pages.

Original E2 proposed using DeepSeek-OCR (a separate ~3B specialised
model) as a quality-floor validator. We rejected that because it
would have added a second VLM dependency on top of the Gemma 4 we
already ship.

This module instead **reuses the existing Gemma 4 E2B backend** (the
one already wired up for figure description / diagram → mermaid) to
re-OCR pages where the primary `pymupdf4llm` extractor produced
suspiciously little text.

Trade-off vs DeepSeek-OCR:
  * Gemma 4 isn't trained as an OCR-specialist, so character-level
    accuracy will be lower than a dedicated OCR model.
  * In exchange: zero new dependencies, zero new model weights, no
    extra environment variables, and the model is already loaded in
    other parts of the pipeline.
  * Speed: ~60-80s per page (vs DeepSeek's hypothetical 5-10 pages/min).
    Acceptable because we only fire on the small subset of pages
    that pymupdf4llm flagged as low-confidence.

Usage:
    from pipeline_v2.gemma_ocr import select_low_confidence_pages,
                                       reocr_pages
    bad = select_low_confidence_pages({1: 50, 2: 5000, 3: 80})
    res = reocr_pages(pdf_path, bad)

If the Gemma backend is unavailable (weights / llama.cpp not
installed) we gracefully degrade: status="unavailable", no crash.

CLI:
    python3 -m pipeline_v2.gemma_ocr paper.pdf --page 3
    python3 -m pipeline_v2.gemma_ocr paper.pdf --low-conf-from stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


OCR_PROMPT = (
    "<image>\n"
    "Transcribe ALL the text on this page exactly as it appears. "
    "Preserve paragraph breaks but DO NOT add commentary. "
    "Output the page text only."
)


@dataclass
class PageResult:
    page: int
    text: str = ""
    elapsed_seconds: float = 0.0
    status: str = "unavailable"   # ok | error | unavailable | timeout
    reason: str = ""


@dataclass
class OCRResult:
    status: str = "unavailable"   # ok | unavailable | partial
    reason: str = ""
    backend: str = "gemma4-e2b"
    n_pages_attempted: int = 0
    n_pages_ok: int = 0
    pages: List[PageResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ----------------------------------------------------------------------
# Backend wiring (shared with vision/runner.py's Gemma path)
# ----------------------------------------------------------------------

_BACKEND = None
_BACKEND_TRIED = False


def _try_backend(per_image_timeout: float = 180.0):
    global _BACKEND, _BACKEND_TRIED
    if _BACKEND_TRIED:
        return _BACKEND
    _BACKEND_TRIED = True
    if os.environ.get("PDF2MD_DISABLE_GEMMA_OCR") == "1":
        return None
    try:
        from pipeline_v2.vision.backends.gemma4_subprocess import (
            Gemma4SubprocessModel)
    except Exception:
        return None
    try:
        _BACKEND = Gemma4SubprocessModel(
            per_image_timeout=per_image_timeout,
            ctx_size=2048,            # need more ctx for full-page text
            image_max_tokens=120,
        )
        return _BACKEND
    except Exception:
        return None


def available() -> bool:
    """True iff the Gemma 4 backend can be initialised right now."""
    return _try_backend() is not None


# ----------------------------------------------------------------------
# Helper: which pages should we re-OCR?
# ----------------------------------------------------------------------

def select_low_confidence_pages(per_page_chars: Dict[int, int],
                                  *,
                                  threshold: int = 100) -> List[int]:
    """Return page numbers whose extracted char count is below threshold.

    The pymupdf4llm-side stats.json / provenance.json already records
    `chars` per page. Pages below ~100 chars on a journal article are
    almost always extraction failures (image-only pages, weird fonts,
    broken streams), so they're the right targets for VLM re-OCR.
    """
    return [p for p, c in per_page_chars.items() if c < threshold]


def page_chars_from_provenance(provenance_path: Path) -> Dict[int, int]:
    """Convenience: parse pages[] from provenance.json into a dict."""
    try:
        d = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for entry in d.get("pages", []):
        try:
            out[int(entry["page"])] = int(entry.get("chars", 0))
        except Exception:
            continue
    return out


# ----------------------------------------------------------------------
# Top-level: re-OCR a set of pages of a PDF
# ----------------------------------------------------------------------

def reocr_pages(pdf_path: Path,
                pages: List[int],
                *,
                per_image_timeout: float = 180.0,
                tmpdir: Optional[Path] = None) -> OCRResult:
    """Re-OCR the given page numbers (1-indexed) using the Gemma 4 backend."""
    t0 = time.time()
    r = OCRResult()
    backend = _try_backend(per_image_timeout=per_image_timeout)
    if backend is None:
        r.reason = ("Gemma 4 backend unavailable (llama.cpp / GGUF weights "
                     "missing or PDF2MD_DISABLE_GEMMA_OCR=1)")
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r
    try:
        import fitz
    except ImportError:
        r.reason = "pymupdf missing"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r

    tmpdir = tmpdir or Path(os.environ.get("TMPDIR") or "/home/user/.tmp")
    tmpdir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        n = doc.page_count
        results: List[PageResult] = []
        for p in pages:
            if p < 1 or p > n:
                continue
            t_p = time.time()
            page = doc[p - 1]
            pix = page.get_pixmap(dpi=150)
            img_path = tmpdir / f"_gemma_ocr_p{p}.png"
            img_path.write_bytes(pix.tobytes("png"))
            try:
                text = backend.describe(img_path, OCR_PROMPT,
                                          max_new_tokens=600)
                results.append(PageResult(
                    page=p, text=(text or "").strip(),
                    elapsed_seconds=round(time.time() - t_p, 2),
                    status="ok" if text and text.strip() else "error",
                    reason="" if (text and text.strip()) else "empty response",
                ))
            except Exception as e:
                results.append(PageResult(
                    page=p, status="error", reason=str(e),
                    elapsed_seconds=round(time.time() - t_p, 2),
                ))
            finally:
                try: img_path.unlink()
                except Exception: pass

        r.pages = results
        r.n_pages_attempted = len(results)
        r.n_pages_ok = sum(1 for x in results if x.status == "ok")
        r.status = "ok" if r.n_pages_ok else "partial"
    finally:
        doc.close()
    r.elapsed_seconds = round(time.time() - t0, 3)
    return r


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pdf", type=Path)
    p.add_argument("--page", type=int, action="append", default=[],
                    help="Specific page(s) to re-OCR (1-indexed); repeatable")
    p.add_argument("--low-conf-from", type=Path, default=None,
                    help="provenance.json to read per-page char counts from "
                          "and select low-confidence pages")
    p.add_argument("--threshold", type=int, default=100,
                    help="Char-count threshold for low-confidence selection")
    args = p.parse_args(argv)

    pages = list(args.page)
    if args.low_conf_from:
        per_page = page_chars_from_provenance(args.low_conf_from)
        pages.extend(select_low_confidence_pages(per_page,
                                                    threshold=args.threshold))
    pages = sorted(set(pages))
    if not pages:
        print("no pages selected for re-OCR (use --page or --low-conf-from)")
        return 1

    res = reocr_pages(args.pdf, pages)
    if res.status == "unavailable":
        print(f"status=unavailable reason={res.reason}")
        return 2
    for pr in res.pages:
        print(f"--- page {pr.page} ({pr.elapsed_seconds}s, "
                f"status={pr.status}) ---")
        if pr.reason:
            print(f"  reason: {pr.reason}")
        print(pr.text[:1500])
        if len(pr.text) > 1500:
            print(f"... ({len(pr.text) - 1500} more chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
