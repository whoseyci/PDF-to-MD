"""Smart text extraction dispatcher.

After running the eval harness we learned that on born-digital PDFs:
  * `pdftotext` gives F1 = 0.802 in 0.10 s
  * `pymupdf4llm` gives F1 = 0.803 in 15 s   (150× slower, ~no gain)

But on image-only PDFs:
  * `pdftotext` returns nearly empty
  * `pymupdf4llm` falls back to Tesseract OCR and returns useful text

So the right default is: **try pdftotext first, fall back to pymupdf4llm
if pdftotext output looks too thin per page.** This module implements
that dispatcher.

Public API:
    extract_text(pdf_path, *, mode="auto") -> ExtractionResult

Modes:
  * "auto"        -- try pdftotext, fall back to pymupdf4llm per-page
                       when chars-per-page < threshold
  * "pdftotext"   -- pdftotext only
  * "pymupdf4llm" -- pymupdf4llm only
  * "compare"     -- run both, return the longer one
  * "reorder"     -- E1 column-reorder + dehyphenate
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExtractionResult:
    text: str = ""
    backend_used: str = ""             # "pdftotext" | "pymupdf4llm" | mixed
    n_chars: int = 0
    elapsed_s: float = 0.0
    per_page_chars: List[int] = field(default_factory=list)
    fallback_pages: List[int] = field(default_factory=list)
    error: Optional[str] = None


# ----------------------------------------------------------------------
# Single-backend helpers
# ----------------------------------------------------------------------

def _pdftotext(pdf: Path, *, layout: bool = False,
                timeout: int = 120) -> str:
    """Run poppler pdftotext, return stdout."""
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd += [str(pdf), "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout)
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _pymupdf4llm(pdf: Path) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(pdf))


def _pdftotext_per_page(pdf: Path) -> List[str]:
    """Use `-f N -l N` to extract one page at a time; allows per-page
    fallback decisions in the auto-dispatcher."""
    try:
        import fitz
    except ImportError:
        return [_pdftotext(pdf)]
    try:
        doc = fitz.open(str(pdf))
    except Exception:
        return [_pdftotext(pdf)]
    n_pages = doc.page_count
    doc.close()
    pages: List[str] = []
    for i in range(1, n_pages + 1):
        try:
            proc = subprocess.run(
                ["pdftotext", "-f", str(i), "-l", str(i),
                 str(pdf), "-"],
                capture_output=True, text=True, timeout=20,
            )
            pages.append(proc.stdout if proc.returncode == 0 else "")
        except Exception:
            pages.append("")
    return pages


def _pymupdf4llm_per_page(pdf: Path, page_idx: int) -> str:
    """Single-page extraction. Returns markdown for that page only."""
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(str(pdf), pages=[page_idx])
    except Exception:
        return ""


# ----------------------------------------------------------------------
# Auto dispatcher
# ----------------------------------------------------------------------

def extract_auto(pdf: Path,
                  *, min_chars_per_page: int = 100,
                  rotation_fix: bool = True) -> ExtractionResult:
    """Try pdftotext per page; fall back to pymupdf4llm (which OCRs)
    on any page whose pdftotext output is below `min_chars_per_page`."""
    t0 = time.time()
    res = ExtractionResult(backend_used="auto")

    # Optional: detect + correct page rotations first
    if rotation_fix:
        try:
            import fitz
            from pipeline_v2.rotation_fix import correct_document
            doc = fitz.open(str(pdf))
            correct_document(doc, conf_threshold=2.0)
            # Save a temp copy with fixed rotations
            import tempfile
            with tempfile.NamedTemporaryFile(
                    suffix=".pdf", dir="/home/user/.tmp",
                    delete=False) as tf:
                fixed_pdf = Path(tf.name)
            doc.save(str(fixed_pdf))
            doc.close()
            work_pdf = fixed_pdf
        except Exception:
            work_pdf = pdf
            fixed_pdf = None
    else:
        work_pdf = pdf
        fixed_pdf = None

    try:
        per_page = _pdftotext_per_page(work_pdf)
        out_pages: List[str] = []
        fallback_pages: List[int] = []
        for i, txt in enumerate(per_page, start=1):
            if len(txt.strip()) >= min_chars_per_page:
                out_pages.append(txt)
            else:
                fb = _pymupdf4llm_per_page(work_pdf, i - 1)
                if len(fb.strip()) > len(txt.strip()):
                    out_pages.append(fb)
                    fallback_pages.append(i)
                else:
                    out_pages.append(txt)
        text = "\n\n".join(out_pages)
        res.text = text
        res.n_chars = len(text)
        res.per_page_chars = [len(p) for p in out_pages]
        res.fallback_pages = fallback_pages
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"

    if fixed_pdf is not None:
        try: fixed_pdf.unlink()
        except Exception: pass

    res.elapsed_s = round(time.time() - t0, 2)
    return res


# ----------------------------------------------------------------------
# Top-level entrypoint
# ----------------------------------------------------------------------

def extract_text(pdf: Path, *, mode: str = "auto",
                  rotation_fix: bool = True) -> ExtractionResult:
    """Smart text extraction.

    mode:
      auto         -- pdftotext per page; pymupdf4llm fallback when
                       a page has < 100 chars (default)
      pdftotext    -- pdftotext layout-aware only
      pymupdf4llm  -- pymupdf4llm only (slow but OCR-capable)
      compare      -- run both, return whichever produced more chars
      reorder      -- E1 column-reorder + dehyphenate (text-only)
    """
    t0 = time.time()
    if mode == "auto":
        return extract_auto(pdf, rotation_fix=rotation_fix)
    if mode == "pdftotext":
        text = _pdftotext(pdf, layout=False)
        return ExtractionResult(text=text, backend_used="pdftotext",
                                  n_chars=len(text),
                                  elapsed_s=round(time.time() - t0, 2))
    if mode == "pymupdf4llm":
        text = _pymupdf4llm(pdf)
        return ExtractionResult(text=text, backend_used="pymupdf4llm",
                                  n_chars=len(text),
                                  elapsed_s=round(time.time() - t0, 2))
    if mode == "compare":
        pt = _pdftotext(pdf)
        pm = _pymupdf4llm(pdf)
        chosen, name = (pt, "pdftotext") if len(pt) >= len(pm) else (pm, "pymupdf4llm")
        return ExtractionResult(text=chosen, backend_used=f"compare→{name}",
                                  n_chars=len(chosen),
                                  elapsed_s=round(time.time() - t0, 2))
    if mode == "reorder":
        from pipeline_v2.reading_order import reorder_pdf_text
        text = reorder_pdf_text(pdf)
        return ExtractionResult(text=text, backend_used="reorder",
                                  n_chars=len(text),
                                  elapsed_s=round(time.time() - t0, 2))
    raise ValueError(f"unknown mode: {mode}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _cli(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Smart PDF→text dispatcher. pdftotext first, "
                     "pymupdf4llm fallback when a page has < 100 chars."
    )
    p.add_argument("pdf", type=Path)
    p.add_argument("--mode",
                    choices=["auto", "pdftotext", "pymupdf4llm",
                              "compare", "reorder"],
                    default="auto")
    p.add_argument("--no-rotation-fix", action="store_true",
                    help="Skip rotation detection step (faster)")
    p.add_argument("--out", type=Path, default=None,
                    help="Write text to this file instead of stdout")
    p.add_argument("--stats", action="store_true",
                    help="Print backend choice + per-page char counts")
    args = p.parse_args(argv)
    res = extract_text(args.pdf, mode=args.mode,
                        rotation_fix=not args.no_rotation_fix)
    if args.out:
        args.out.write_text(res.text or "", encoding="utf-8")
        print(f"wrote {args.out}  ({res.n_chars:,} chars, "
                f"backend={res.backend_used}, "
                f"{res.elapsed_s}s)")
    else:
        print(res.text)
    if args.stats:
        print(f"\n# backend: {res.backend_used}", file=__import__('sys').stderr)
        print(f"# elapsed: {res.elapsed_s}s", file=__import__('sys').stderr)
        if res.fallback_pages:
            print(f"# pymupdf4llm fallback used on pages: "
                    f"{res.fallback_pages}", file=__import__('sys').stderr)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
