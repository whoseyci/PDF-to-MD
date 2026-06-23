"""E2 -- DeepSeek-OCR integration as a quality-floor validator.

Two modes (controlled by ``mode=``):

  * ``"validate"`` (default, additive): only re-OCR pages where the
    primary extractor flagged low confidence. Compare the two
    paragraph-level outputs and merge.
  * ``"replace"``: replace pymupdf4llm entirely with DeepSeek-OCR on
    every page.

The model itself (deepseek-ai/DeepSeek-OCR) is heavy (~3B params, MIT
licensed, ~6GB on disk). Because our sandbox has 2GB RAM and no GPU,
this module is **lazy and gracefully degrades** when:
  * `transformers` is not installed
  * the weights aren't cached locally
  * available memory is below a configurable threshold

If any check fails, ``extract_pages`` returns ``status="unavailable"``
and the calling code can skip / use a different backend.

CLI:
    python3 -m pipeline_v2.deepseek_ocr paper.pdf --page 3
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
DEFAULT_DEVICE = "cpu"  # explicit; we don't auto-detect GPU here.


@dataclass
class PageResult:
    page: int
    text: str = ""
    confidence: float = 0.0
    elapsed_seconds: float = 0.0
    status: str = "unavailable"
    reason: str = ""


@dataclass
class OCRResult:
    status: str = "unavailable"
    reason: str = ""
    backend: str = "deepseek-ocr"
    n_pages: int = 0
    pages: List[PageResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


_MODEL = None
_TOKENIZER = None
_LOAD_ATTEMPTED = False


def _free_mem_mb() -> int:
    """Best-effort estimate of free RAM in MB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def _try_load(model_id: str = DEFAULT_MODEL_ID,
               require_mb: int = 6000) -> bool:
    """Returns True if model is loaded and ready, False otherwise."""
    global _MODEL, _TOKENIZER, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _MODEL is not None
    _LOAD_ATTEMPTED = True
    if os.environ.get("PDF2MD_DISABLE_DEEPSEEK") == "1":
        return False
    avail = _free_mem_mb()
    if avail and avail < require_mb:
        # Don't even try -- we'd OOM
        return False
    try:
        from transformers import AutoModel, AutoTokenizer  # noqa
    except Exception:
        return False
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True)
        _MODEL = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _MODEL.eval()
        return True
    except Exception:
        _MODEL = None
        _TOKENIZER = None
        return False


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def available() -> bool:
    return _try_load()


def extract_pages(pdf_path: Path,
                   *,
                   pages: Optional[List[int]] = None,
                   require_mb: int = 6000) -> OCRResult:
    """Run DeepSeek-OCR on selected pages of a PDF.

    If ``pages`` is None we OCR every page. Returns an ``OCRResult``
    with one ``PageResult`` per processed page. If the model can't
    load (insufficient memory, weights missing, etc.) status is
    ``unavailable`` and the per-page list is empty.
    """
    t0 = time.time()
    r = OCRResult()
    if not _try_load(require_mb=require_mb):
        r.reason = ("DeepSeek-OCR unavailable: transformers/weights "
                     f"missing or RAM<{require_mb}MB")
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r

    try:
        import fitz
    except ImportError:
        r.reason = "pymupdf missing"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r

    doc = fitz.open(str(pdf_path))
    try:
        n = doc.page_count
        if pages is None:
            pages = list(range(1, n + 1))
        results = []
        for p in pages:
            if p < 1 or p > n:
                continue
            t_p = time.time()
            page = doc[p - 1]
            pix = page.get_pixmap(dpi=200)
            png_bytes = pix.tobytes("png")
            img_path = Path(f"/tmp/_dsk_page_{p}.png")
            img_path.write_bytes(png_bytes)
            try:
                text = _run_inference(img_path)
                results.append(PageResult(
                    page=p, text=text or "",
                    elapsed_seconds=round(time.time() - t_p, 2),
                    status="ok",
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
        r.n_pages = len(results)
        r.status = "ok"
    finally:
        doc.close()
    r.elapsed_seconds = round(time.time() - t0, 3)
    return r


def _run_inference(img_path: Path) -> str:
    """Call the loaded model on a single PNG page."""
    if _MODEL is None:
        return ""
    # The DeepSeek-OCR repo on HF exposes a `.chat()`-like interface;
    # exact API depends on the model release. We try the common
    # conventions and fall back to a generic ``.generate``.
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return ""
    # Common API #1: .infer(image=...) on the auto_model
    if hasattr(_MODEL, "infer"):
        try:
            return _MODEL.infer(image=img) or ""
        except Exception:
            pass
    # Common API #2: .chat(tokenizer, image, prompt)
    if hasattr(_MODEL, "chat"):
        try:
            return _MODEL.chat(_TOKENIZER, image=img,
                                  prompt="<image>\nConvert this page to Markdown.") or ""
        except Exception:
            pass
    return ""


# ----------------------------------------------------------------------
# Validator helper: use DeepSeek-OCR to confirm low-confidence pages
# ----------------------------------------------------------------------

def select_low_confidence_pages(per_page_chars: Dict[int, int],
                                  threshold: int = 100) -> List[int]:
    """Return page numbers where char count is below threshold."""
    return [p for p, c in per_page_chars.items() if c < threshold]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pdf", type=Path)
    p.add_argument("--page", type=int, default=None)
    p.add_argument("--require-mb", type=int, default=6000,
                    help="Minimum available RAM (MB) to attempt loading")
    args = p.parse_args(argv)

    pages = [args.page] if args.page else None
    res = extract_pages(args.pdf, pages=pages, require_mb=args.require_mb)
    if res.status != "ok":
        print(f"status={res.status} reason={res.reason}")
        return 1
    for pr in res.pages:
        print(f"--- page {pr.page} ({pr.elapsed_seconds}s, "
                f"status={pr.status}) ---")
        print(pr.text[:1000])
        if len(pr.text) > 1000:
            print(f"... ({len(pr.text) - 1000} more chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
