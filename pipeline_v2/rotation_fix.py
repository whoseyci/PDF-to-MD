"""Rotation detection + counter-rotation for PDFs.

Many scanned or assembled PDFs have pages whose page rotation flag is
set (90°, 180°, 270°). pymupdf4llm respects the flag during its text
positioning, but downstream stages (especially our reading-order
reorder and chart_extract which renders pixmaps) get confused: text
ends up sideways in OCR, bboxes become wrong.

This module provides two services:

  1. `detect_rotation(page)` -- inspect a `fitz.Page` and decide whether
     the page text is correctly oriented. Returns 0 / 90 / 180 / 270.
     Two signals are used:
       a. The page-rotation flag itself (cheap, accurate when set)
       b. Tesseract's orientation+script detection on a rasterised
          page (slower, but catches mis-flagged pages too)

  2. `apply_rotation(page, target_rotation)` -- set the page's
     rotation flag so subsequent extraction sees text the right way up.
     We do NOT rewrite the PDF; we use ``page.set_rotation(0)`` after
     adjusting so PyMuPDF returns un-rotated text.

CLI:
    python3 -m pipeline_v2.rotation_fix paper.pdf
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class PageRotation:
    page: int                  # 1-indexed
    flag_rotation: int         # PDF metadata rotation (0/90/180/270)
    detected_rotation: int     # OSD-detected rotation (0/90/180/270)
    confidence: float          # 0..100 from tesseract
    corrected: bool            # whether we adjusted the flag


def _osd_via_tesseract(image_path: Path) -> Tuple[int, float]:
    """Run Tesseract OSD on an image, return (rotation_degrees, conf)."""
    if shutil.which("tesseract") is None:
        return 0, 0.0
    try:
        proc = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--psm", "0"],
            capture_output=True, text=True, timeout=15,
        )
        # tesseract OSD output looks like:
        #   Rotate: 0
        #   Orientation in degrees: 0
        #   Orientation confidence: 4.13
        rot = 0; conf = 0.0
        for line in proc.stdout.splitlines() + proc.stderr.splitlines():
            line = line.strip()
            if line.startswith("Rotate:"):
                rot = int(line.split(":")[1].strip())
            elif line.startswith("Orientation confidence:"):
                conf = float(line.split(":")[1].strip())
        return rot, conf
    except Exception:
        return 0, 0.0


def detect_rotation(doc, page_idx: int) -> PageRotation:
    """Inspect one page and decide its true rotation."""
    page = doc[page_idx]
    flag = page.rotation
    # Render at low DPI for speed
    with tempfile.NamedTemporaryFile(
            suffix=".png", dir="/home/user/.tmp", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        try:
            pix = page.get_pixmap(dpi=110)
            tmp.write_bytes(pix.tobytes("png"))
            osd_rot, conf = _osd_via_tesseract(tmp)
        except Exception:
            osd_rot, conf = 0, 0.0
    finally:
        try: tmp.unlink()
        except Exception: pass
    return PageRotation(
        page=page_idx + 1,
        flag_rotation=int(flag),
        detected_rotation=int(osd_rot),
        confidence=float(conf),
        corrected=False,
    )


def correct_document(doc, *, conf_threshold: float = 1.5
                      ) -> List[PageRotation]:
    """For each page: detect rotation, set the page rotation flag
    so that subsequent text extraction reads the page upright.

    Heuristic:
      * If page rotation flag is non-zero, trust it (set to 0 after
        rotating content rather than flag).
      * If page flag is 0 but OSD reports a non-zero angle with
        confidence ≥ conf_threshold, apply OSD's rotation.

    Returns a list of PageRotation, one per page.
    """
    out = []
    for i in range(doc.page_count):
        rep = detect_rotation(doc, i)
        page = doc[i]
        if rep.flag_rotation != 0:
            # Already flagged -- pymupdf already returns un-rotated text,
            # we don't need to do anything. Mark as not corrected.
            out.append(rep)
            continue
        if (rep.detected_rotation != 0
                and rep.confidence >= conf_threshold):
            # Apply correction: set page rotation flag, since extraction
            # respects it.
            try:
                page.set_rotation(int(-rep.detected_rotation) % 360)
                rep.corrected = True
            except Exception:
                pass
        out.append(rep)
    return out


def render_summary(reports: List[PageRotation]) -> str:
    lines = ["page | flag | detected | conf | corrected",
              "-----|------|----------|------|----------"]
    for r in reports:
        lines.append(
            f"{r.page:4d} | {r.flag_rotation:>4} | "
            f"{r.detected_rotation:>8} | {r.confidence:>4.1f} | "
            f"{'yes' if r.corrected else 'no'}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pdf", type=Path)
    p.add_argument("--apply", action="store_true",
                    help="Apply corrections and write a fixed PDF")
    p.add_argument("--out", type=Path, default=None,
                    help="Where to write the fixed PDF (defaults to <pdf>.fixed.pdf)")
    p.add_argument("--conf-threshold", type=float, default=1.5)
    args = p.parse_args(argv)
    try:
        import fitz
    except ImportError:
        print("pymupdf not installed")
        return 1
    doc = fitz.open(str(args.pdf))
    reports = correct_document(doc, conf_threshold=args.conf_threshold)
    print(render_summary(reports))
    if args.apply:
        out = args.out or args.pdf.with_suffix(".fixed.pdf")
        doc.save(str(out))
        print(f"wrote {out}")
    doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
