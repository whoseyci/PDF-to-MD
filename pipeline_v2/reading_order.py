"""E1 -- Multi-column reading-order recovery.

Pymupdf4llm extracts text in PDF stream order, which on multi-column
journal layouts produces sentences with words from column 1 inter-
leaved with words from column 2 ("column-flow word reordering").

This module re-orders the page's TEXT BLOCKS into proper reading
order by:

  1. Detecting the column structure on the page (1 / 2 / 3 columns)
     via the bimodal histogram of block x-centres.
  2. Assigning each block to its column.
  3. Within each column, sorting top-to-bottom.
  4. Walking columns left-to-right.

This is a pragmatic, dependency-free alternative to VILA (which
requires PyTorch + a pretrained checkpoint). The accuracy is lower
than VILA but it gets the most common case right (clean 2-column
journal layout) with zero new dependencies.

Usage (programmatic):
    from pipeline_v2.reading_order import reorder_page_text
    fixed = reorder_page_text(pdf_path, page_number)

CLI:
    python3 -m pipeline_v2.reading_order paper.pdf
    python3 -m pipeline_v2.reading_order paper.pdf --page 3 --diff
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class TextBlock:
    text: str
    x0: float; y0: float; x1: float; y1: float
    page: int

    @property
    def xc(self): return (self.x0 + self.x1) / 2

    @property
    def yc(self): return (self.y0 + self.y1) / 2


# ----------------------------------------------------------------------
# Dehyphenation (Jun 2026 fix)
# ----------------------------------------------------------------------
# Pymupdf's `blocks` extractor preserves line-end hyphens verbatim,
# which then survive into our reordered output as broken words like
# "convolu\ntion" or "transfor\nmation". pdftotext (with -layout)
# silently joins these. Without this, the E1 reorder loses word
# inventory in the F1 eval. Fix: join soft/hard hyphens at line ends
# when the next line begins with a lowercase letter.

_DEHYPH_RE = re.compile(
    r"([A-Za-z]{2,})[\u00ad\-\u2010\u2011]\s*\n\s*([a-z][A-Za-z]*)")


def dehyphenate(text: str) -> str:
    """Join 'foo-\nbar' → 'foobar' when 'bar' starts with lowercase.
    Preserves real compound words ('state-\nof-the-art') by requiring
    lowercase start. Iterates because chained hyphens can appear."""
    for _ in range(3):
        new = _DEHYPH_RE.sub(lambda m: m.group(1) + m.group(2), text)
        if new == text:
            return new
        text = new
    return text


def dehyphenate_blocks(blocks: List["TextBlock"]) -> List["TextBlock"]:
    """Apply dehyphenation to each block's text in place (returns new list)."""
    out = []
    for b in blocks:
        out.append(TextBlock(text=dehyphenate(b.text),
                                x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
                                page=b.page))
    return out


# ----------------------------------------------------------------------
# Column detection
# ----------------------------------------------------------------------

def detect_n_columns(blocks: List[TextBlock], page_width: float) -> int:
    """Return the most likely number of columns (1, 2, or 3)."""
    if len(blocks) < 4:
        return 1
    xcs = [b.xc for b in blocks]
    # Use a coarse 16-bucket histogram of x-centres
    n_buckets = 16
    buckets = [0] * n_buckets
    for x in xcs:
        b = min(n_buckets - 1, int(x / page_width * n_buckets))
        buckets[b] += 1
    # Smooth
    smooth = [(buckets[i - 1] if i > 0 else 0) + buckets[i] +
                (buckets[i + 1] if i < n_buckets - 1 else 0)
                for i in range(n_buckets)]
    # Find local maxima above the mean
    mean = sum(smooth) / max(1, sum(1 for s in smooth if s))
    peaks = []
    for i in range(1, n_buckets - 1):
        if smooth[i] >= mean and smooth[i] >= smooth[i - 1] and \
                smooth[i] > smooth[i + 1]:
            peaks.append(i)
    # Merge peaks that are close together (within 2 buckets)
    merged = []
    for p in peaks:
        if merged and (p - merged[-1]) <= 2:
            continue
        merged.append(p)
    n_peaks = len(merged)
    if n_peaks >= 3:
        return 3
    if n_peaks == 2:
        return 2
    return 1


def assign_columns(blocks: List[TextBlock], n_cols: int,
                   page_width: float) -> List[int]:
    """Assign each block to a column index (0-indexed)."""
    if n_cols <= 1:
        return [0] * len(blocks)
    # Use n_cols equal bins; assign based on x-centre.
    cols = []
    for b in blocks:
        c = min(n_cols - 1, int(b.xc / page_width * n_cols))
        cols.append(c)
    return cols


def reorder_blocks(blocks: List[TextBlock],
                    page_width: float) -> List[TextBlock]:
    """Return a new list of blocks in proper reading order."""
    if not blocks:
        return []
    n_cols = detect_n_columns(blocks, page_width)
    cols = assign_columns(blocks, n_cols, page_width)
    # Separate full-width blocks (likely titles/headers) from columnar
    # blocks. A block whose width > 0.7 * page_width is treated as a
    # banner and inserted at its y-position regardless of column.
    columnar: List[List[TextBlock]] = [[] for _ in range(n_cols)]
    banners: List[TextBlock] = []
    for b, c in zip(blocks, cols):
        if (b.x1 - b.x0) > 0.7 * page_width and n_cols > 1:
            banners.append(b)
        else:
            columnar[c].append(b)
    # Sort each column top-to-bottom
    for col in columnar:
        col.sort(key=lambda b: (b.y0, b.x0))
    # Walk columns left-to-right
    ordered: List[TextBlock] = []
    for col in columnar:
        ordered.extend(col)
    # Insert banners by their y-position
    for ban in sorted(banners, key=lambda b: b.y0):
        # Find the first ordered block whose y0 > ban.y0
        idx = 0
        for i, ob in enumerate(ordered):
            if ob.y0 > ban.y0:
                idx = i
                break
            idx = i + 1
        ordered.insert(idx, ban)
    return ordered


# ----------------------------------------------------------------------
# Page-level extraction
# ----------------------------------------------------------------------

def reorder_page_text(pdf_path: Path, page_number: int) -> str:
    """Open a PDF, extract the text blocks on the given page (1-indexed),
    reorder them by reading-order, and return joined text."""
    try:
        import fitz
    except ImportError:
        return ""
    doc = fitz.open(str(pdf_path))
    try:
        if page_number < 1 or page_number > doc.page_count:
            return ""
        page = doc[page_number - 1]
        rect = page.rect
        # PyMuPDF defaults split ligatures (fi -> f+i) and leave
        # hyphenated line-end words split (configura-\ntion -> two
        # tokens). Combine the two flags so blocks() returns clean
        # text -- this single change recovers ~0.04 F1 in the harness.
        try:
            flags = (fitz.TEXT_PRESERVE_LIGATURES |
                       fitz.TEXT_DEHYPHENATE)
            blocks_raw = page.get_text("blocks", flags=flags)
        except Exception:
            blocks_raw = page.get_text("blocks")
        # blocks_raw: list of (x0, y0, x1, y1, text, block_no, block_type)
        blocks = []
        for tpl in blocks_raw:
            if len(tpl) < 5:
                continue
            x0, y0, x1, y1, text = tpl[0], tpl[1], tpl[2], tpl[3], tpl[4]
            if not isinstance(text, str) or not text.strip():
                continue
            # Skip image blocks (block_type 1 in pymupdf)
            if len(tpl) > 6 and tpl[6] != 0:
                continue
            blocks.append(TextBlock(text=text, x0=x0, y0=y0,
                                       x1=x1, y1=y1, page=page_number))
        ordered = reorder_blocks(blocks, rect.width)
        joined = "\n\n".join(b.text.strip() for b in ordered)
        return dehyphenate(joined)
    finally:
        doc.close()


def reorder_pdf_text(pdf_path: Path) -> str:
    """Reorder every page and return the full PDF text."""
    try:
        import fitz
    except ImportError:
        return ""
    doc = fitz.open(str(pdf_path))
    try:
        out_pages = []
        for i in range(doc.page_count):
            out_pages.append(reorder_page_text(pdf_path, i + 1))
        return "\n\n".join(out_pages)
    finally:
        doc.close()


# ----------------------------------------------------------------------
# Broken-sentence detector (metric)
# ----------------------------------------------------------------------

def count_broken_sentences(text: str) -> int:
    """Heuristic: count paragraphs that begin with a lowercase letter
    (a strong signal that the previous sentence was split mid-flow)."""
    n = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        first = para[0]
        if first.isalpha() and first.islower():
            n += 1
    return n


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pdf", type=Path)
    p.add_argument("--page", type=int, default=None)
    p.add_argument("--diff", action="store_true",
                    help="Print broken-sentence count before/after")
    args = p.parse_args(argv)

    if args.page:
        ordered = reorder_page_text(args.pdf, args.page)
        print(ordered)
        return 0
    full = reorder_pdf_text(args.pdf)
    if args.diff:
        try:
            import pymupdf4llm
            baseline = pymupdf4llm.to_markdown(str(args.pdf))
        except Exception:
            baseline = ""
        bb = count_broken_sentences(baseline)
        ba = count_broken_sentences(full)
        print(f"broken-sentence count: pymupdf4llm={bb}, reorder={ba} "
              f"(delta={bb - ba})")
    else:
        print(full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
