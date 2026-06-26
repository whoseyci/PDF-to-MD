"""Recover superscripts, subscripts, math glyphs, and degree-units from PDFs.

What this module does (and why)
-------------------------------
Comparison with Baidu's Unlimited-OCR on an MDPI olive-grove paper exposed
one clear gap: vision-based OCR captures math notation like ``7 × 7``,
``m²``, ``35°30'59.5''`` and renders it as LaTeX (``\\( 7 \\times 7 \\)``,
``\\( ^{2} \\)``) -- whereas our PyMuPDF / pdftotext outputs collapse the
super/subscript baseline shifts to inline characters (``m2`` instead of
``m²``, ``35◦30′59.5′′`` with junk glyphs from the CMSY10 math font).

The good news is the signal is *already there* in PyMuPDF spans. For
each glyph, we have:
- ``size`` -- font size in points
- ``bbox`` -- baseline y delta against the surrounding line
- ``font`` -- CMSY10 / CMR10 indicate math-mode origin

So we can detect super/subscripts deterministically without OCR or an
LLM by walking spans and flagging any span whose size is < ~85% of the
local line's body size AND whose vertical centre is offset above/below
the line's body baseline.

We also normalise the LaTeX math-font glyphs that came across as
strange Unicode (e.g. CMSY10's ``◦`` -> ``°``, ``′′`` -> ``″``) so the
output renders correctly in Markdown.

Public API
----------
``recover_superscripts(pdf_path) -> Dict[int, str]``
  Returns ``{page_idx: page_markdown}`` for every page, with super/sub
  markers and math glyphs converted.

``annotate_page(page, *, html_supsub=True) -> str``
  Process a single PyMuPDF page object.

``test_with_unlimited_ocr_paper()``
  Sanity-check on the MDPI olive-grove PDF; prints before/after on a
  handful of lines that contain ``m²``, ``°``, etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# CMSY10 / CMR10 / other math-font glyph fix-ups
# ----------------------------------------------------------------------
# These are characters that PyMuPDF surfaces from LaTeX math fonts but
# that don't render as math symbols in Markdown. Map them to the
# Unicode equivalents that DO render.
_MATH_GLYPH_FIXUPS = {
    # CMSY10 -- LaTeX symbol font
    "\u25e6": "\u00b0",   # ◦ -> °   (LaTeX \circ used as degree sign)
    "\u2032": "\u2032",   # ′  prime (already correct)
    "\u2033": "\u2033",   # ″  double prime
    # PyMuPDF often returns CMSY10's double-prime as two single-primes.
    # Replace any "′′" pair with U+2033 (double prime).
    # (handled in code as a regex, not a single-char map)
}

# Greek letters from CMMI10
_GREEK_ALIASES = {
    "α": "α", "β": "β", "γ": "γ", "δ": "δ",
    "ε": "ε", "λ": "λ", "μ": "μ", "π": "π",
    "ρ": "ρ", "σ": "σ", "φ": "φ", "ω": "ω",
}

# Fonts that indicate math-mode glyphs (so single chars like '=' or
# subscript indices come from these). Used to decide whether to wrap a
# run in $...$.
_MATH_FONTS = ("CMSY", "CMR", "CMMI", "CMEX", "CMTI")


# ----------------------------------------------------------------------
# Span data
# ----------------------------------------------------------------------

@dataclass
class _Span:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str
    flags: int

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def baseline(self) -> float:
        return self.y1  # PyMuPDF bbox bottom approximates the baseline


@dataclass
class _Line:
    spans: List[_Span]

    def median_size(self) -> float:
        sizes = sorted(s.size for s in self.spans if s.text.strip())
        if not sizes:
            return 10.0
        return sizes[len(sizes) // 2]

    def body_baseline(self) -> float:
        """Median baseline of the body-sized spans (ignore tiny/large)."""
        med = self.median_size()
        body = [s.baseline for s in self.spans
                if s.text.strip() and abs(s.size - med) < 0.5]
        if not body:
            body = [s.baseline for s in self.spans if s.text.strip()]
        if not body:
            return 0.0
        body.sort()
        return body[len(body) // 2]


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------

# A span is a superscript when:
#   * its font size is <= SIZE_RATIO * line's median body size
#   * its vertical centre is at least MIN_OFFSET above the body baseline
# A span is a subscript when offset is below the body baseline by the
# same threshold.

SIZE_RATIO = 0.85
MIN_OFFSET_RATIO = 0.15   # at least 15 % of body size above/below baseline


def _classify(span: _Span, line: _Line) -> str:
    """Return 'super' / 'sub' / 'body'."""
    if not span.text.strip():
        return "body"
    body_size = line.median_size()
    if span.size > body_size * SIZE_RATIO:
        return "body"
    # baseline of body text
    body_y = line.body_baseline()
    if body_y == 0.0:
        return "body"
    offset = body_y - span.baseline  # positive == raised above baseline
    threshold = body_size * MIN_OFFSET_RATIO
    if offset >= threshold:
        return "super"
    if offset <= -threshold:
        return "sub"
    return "body"


def _is_math_font(font: str) -> bool:
    return any(prefix in font for prefix in _MATH_FONTS)


# ----------------------------------------------------------------------
# Glyph normalisation
# ----------------------------------------------------------------------

_DOUBLE_PRIME_PAIR = re.compile(r"\u2032\u2032")


def _normalise_glyphs(text: str) -> str:
    """Apply char-level fix-ups for math-font glyph quirks."""
    for old, new in _MATH_GLYPH_FIXUPS.items():
        if old != new:
            text = text.replace(old, new)
    # Two single-primes -> one double-prime (CMSY10 artefact).
    text = _DOUBLE_PRIME_PAIR.sub("\u2033", text)
    return text


# ----------------------------------------------------------------------
# Span-to-Markdown rendering
# ----------------------------------------------------------------------

def _render_line(line: _Line, *, html_supsub: bool = True) -> str:
    """Render a single line with super/sub markers."""
    out: List[str] = []
    prev_role = "body"
    buf: List[str] = []

    def flush(role: str) -> None:
        nonlocal buf
        if not buf:
            return
        chunk = _normalise_glyphs("".join(buf))
        if role == "super":
            if html_supsub:
                out.append(f"<sup>{chunk}</sup>")
            else:
                out.append(f"^{{{chunk}}}")
        elif role == "sub":
            if html_supsub:
                out.append(f"<sub>{chunk}</sub>")
            else:
                out.append(f"_{{{chunk}}}")
        else:
            out.append(chunk)
        buf = []

    for span in line.spans:
        role = _classify(span, line)
        # Merge runs of same role into one tag.
        if role != prev_role:
            flush(prev_role)
            prev_role = role
        buf.append(span.text)
    flush(prev_role)
    return "".join(out)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def annotate_page(page, *, html_supsub: bool = True) -> str:
    """Process a single PyMuPDF page and return Markdown-with-supsub text.

    `html_supsub`:
        True  -- use ``<sup>``/``<sub>`` (renders on GitHub Markdown)
        False -- use LaTeX-style ``^{}`` / ``_{}`` (good for math passes)
    """
    try:
        d = page.get_text("dict")
    except Exception:
        return ""
    out_lines: List[str] = []
    for block in d.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans: List[_Span] = []
            for s in line["spans"]:
                bbox = s.get("bbox", (0.0, 0.0, 0.0, 0.0))
                spans.append(_Span(
                    text=s.get("text", ""),
                    x0=float(bbox[0]), y0=float(bbox[1]),
                    x1=float(bbox[2]), y1=float(bbox[3]),
                    size=float(s.get("size", 10.0)),
                    font=str(s.get("font", "")),
                    flags=int(s.get("flags", 0)),
                ))
            if not spans:
                continue
            out_lines.append(_render_line(_Line(spans),
                                          html_supsub=html_supsub))
        out_lines.append("")  # paragraph break between blocks
    return "\n".join(out_lines)


def recover_superscripts(pdf_path, *,
                          html_supsub: bool = True) -> Dict[int, str]:
    """Return ``{page_idx: page_markdown}`` for every page."""
    import fitz
    doc = fitz.open(pdf_path)
    try:
        out: Dict[int, str] = {}
        for i, page in enumerate(doc):
            out[i] = annotate_page(page, html_supsub=html_supsub)
        return out
    finally:
        doc.close()


# ----------------------------------------------------------------------
# Integration helper: post-process pdftotext / pymupdf4llm output
# ----------------------------------------------------------------------
# The text_extract pipeline already has the markdown body. We don't want
# to RE-extract -- we want to AUGMENT the existing markdown with the
# super/sub markers we discover. Strategy:
#  1. Build a per-page list of (text, role) tuples from PyMuPDF spans.
#  2. For each existing line in the markdown, find the matching PyMuPDF
#     text run and splice in the role markers.
# That's brittle on heavily-postprocessed markdown though. The simpler
# integration: callers who want supsub-aware output should call
# `recover_superscripts(pdf)` directly and merge into their MD pass.

# Apply just the glyph fixups to a chunk of already-extracted text.
# This is a cheap, safe pass that callers can sprinkle into existing
# pipelines (postprocess_md) without re-walking the PDF.

def normalise_math_glyphs(text: str) -> str:
    """Apply only the per-character glyph fix-ups (degree / prime / etc).

    Safe to call on any string -- no PDF re-extraction needed.
    Use this as a cheap upgrade in postprocess_md.
    """
    return _normalise_glyphs(text)


# ----------------------------------------------------------------------
# pymupdf4llm output cleanup
# ----------------------------------------------------------------------
# pymupdf4llm emits math-mode characters wrapped in markdown italic
# markers, e.g. ``35 _°_ 30 _[′]_ 59.5 _[″]_ N``. Visually this is
# *worse* than the underlying text because the italic markers don't
# render meaningfully on degree/prime symbols. Strip them in the
# specific contexts where they're math-mode artifacts.

# Pattern 1a: ``_[XX]_`` where XX is one or two math/super glyphs.
# This must come BEFORE single-glyph stripping so the brackets vanish.
_PYMUPDF_SUP_BRACKET_MULTI = re.compile(
    r"_\[((?:[′″°·×\u00b0\u00d7\u00b7\u2032\u2033]){1,3})\]_"
)
# Pattern 1b: ``X[Y]_`` -- common where X is an italic letter (e.g.
# Shannon's H) and Y is a superscript prime / digit. Found in MDPI
# papers like ``_H[′]_`` meaning italic H with superscript prime.
_PYMUPDF_SUP_BRACKET_INLINE = re.compile(
    r"\[((?:[′″°·×\u00b0\u00d7\u00b7\u2032\u2033]){1,3})\](?=_)"
)
# Pattern 2: ``_X_`` where X is a single math glyph and surrounded by
# digits / common math-mode neighbours.
_PYMUPDF_MATH_ITAL = re.compile(
    r"(?<=[\d\)])\s*_([×°·\u00b0\u00d7\u00b7])_\s*(?=[\d\(])"
)
# Pattern 3: degree-prime sequence with stray spaces. We run this AFTER
# the glyph and bracket cleanups so each math char is already standalone.
# Matches ``35°30′59.5″ N`` with optional spaces.
_COORD_DEGREE_RE = re.compile(
    r"(\d+)\s*°\s*(\d+)\s*′\s*(\d+(?:\.\d+)?)\s*″\s*([NSEW])"
)


def clean_pymupdf4llm_math(text: str) -> str:
    """Strip pymupdf4llm's italic markers around math/coord glyphs.

    Turns ``35 _°_ 30 _[′]_ 59.5 _[′′]_ N`` into ``35°30′59.5″ N``
    and ``7 _×_ 7`` into ``7×7``. Idempotent.

    Order of operations matters:
      1. Normalise glyphs (``′′`` -> ``″``, ``◦`` -> ``°``) first so
         the bracket-strip regex sees the canonical form.
      2. Strip ``_[X]_`` and ``_X_`` italic-wrapped math glyphs.
      3. Collapse coordinate sequences ``35° 30′ 59.5″ N`` to no spaces.
    """
    text = _normalise_glyphs(text)
    text = _PYMUPDF_SUP_BRACKET_MULTI.sub(r"\1", text)
    text = _PYMUPDF_SUP_BRACKET_INLINE.sub(r"\1", text)
    text = _PYMUPDF_MATH_ITAL.sub(r"\1", text)
    # Also drop _×_ even without strict digit neighbours (common math op).
    text = re.sub(r"\s_×_\s", " × ", text)
    # _±_ commonly appears in stats (mean ± SD); pymupdf4llm wraps it in
    # italic markers but ± is plain text, not italic.
    text = re.sub(r"_±_", "±", text)
    # Generic ``_X_`` where X is a single math operator and direct
    # digit neighbours on both sides (no whitespace, table-cell style).
    text = re.sub(r"(?<=\d)_([×÷±])_(?=\d)", r"\1", text)
    # Final coord-cleanup pass: turn "35° 30′ 59.5″ N" into "35°30′59.5″ N"
    text = _COORD_DEGREE_RE.sub(r"\1°\2′\3″ \4", text)
    return text


# Apply the supsub recovery to MARKDOWN, by matching it back against
# the PDF's PyMuPDF spans. Returns the same markdown with supsub tags
# spliced in where a PyMuPDF span tells us so.

def splice_supsub_into_markdown(markdown_per_page: Dict[int, str],
                                  pdf_path,
                                  *,
                                  html_supsub: bool = True) -> Dict[int, str]:
    """Augment per-page markdown with <sup>/<sub> markers from the PDF.

    For each page:
      1. Walk PyMuPDF spans, find every superscript and subscript run.
      2. Build short search keys: 2-3 chars BEFORE the superscript +
         the superscript text + 2-3 chars AFTER.
      3. In the markdown, find that key (allowing some whitespace flex)
         and replace the superscript text with the tagged version.
    This is heuristic but robust to most postprocessing because we
    anchor on neighbouring body text. Skips spans whose context is
    ambiguous.
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        out: Dict[int, str] = dict(markdown_per_page)
        for i, page in enumerate(doc):
            if i not in out:
                continue
            md = out[i]
            try:
                d = page.get_text("dict")
            except Exception:
                continue
            replacements: List[Tuple[str, str]] = []
            for block in d.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    spans: List[_Span] = []
                    for s in line["spans"]:
                        bb = s.get("bbox", (0, 0, 0, 0))
                        spans.append(_Span(
                            text=s.get("text", ""),
                            x0=bb[0], y0=bb[1], x1=bb[2], y1=bb[3],
                            size=float(s.get("size", 10.0)),
                            font=str(s.get("font", "")),
                            flags=int(s.get("flags", 0)),
                        ))
                    if not spans:
                        continue
                    L = _Line(spans)
                    for j, sp in enumerate(spans):
                        role = _classify(sp, L)
                        if role == "body":
                            continue
                        # Build context anchor (3 chars left + the ss
                        # text + 3 chars right).
                        sup_text = sp.text
                        if not sup_text.strip():
                            continue
                        left = "".join(spans[k].text
                                       for k in range(max(0, j - 2), j))[-4:]
                        right = "".join(spans[k].text
                                        for k in range(j + 1,
                                                       min(len(spans), j + 3)))[:4]
                        # Only attempt if we have at least one char of
                        # left context to avoid matching wrongly.
                        if not (left.strip() or right.strip()):
                            continue
                        if role == "super":
                            tag = (f"<sup>{_normalise_glyphs(sup_text)}</sup>"
                                   if html_supsub
                                   else f"^{{{_normalise_glyphs(sup_text)}}}")
                        else:
                            tag = (f"<sub>{_normalise_glyphs(sup_text)}</sub>"
                                   if html_supsub
                                   else f"_{{{_normalise_glyphs(sup_text)}}}")
                        old = left + sup_text + right
                        new = left + tag + right
                        if old != new:
                            replacements.append((old, new))
            # Apply first-occurrence replacements; later runs of the
            # same pattern stay alone to avoid double-tagging.
            for old, new in replacements:
                idx = md.find(old)
                if idx >= 0:
                    md = md[:idx] + new + md[idx + len(old):]
            out[i] = _normalise_glyphs(md)
        return out
    finally:
        doc.close()
