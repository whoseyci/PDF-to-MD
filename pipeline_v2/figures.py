"""
Figure pipeline: classify, filter, OCR, and caption-pair the images that
pymupdf4llm extracts from each PDF.

Pipeline per image
------------------

1.  **Junk filter** — drop publisher logos / journal cover thumbnails /
    ORCID "Check for updates" badges / decorative initials. Combines:
      * file-size threshold (very small files almost never carry data)
      * dimension thresholds (badge-sized squares ~100-180px)
      * tesseract OCR ASCII match against a publisher-keyword denylist

2.  **Caption pairing** — for every surviving image, scan the per-page
    raw markdown for a `**Fig. N.**` / `**Figure N.**` caption that
    follows the image reference. The caption text becomes the alt-text
    in the rewritten markdown.

3.  **OCR sidecar** — for figures that survive the junk filter, run
    tesseract over the image and store the OCR'd text in
    ``<slug>/figures/fig-NNN.txt``. This makes the figure's textual
    content searchable / accessible without bloating paper.md.

4.  **Rename + copy** — sequential ``fig-001.png``, ``fig-002.png`` …
    with stable mapping back to the original pymupdf4llm filename.

The module exposes a single entry point used by ``convert.py``:

    mapping, figs_meta = reorganize_and_describe_figures(
        raw_figs, out_dir, chunks
    )

``mapping`` maps original filename → ``"figures/fig-NNN.png ALT-TEXT"``
string that's spliced into the per-page markdown so the IMG_RE replacer
can build the final ``![alt](./figures/fig-001.png)`` line.

``figs_meta`` is the list of structured figure records written to
``paper.json``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard size floor: files smaller than this are almost always decorative.
_MIN_FILESIZE_BYTES = 5 * 1024  # 5 KB

# Soft size ceiling for "likely badge / logo" candidates (in bytes).
_LIKELY_LOGO_FILESIZE = 35 * 1024  # 35 KB

# Dimension ceiling for "likely badge / logo / icon" candidates (in pixels).
# Real figures in research papers are almost always > 250 px on the long axis.
_LIKELY_LOGO_DIM = 250

# Hard max aspect ratio — extreme strips (~ horizontal rules) are decorative.
_MAX_ASPECT_RATIO = 12.0

# OCR-detected text fragments that mark a SMALL image as decorative.
# Casefold-compared, substring match. We only invoke this list against
# images that already look badge-sized (< 250 px, < 35 KB) so the
# false-positive risk on real figures is low.
_PUBLISHER_DENYLIST = {
    # Publishers / aggregators
    "elsevier", "sciencedirect", "science direct",
    "springer", "springer nature",
    "wiley", "wiley online", "wiley-vch",
    "mdpi", "mdpi.com",
    "frontiers", "frontiersin",
    "copernicus",
    "cambridge university press",
    "oecd publishing",
    "taylor & francis", "routledge",
    # Cover / banner artefacts
    "iso", "issn",
    "orcid", "orcid.org",
    "creative commons", "cc by", "cc-by", "by-nc",
    "check for updates", "check for\nupdates",
    "open access",
    "this article is licensed",
    # Common journal-title cover-thumbnail strings in our corpus
    "ecosystems and environment", "ecosystems &",
    "ecosyst.", "agric. ecosyst", "ag. ecosyst",
    "j sustain agric environ",
    "soil tillage res", "soil & tillage res", "soil tillage",
    "renewable agriculture", "food systems",
    "plant and soil",
    "environmental science",
    "sustainability science",
    "land use policy",
    "earth systems",
    "geoderma",
    # Catch-all: a small image whose OCR contains BOTH the words
    # "agriculture" and ("ecosystems" OR "environment") is the Elsevier
    # Agric. Ecosyst. Environ. cover thumbnail.
}

# Two captioned-image patterns. Tried in order.
_CAPTION_RE_BOLD = re.compile(
    r"\*\*\s*Fig(?:ure|\.)?\s*(\d+)\s*[\.\)]?\s*\*\*\s*[—\-:]?\s*([^\n]+)"
)
_CAPTION_RE_PLAIN = re.compile(
    r"^\s*Fig(?:ure|\.)?\s+(\d+)\s*[\.\)]\s*([^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)

# Pymupdf4llm's image syntax: ![](path)
_IMG_REF_RE = re.compile(r"!\[\]\(([^)]+)\)")


# ---------------------------------------------------------------------------
# Junk filter
# ---------------------------------------------------------------------------

def _likely_junk_image(path: Path) -> Tuple[bool, str]:
    """Return (is_junk, reason) for a raw image path.

    Combines size, dimensions, aspect ratio and OCR-text matching against
    the publisher denylist. We're conservative: an image is only marked
    junk when the evidence is fairly clear.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return True, "file-missing"

    if size < _MIN_FILESIZE_BYTES:
        return True, f"size<{_MIN_FILESIZE_BYTES}"

    # Lazy import (PIL is mandatory, but keep it scoped).
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        # If we can't open the image, fall back to "keep" — better than
        # silently deleting an unknown asset.
        return False, "open-failed"

    # Strip-shaped images (horizontal rules, separator bars) are decorative.
    ar = max(w, h) / max(1, min(w, h))
    if ar > _MAX_ASPECT_RATIO:
        return True, f"aspect-ratio>{_MAX_ASPECT_RATIO}"

    # "Likely badge" candidates: small dimensions AND small file size.
    # These are the publisher logos / journal cover thumbnails / ORCID
    # badges / "Check for updates" buttons. We run OCR to confirm before
    # discarding.
    looks_like_badge = max(w, h) < _LIKELY_LOGO_DIM and size < _LIKELY_LOGO_FILESIZE
    if looks_like_badge:
        ocr_text = _ocr_image_text(path)
        if ocr_text:
            haystack = ocr_text.casefold()
            for needle in _PUBLISHER_DENYLIST:
                if needle in haystack:
                    return True, f"publisher-text:{needle}"
            # Combo: 'agriculture' AND ('ecosystem' OR 'environ') is the
            # Elsevier Agric. Ecosyst. Environ. cover thumbnail.
            if "agricultur" in haystack and (
                "ecosystem" in haystack or "environ" in haystack
            ):
                return True, "journal-cover:agric"
            # Combo: 'sustain' AND 'agri' is J. Sustain. Agric. cover
            if "sustain" in haystack and "agri" in haystack:
                return True, "journal-cover:sustain"
            # A tiny image (badge-sized) whose OCR text contains words
            # like 'agricultur', 'biological', 'biology', 'science' or
            # 'research' alone (with no chart-style numeric content) is
            # almost certainly a journal-cover thumbnail or section
            # banner — real figures with that single word would have
            # additional data labels.
            has_journal_word = any(w in haystack for w in (
                "agricultur", "biolog", "biology",
                "scientif", "research", "journal",
                "conservation",
            ))
            digits = sum(1 for c in haystack if c.isdigit())
            if has_journal_word and digits < 3:
                return True, "tiny+journal-word"
        # No OCR text in a tiny image = decorative icon (initial cap, logo, etc.)
        if not ocr_text or len(ocr_text.strip()) < 5:
            return True, "tiny-no-text"

    return False, ""


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _have_tesseract() -> bool:
    try:
        subprocess.run(
            ["tesseract", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


_TESSERACT_OK: Optional[bool] = None


def _ocr_image_text(path: Path, lang: str = "eng") -> str:
    """Return tesseract OCR output for an image, or '' on failure."""
    global _TESSERACT_OK
    if _TESSERACT_OK is None:
        _TESSERACT_OK = _have_tesseract()
    if not _TESSERACT_OK:
        return ""
    try:
        import pytesseract
        from PIL import Image
        with Image.open(path) as img:
            return pytesseract.image_to_string(img, lang=lang)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Caption pairing
# ---------------------------------------------------------------------------

def _find_caption_for_image(
    chunk_text: str, image_filename: str
) -> Optional[Tuple[str, str]]:
    """
    Look for a "**Fig. N.** caption..." pattern that immediately follows
    the image reference in the page markdown.

    Returns (caption_number, caption_text) or None.
    """
    # Locate the image ref by basename
    base = Path(image_filename).name
    pat = re.compile(r"!\[\]\([^)]*" + re.escape(base) + r"\)")
    m = pat.search(chunk_text)
    if not m:
        return None
    # Look in the next ~600 chars for a caption
    tail = chunk_text[m.end():m.end() + 600]
    cap = _CAPTION_RE_BOLD.search(tail)
    if cap:
        return cap.group(1), cap.group(2).strip()
    cap = _CAPTION_RE_PLAIN.search(tail)
    if cap:
        return cap.group(1), cap.group(2).strip()
    return None


def _ascii_alt(caption: str, max_len: int = 200) -> str:
    """Sanitise a caption for use as image alt-text in markdown.
    Strips inline formatting markers and trims length.
    """
    if not caption:
        return ""
    # Remove markdown italic/bold/underscore markers — they break alt-text
    s = re.sub(r"[\*_`]+", "", caption)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Escape characters that break markdown image syntax
    s = s.replace("]", ")").replace("[", "(")
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reorganize_and_describe_figures(
    raw_figs: Path, out_dir: Path, chunks
) -> Tuple[Dict[str, Optional[str]], List[dict]]:
    """
    Walk every image reference in the per-page chunks, filter out
    decorative junk, OCR the survivors, pair them with captions, and
    copy them into ``<out_dir>/figures/`` under sequential names.

    Returns
    -------
    mapping : dict
        ``original_filename → "figures/fig-NNN.png|alt text"``
        (or ``None`` when the image was filtered out).
        The vertical-bar separator is consumed by ``cleanup_page_markdown``
        to construct the final ``![alt](./figures/fig-NNN.png)`` line.
    figs_meta : list[dict]
        Per-figure structured metadata for ``paper.json``.
    """
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    mapping: Dict[str, Optional[str]] = {}
    figs_meta: List[dict] = []
    fig_idx = 0
    seen: set[str] = set()

    for ch in chunks:
        meta = ch.get("metadata", {})
        pno = meta.get("page_number") or meta.get("page") or 0
        chunk_text = ch.get("text", "")
        for m in _IMG_REF_RE.finditer(chunk_text):
            fname = Path(m.group(1)).name
            if fname in seen:
                continue
            seen.add(fname)

            raw = raw_figs / fname
            if not raw.exists():
                mapping[fname] = None
                continue

            # Junk filter
            is_junk, reason = _likely_junk_image(raw)
            if is_junk:
                mapping[fname] = None
                continue

            fig_idx += 1
            new_name = f"fig-{fig_idx:03d}.png"
            new_path = figs_dir / new_name
            shutil.copy2(raw, new_path)

            # Caption pairing
            cap = _find_caption_for_image(chunk_text, fname)
            cap_num = cap[0] if cap else None
            cap_text = cap[1] if cap else ""
            alt_text = (
                _ascii_alt(f"Fig. {cap_num} — {cap_text}")
                if cap_num
                else _ascii_alt(f"Figure {fig_idx} (page {pno})")
            )

            # OCR sidecar (only when tesseract is available; cost is ~50ms / image)
            ocr_text = _ocr_image_text(new_path).strip()
            ocr_path: Optional[str] = None
            if len(ocr_text) >= 40:  # only save substantive OCR output
                ocr_file = figs_dir / f"fig-{fig_idx:03d}.txt"
                ocr_file.write_text(ocr_text, encoding="utf-8")
                ocr_path = f"figures/fig-{fig_idx:03d}.txt"

            # The mapping value uses a `|` delimiter so the per-page cleanup
            # can carry both the rewritten path and the alt-text through.
            mapping[fname] = f"figures/{new_name}|{alt_text}"

            figs_meta.append({
                "id": f"fig-{fig_idx:03d}",
                "file": f"figures/{new_name}",
                "page": pno,
                "caption_number": cap_num,
                "caption_text": cap_text or None,
                "alt_text": alt_text,
                "ocr_text_file": ocr_path,
                "ocr_chars": len(ocr_text) if ocr_text else 0,
                "bytes": raw.stat().st_size,
            })

    shutil.rmtree(raw_figs, ignore_errors=True)
    return mapping, figs_meta
