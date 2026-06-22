"""
Layout-aware extraction of the abstract / keywords from page 1 of a PDF.

`pymupdf4llm` flattens 2-column layouts (like Elsevier journals' standard
"A R T I C L E I N F O  |  A B S T R A C T" page-1 panel) into a single line,
which interleaves keywords *inside* the abstract paragraph and breaks
ligatures in the process. The result is unreadable.

`pymupdf`'s `get_text("blocks")` mode preserves the column geometry, so we
can identify the abstract block and the keywords block from their bounding
boxes and emit clean markdown that the rest of the pipeline can then splice
into the per-page output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pymupdf


@dataclass
class FirstPageAbstract:
    """Structured page-1 metadata extracted via layout-aware block parsing."""

    abstract: str = ""
    keywords: List[str] = None  # type: ignore[assignment]
    abstract_bbox: Optional[Tuple[float, float, float, float]] = None
    keywords_bbox: Optional[Tuple[float, float, float, float]] = None
    layout: str = ""  # one of: "elsevier-2col", "wiley-abstract", "frontiers-abstract"


_ABSTRACT_HDR_PATTERNS = [
    re.compile(r"^\s*A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*$", re.IGNORECASE),
    re.compile(r"^\s*Abstract\s*$"),
]
_ARTICLE_INFO_HDR = re.compile(r"^\s*A\s*R\s*T\s*I\s*C\s*L\s*E\s*I\s*N\s*F\s*O\s*$", re.IGNORECASE)
_KEYWORDS_INLINE = re.compile(r"^\s*Keywords?\s*[:\.]?\s*$", re.IGNORECASE)
# Spaced versions: "K E Y W O R D S" (Wiley)
_KEYWORDS_SPACED = re.compile(r"^\s*K\s*E\s*Y\s*W\s*O\s*R\s*D\s*S\s*$", re.IGNORECASE)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _block_text(b) -> str:
    # pymupdf returns blocks as (x0, y0, x1, y1, text, block_no, block_type)
    return b[4]


def _bbox(b) -> Tuple[float, float, float, float]:
    return (b[0], b[1], b[2], b[3])


def _is_abstract_header(text: str) -> bool:
    t = text.strip().rstrip(":").strip()
    if len(t) > 30:
        return False
    return any(p.match(t) for p in _ABSTRACT_HDR_PATTERNS)


def _is_article_info_header(text: str) -> bool:
    t = text.strip()
    return bool(_ARTICLE_INFO_HDR.match(t))


def _is_keywords_header(text: str) -> bool:
    return bool(_KEYWORDS_INLINE.match(text.strip()))


def _split_keywords(raw: str) -> List[str]:
    """Split a raw keywords blob into a deduped, cleaned list."""
    # Strip leading 'Keywords:' / 'KEYWORDS\n' / 'Keywords ' / 'Keywords\u2002'
    cleaned = re.sub(
        r"(?i)^(?:keywords?|key\s*words?)\s*[:\.\u2002\s]*",
        "",
        raw,
        count=1,
    )
    # Decide on the separator family. We want to avoid splitting a single
    # multi-line keyword like 'European Green\nDeal' into two entries.
    # Rule: if the text uses an explicit non-comma separator (·, ;, |, •,
    # ' . '), use ONLY that; otherwise use commas with newlines flattened
    # to spaces.
    has_middot = "·" in cleaned
    has_semi = ";" in cleaned
    has_bullet = "•" in cleaned or " | " in cleaned
    has_springer_dot = bool(re.search(r"\S\s+\.\s+\S", cleaned))
    has_comma = "," in cleaned
    
    if has_springer_dot:
        # Springer 'A . B . C' format: dot-separated, BUT keywords often wrap
        # to a second line mid-word ('Olive\ngrove'). Replace inter-keyword
        # dots with semicolons, then flatten remaining newlines to spaces so
        # the wrap isn't a split.
        cleaned = re.sub(r"\s+\.\s+", "; ", cleaned)
        cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
        sep = r";"
    elif has_middot:
        # Middle-dot separator: flatten newlines so 'European Green\nDeal'
        # doesn't split.
        cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
        sep = r"·"
    elif has_bullet:
        sep = r"[•|]|\s*\n\s*"
    elif has_semi:
        sep = r"[;\n]"
    elif has_comma:
        # Comma-separated. Two competing publisher conventions:
        #   (a) Cuadros: "kw1, kw2, kw3\nkw4, kw5" — newline mid-list is
        #       just a wrap and must NOT split a multi-word keyword like
        #       "European Green\nDeal".
        #   (b) Baden-Böhm: "kw1, kw2 \nkw3 \nkw4 \nkw5" — line-wrapped
        #       keyword list where subsequent lines each carry exactly one
        #       keyword (no inter-keyword comma on those lines).
        # Heuristic: if EVERY line (when split on \n) contains at least one
        # comma, treat \n as a wrap. Otherwise, treat \n as a separator.
        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        all_lines_have_comma = lines and all("," in l for l in lines)
        if all_lines_have_comma:
            cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
            sep = r","
        else:
            sep = r"[,\n]"
    else:
        sep = r"\n"
    parts = re.split(sep, cleaned)
    cleaned_parts: List[str] = []
    for p in parts:
        s = re.sub(r"\u00a0|\u2002", " ", p)
        s = re.sub(r"\s+", " ", s).strip().strip(",;.")
        if s:
            cleaned_parts.append(_expand_ligatures(s))
    # Post-process: if we split on newline only (no commas/etc.) and we got
    # adjacent fragments where one starts with a lowercase letter ('Deal'
    # after 'European Green'), merge them. Skip when commas were present —
    # in that case newline-splits inside multi-word keywords are unusual.
    merged: List[str] = []
    for s in cleaned_parts:
        if (
            merged
            and not has_comma
            and not has_semi
            and not has_middot
            and not has_bullet
            and not has_springer_dot
            and s[0].islower()
        ):
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    out: List[str] = []
    seen = set()
    for s in merged:
        if len(s) > 80:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# Unicode presentation-form ligatures that need expanding back to ASCII.
_LIGATURE_EXPAND = {
    "\uFB00": "ff",  # ﬀ
    "\uFB01": "fi",  # ﬁ
    "\uFB02": "fl",  # ﬂ
    "\uFB03": "ffi",  # ﬃ
    "\uFB04": "ffl",  # ﬄ
    "\uFB05": "st",  # ﬅ
    "\uFB06": "st",  # ﬆ
}


def _expand_ligatures(text: str) -> str:
    for src, dst in _LIGATURE_EXPAND.items():
        text = text.replace(src, dst)
    return text


def _join_block_text(raw: str) -> str:
    """Join soft-wrapped lines inside a block into one paragraph.

    pymupdf block text breaks lines at line wraps. Joining with a single
    space (and respecting hyphenation `infiltrat\nion` → `infiltration`)
    reconstructs the original sentence. Presentation-form ligatures
    (ﬁ, ﬂ, ﬃ…) are also expanded.
    """
    # Soft-hyphen-style line breaks: 'word-\nrest' → 'wordrest'
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
    # Other line breaks → spaces
    raw = re.sub(r"\s*\n\s*", " ", raw)
    # Collapse whitespace
    raw = re.sub(r"\s+", " ", raw).strip()
    # Expand presentation-form ligatures
    raw = _expand_ligatures(raw)
    return raw


def extract_first_page_abstract(pdf_path: str) -> Optional[FirstPageAbstract]:
    """
    Run pymupdf block extraction on page 1 and try to recover a clean
    abstract + keywords pair, regardless of how pymupdf4llm rendered it.

    Returns None if no abstract block could be identified.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return None
    try:
        if len(doc) == 0:
            return None
        page = doc[0]
        blocks = page.get_text("blocks")
        # Also grab page 2 blocks — some publishers (Springer Sustain. Sci.)
        # put the Keywords section at the top of page 2.
        page2_blocks = doc[1].get_text("blocks") if len(doc) > 1 else []
    finally:
        doc.close()

    if not blocks:
        return None
    
    result = _extract_from_blocks(blocks)
    if result is None:
        return None
    
    # If no keywords found on page 1, try page 2 (only its top portion).
    if not result.keywords and page2_blocks:
        for b in page2_blocks:
            x0, y0, x1, y1, txt, _, _ = b
            if y0 > 300:  # only top of page 2
                continue
            m = re.match(
                r"^\s*(?:Keywords?|KEYWORDS?|Key\s*words?)\s*[:\.\u2002\s]+(.*)",
                txt,
                re.DOTALL,
            )
            if m:
                result.keywords = _split_keywords(m.group(1))
                break

    return result


def _extract_from_blocks(blocks):
    """Inner page-1 extractor used by `extract_first_page_abstract`."""

    abs_hdr = None
    aninfo_hdr = None
    inline_abs_block = None
    for b in blocks:
        text = _block_text(b)
        if _is_abstract_header(text):
            if abs_hdr is None or b[1] < abs_hdr[1]:
                abs_hdr = b
        if _is_article_info_header(text):
            if aninfo_hdr is None or b[1] < aninfo_hdr[1]:
                aninfo_hdr = b
        # Inline abstract patterns:
        #   'Abstract: ...prose...'      MDPI
        #   'Abstract\n...prose...'      MDPI/Springer
        #   'Abstract. ...prose...'      Copernicus (SOIL journal)
        #   'Abstract\u2003...prose...'  Springer (em-space)
        #   'Abstract ...prose...'       Springer/Wiley (regular space) — only
        #                                accept this when the next char is
        #                                a capital letter to avoid false hits.
        stripped = text.lstrip()
        if len(text) > 120 and re.match(
            r"(?:Abstract|ABSTRACT)(?:[:\.\u2003]|\s*\n|\s+[A-Z])",
            stripped,
        ):
            if inline_abs_block is None or b[1] < inline_abs_block[1]:
                inline_abs_block = b

    if abs_hdr is None and aninfo_hdr is None and inline_abs_block is None:
        # No header-based abstract; the Frontiers fallback still might
        # recognise the layout by anchoring on a KEYWORDS block.
        return _extract_frontiers_abstract(blocks)

    # -------- Case 1: Elsevier 2-column layout (both headers present) --------
    if abs_hdr is not None and aninfo_hdr is not None:
        # The two headers should be roughly horizontally aligned (similar y0)
        if abs(abs_hdr[1] - aninfo_hdr[1]) < 30 and aninfo_hdr[0] < abs_hdr[0]:
            return _extract_elsevier_two_col(blocks, aninfo_hdr, abs_hdr)

    # -------- Case 2: lone "Abstract" header (Wiley / Frontiers / etc.) --------
    if abs_hdr is not None:
        return _extract_lone_abstract(blocks, abs_hdr)

    # -------- Case 3: MDPI inline "Abstract: ..." (no header) --------
    if inline_abs_block is not None:
        return _extract_mdpi_inline_abstract(blocks, inline_abs_block)

    # -------- Case 4: Frontiers — abstract is a wide right-column block
    # immediately followed by a 'Keywords:' block in the same column.
    # There's no 'Abstract' label on the page at all.
    frontiers = _extract_frontiers_abstract(blocks)
    if frontiers is not None:
        return frontiers

    return None


def _extract_frontiers_abstract(blocks) -> Optional[FirstPageAbstract]:
    """Frontiers-style page 1: title block, author block, then a wide
    right-column block that IS the abstract, then a 'KEYWORDS' header
    block, then a keyword-list block."""
    # Find a 'KEYWORDS' or 'Keywords:' block first (anchor point).
    kw_hdr_block = None  # header-only block
    kw_inline_block = None  # 'Keywords: kw1, kw2' inline block
    for b in blocks:
        text = _block_text(b).strip()
        # Header-only (CAST/Frontiers): "KEYWORDS" (very short block)
        if re.fullmatch(r"(?:Keywords?|KEYWORDS?)\s*[:\.]?", text):
            if kw_hdr_block is None or b[1] < kw_hdr_block[1]:
                kw_hdr_block = b
        # Inline (other Frontiers): "Keywords: a, b, c..."
        elif re.match(r"(?:Keywords?|KEYWORDS?)\s*[:\.]\s*\S", text, re.IGNORECASE):
            if kw_inline_block is None or b[1] < kw_inline_block[1]:
                kw_inline_block = b
    # Prefer the inline block (it carries the keywords directly).
    kw_block = kw_inline_block or kw_hdr_block
    if kw_block is None:
        return None
    
    # The abstract is the wide column block immediately preceding the
    # Keywords block (similar x0, similar width, and y1 close to kw_block's y0).
    kw_x0, kw_y0 = kw_block[0], kw_block[1]
    kw_width = kw_block[2] - kw_block[0]
    candidate = None
    for b in blocks:
        x0, y0, x1, y1, txt, _, _ = b
        if y1 > kw_y0 or y1 < kw_y0 - 400:
            continue
        # Same column (within 30pt of x0) and at least 200pt wide
        if abs(x0 - kw_x0) > 30:
            continue
        if (x1 - x0) < 200:
            continue
        # Must look like prose (≥ 300 chars and starts with a capital)
        stripped = txt.lstrip()
        if len(stripped) < 300:
            continue
        if not stripped[0].isupper():
            continue
        # Prefer the block with the largest y1 (latest, closest to kw_block).
        if candidate is None or y1 > candidate[3]:
            candidate = b
    if candidate is None:
        return None
    
    abstract = _join_block_text(_block_text(candidate))
    if len(abstract) < 200:
        return None
    
    # Parse keywords. If we anchored on the header-only KEYWORDS block,
    # the keywords themselves are in the next block down in the same column.
    keywords: List[str] = []
    kw_text = _block_text(kw_block).strip()
    if re.fullmatch(r"(?:Keywords?|KEYWORDS?)\s*[:\.]?", kw_text):
        # Header only — find the next block immediately below
        nxt = None
        for b in sorted(blocks, key=lambda b: b[1]):
            if b[1] <= kw_block[3]:
                continue
            if abs(b[0] - kw_block[0]) > 30:
                continue
            if b[1] - kw_block[3] > 50:
                break
            nxt = b
            break
        if nxt is not None:
            keywords = _split_keywords(_block_text(nxt))
    else:
        m = re.match(
            r"\s*(?:Keywords?|KEYWORDS?)\s*[:\.\u2002\s]+(.*)",
            kw_text, re.DOTALL,
        )
        if m:
            keywords = _split_keywords(m.group(1))
    
    return FirstPageAbstract(
        abstract=abstract,
        keywords=keywords,
        abstract_bbox=_bbox(candidate),
        keywords_bbox=_bbox(kw_block),
        layout="frontiers-no-header",
    )


def _extract_mdpi_inline_abstract(blocks, abs_block) -> Optional[FirstPageAbstract]:
    """MDPI/Springer-style page 1: 'Abstract: ...prose...' (or 'Abstract\n...prose...')
    all in one block, with a separate 'Keywords: ...' block usually nearby in the
    same column."""
    text = _block_text(abs_block)
    # Strip the leading 'Abstract:' / 'Abstract.' / 'Abstract\n' / 'Abstract '
    m = re.match(r"\s*(?:Abstract|ABSTRACT)(?:[:\.\u2003]\s*|\s*\n\s*|\s+(?=[A-Z]))", text)
    if not m:
        return None
    raw = text[m.end():]
    # MDPI sometimes follows the abstract with a 'Keywords:' line inside the
    # same block. Split on first 'Keywords:' to bound the abstract.
    kw_match = re.search(r"\n\s*(?:Keywords?|KEYWORDS?)\s*:\s*(.*)$", raw, re.DOTALL)
    if kw_match:
        abstract = raw[: kw_match.start()].strip()
        kw_raw = kw_match.group(1)
    else:
        abstract = raw.strip()
        kw_raw = ""

    abstract = _join_block_text(abstract)
    if len(abstract) < 120:
        return None

    keywords: List[str] = []
    if kw_raw:
        keywords = _split_keywords(kw_raw)
    else:
        # Pass 1: prefer keywords block in same column (typical MDPI style)
        for b in blocks:
            text2 = _block_text(b)
            m2 = re.match(
                r"^\s*(?:Keywords?|KEYWORDS?|Key\s*words?)\s*[:\.\u2002\s]+(.*)",
                text2,
                re.DOTALL,
            )
            if m2 and abs(b[0] - abs_block[0]) < 60 and b[1] > abs_block[1]:
                keywords = _split_keywords(m2.group(1))
                break
        # Pass 2: anywhere on page (Springer has them in a different column)
        if not keywords:
            for b in blocks:
                text2 = _block_text(b)
                m2 = re.match(
                    r"^\s*(?:Keywords?|KEYWORDS?|Key\s*words?)\s*[:\.\u2002\s]+(.*)",
                    text2,
                    re.DOTALL,
                )
                if m2:
                    keywords = _split_keywords(m2.group(1))
                    break

    return FirstPageAbstract(
        abstract=abstract,
        keywords=keywords,
        abstract_bbox=_bbox(abs_block),
        keywords_bbox=None,
        layout="mdpi-inline-abstract",
    )


def _extract_elsevier_two_col(
    blocks, ainfo_hdr, abst_hdr
) -> Optional[FirstPageAbstract]:
    # Abstract content blocks: x0 close to abst_hdr's x0, y0 > abst_hdr's y1
    abst_y = abst_hdr[3]
    abst_x = abst_hdr[0]
    # Keywords block(s): x0 close to ainfo_hdr's x0, y0 > ainfo_hdr's y1
    ainfo_y = ainfo_hdr[3]
    ainfo_x = ainfo_hdr[0]

    abstract_blocks = []
    kw_block = None
    for b in blocks:
        x0, y0, x1, y1, _, _, _ = b
        if y0 <= abst_y - 1:
            continue
        # Right column (abstract). Allow ±60pt x drift.
        if abs(x0 - abst_x) < 60 and x1 - x0 > 100:
            abstract_blocks.append(b)
        # Left column block whose text starts with "Keywords"
        elif abs(x0 - ainfo_x) < 30 and x0 < abst_x - 40:
            text = _block_text(b).strip()
            if (
                kw_block is None
                and re.match(r"(?i)^keywords?\s*[:\.]?", text)
            ):
                kw_block = b

    # Stop the abstract at the first big vertical gap (heuristic: > 80pt
    # between successive blocks) — this typically marks the end of page-1
    # of the abstract before the body re-flows into the right column.
    abstract_blocks.sort(key=lambda b: b[1])
    if not abstract_blocks:
        return None
    kept = [abstract_blocks[0]]
    for b in abstract_blocks[1:]:
        prev = kept[-1]
        if b[1] - prev[3] > 80:
            break
        kept.append(b)

    abstract_text = " ".join(_join_block_text(_block_text(b)) for b in kept)
    abstract_text = re.sub(r"\s+", " ", abstract_text).strip()

    # Sanity check
    if len(abstract_text) < 120:
        return None

    keywords: List[str] = []
    if kw_block is not None:
        keywords = _split_keywords(_block_text(kw_block))

    return FirstPageAbstract(
        abstract=abstract_text,
        keywords=keywords,
        abstract_bbox=_bbox(kept[0]),
        keywords_bbox=_bbox(kw_block) if kw_block else None,
        layout="elsevier-2col",
    )


def _extract_lone_abstract(blocks, abst_hdr) -> Optional[FirstPageAbstract]:
    """Single-column 'Abstract' page (Wiley, Frontiers, some MDPI)."""
    abst_y = abst_hdr[3]
    abst_x = abst_hdr[0]

    # Locate a KEYWORDS header in the same column to bound the abstract.
    kw_hdr = None
    for b in blocks:
        x0, y0, x1, y1, _, _, _ = b
        if y0 <= abst_y:
            continue
        if abs(x0 - abst_x) > 80:
            continue
        text = _block_text(b).strip()
        if (
            _KEYWORDS_SPACED.match(text)
            or re.match(r"^(?:KEYWORDS?|Keywords?)\s*[:\.]?\s*$", text)
        ):
            kw_hdr = b
            break

    # Locate a section heading ("Introduction", "1. Introduction", "Materials
    # and Methods", "Background", "Overview") in the same column — that
    # bounds the abstract from below.
    section_y_bound = None
    sect_re = re.compile(
        r"^\s*(?:\d+\.?\s*)?(?:Introduction|Background|Overview|"
        r"Materials\s+and\s+Methods|Methods|Results)\s*\.?\s*$",
        re.IGNORECASE,
    )
    for b in blocks:
        x0, y0, x1, y1, _, _, _ = b
        if y0 <= abst_y:
            continue
        if abs(x0 - abst_x) > 80:
            continue
        text = _block_text(b).strip()
        if sect_re.match(text):
            section_y_bound = y0
            break

    # Collect candidate blocks in same column below abstract header, above
    # the keywords header and above any section heading.
    candidate_blocks = []
    for b in blocks:
        x0, y0, x1, y1, _, _, _ = b
        if y0 <= abst_y - 1:
            continue
        if kw_hdr is not None and y0 >= kw_hdr[1] - 1:
            continue
        if section_y_bound is not None and y0 >= section_y_bound - 1:
            continue
        if abs(x0 - abst_x) < 80 and x1 - x0 > 100:
            candidate_blocks.append(b)

    candidate_blocks.sort(key=lambda b: b[1])
    if not candidate_blocks:
        return None

    # Stop at first block that looks like a section heading
    kept = []
    for b in candidate_blocks:
        text = _block_text(b).strip()
        if re.match(r"^(?:KEYWORDS?|Keywords?)[:\s]", text):
            break
        if re.match(r"^\s*\d+\s*[.|]?\s*(Introduction|Materials|Methods|Background|Overview)\b", text, re.IGNORECASE):
            break
        if len(text) < 5:
            continue
        kept.append(b)
        if sum(len(_block_text(k)) for k in kept) > 5000:
            break
        if len(kept) >= 2:
            if b[1] - kept[-2][3] > 80:
                kept.pop()
                break

    if not kept:
        return None

    abstract_text = " ".join(_join_block_text(_block_text(b)) for b in kept)
    abstract_text = re.sub(r"\s+", " ", abstract_text).strip()
    if len(abstract_text) < 120:
        return None

    # Keywords: try the inline "Keywords:" header first, then look at the
    # block immediately after a separate "K E Y W O R D S" header.
    keywords: List[str] = []
    if kw_hdr is not None:
        # Find the block right below kw_hdr in same column
        kw_y = kw_hdr[3]
        kw_x = kw_hdr[0]
        for b in sorted(blocks, key=lambda b: b[1]):
            x0, y0, x1, y1, _, _, _ = b
            if y0 > kw_y and abs(x0 - kw_x) < 60 and x1 - x0 > 50:
                kw_text = _block_text(b)
                # Only the immediate next block — keywords are usually one
                # short line. If next block is more than 30pt below kw header
                # then it's probably not the keywords list.
                if y0 - kw_y < 30:
                    keywords = _split_keywords(kw_text)
                break
    if not keywords:
        # Fallback: find a block that starts with "Keywords:" or "KEYWORDS"
        for b in blocks:
            text = _block_text(b).strip()
            m = re.match(r"^(?:KEYWORDS?|Keywords?)\s*[:\.]\s*(.*)", text, re.DOTALL)
            if m:
                rest = m.group(1).strip()
                if rest:
                    keywords = _split_keywords(rest)
                break

    return FirstPageAbstract(
        abstract=abstract_text,
        keywords=keywords,
        abstract_bbox=_bbox(kept[0]),
        keywords_bbox=_bbox(kw_hdr) if kw_hdr else None,
        layout="single-column-abstract",
    )
