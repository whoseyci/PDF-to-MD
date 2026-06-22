"""
Improved reference parsing + multi-style in-text citation linking.

Supported reference list formats:
  - Bullet list (`- text`)
  - Numbered (`1. text` or `[1] text`)
  - Plain paragraphs separated by blank lines

Supported in-text citation styles (all will be linked when matched):
  - (Author, Year)
  - (Author et al., Year)
  - (Author and Other, Year)
  - (Author & Other, Year)
  - Author (Year)
  - (Author1, Year1; Author2, Year2)   -> each linked separately
  - [12], [12-15], [12, 14]            -> numeric refs

Output:
  references = list of {id, text, doi, year, first_author_surname, authors_short}
  Body text has citations replaced with markdown links to #ref-NNN
"""
import re
from collections import defaultdict


REF_HEADING_RE = re.compile(
    r"^#{1,4}\s*\**\s*(references?|bibliography|literature\s+cited|works\s+cited)\s*\**\s*$",
    re.IGNORECASE | re.MULTILINE,
)

DOI_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|doi[:\s]+)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)

YEAR_PAREN_RE = re.compile(r"\((\d{4})[a-z]?\)")
YEAR_BARE_RE = re.compile(r"(?<!\d)(19|20)\d{2}[a-z]?(?!\d)")

# Surname pattern: starts with capital, may have accents, hyphens, apostrophes
SURNAME_CHARS = r"A-Za-zÀ-ÿĀ-ž\-'"
SURNAME = rf"[A-ZÁÉÍÓÚÑÀÈÌÒÙ][{SURNAME_CHARS}]+"


def split_references_section(md):
    m = REF_HEADING_RE.search(md)
    if not m:
        return md, ""
    return md[:m.start()], md[m.start():]


# ---------------------------------------------------------------------------
# Reference list parsing
# ---------------------------------------------------------------------------

def _parse_one_ref(text, idx):
    # idx can be int or string; format as 3-digit when numeric
    try:
        rid = f"ref-{int(idx):03d}"
    except (TypeError, ValueError):
        rid = f"ref-{idx}"
    # Strip any PDF page-boundary markers that fell inside this ref's text
    # (when a reference's body straddles a page break). The markers are
    # invisible U+2063 tokens; keep them out of the rendered ref text.
    text = re.sub(r"\u2063{3}PB\d+/\d+\u2063{3}", " ", text)
    # Strip leading bullet/number markers that may have leaked through
    text = re.sub(r"^[\-\*\u2022]\s*|^\[\d+\]\s*|^\d+\.\s*", "", text).strip()
    doi_m = DOI_RE.search(text)
    year_m = YEAR_PAREN_RE.search(text)
    year = None
    if year_m:
        try:
            year = int(year_m.group(1))
        except ValueError:
            pass
    if year is None:
        bare = YEAR_BARE_RE.search(text)
        if bare:
            digits = re.match(r"\d{4}", bare.group())
            if digits:
                try:
                    year = int(digits.group())
                except ValueError:
                    pass
    
    # Extract first author surname
    first_author = ""
    # Common patterns: "Smith, J.", "Smith J.", "Smith, J. & Jones, K."
    m_author = re.match(rf"^({SURNAME})(?:,?\s+[A-Z]\.?)*", text)
    if m_author:
        first_author = m_author.group(1).strip()
    
    # Get up to ~3 author surnames (for fuzzy matching multi-author cites)
    authors = re.findall(rf"({SURNAME})(?=,?\s+[A-Z]\.|\s+et\s+al)", text[:300])
    
    return {
        "id": rid,
        "text": text,
        "doi": doi_m.group(1) if doi_m else None,
        "year": year,
        "first_author_surname": first_author or None,
        "co_author_surnames": authors[:5],
    }


def parse_references(ref_md):
    if not ref_md:
        return []
    
    body = re.sub(REF_HEADING_RE, "", ref_md, count=1).strip()
    
    # Some PDFs render the reference list as a 2-column markdown table
    # ( | N. | Author, X.Y... | ). Detect this and flatten back to numbered lines.
    # We also collapse continuation rows ( |  | rest-of-ref | ) into the previous entry.
    if re.search(r"^\s*\|\s*\d+\.\s*\|", body, re.MULTILINE):
        flat_lines = []
        for line in body.split("\n"):
            if not line.lstrip().startswith("|"):
                flat_lines.append(line)
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Pure separator row: |---|---|
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                continue
            if len(cells) >= 2 and re.match(r"^\d+\.?$", cells[0]):
                num = cells[0].rstrip(".")
                rest = " ".join(c for c in cells[1:] if c)
                flat_lines.append(f"\n{num}. {rest}")
            elif len(cells) >= 2 and not cells[0] and cells[-1]:
                # Continuation row — append to previous line in-place
                if flat_lines:
                    flat_lines[-1] = flat_lines[-1].rstrip() + " " + " ".join(c for c in cells if c)
            else:
                # Single-column table cell, treat as continuation/paragraph
                flat_lines.append(" ".join(c for c in cells if c))
        body = "\n".join(flat_lines)
    
    # Pre-split: pymupdf4llm sometimes joins two refs on one line, e.g.:
    # "20. Flexas, J.; ... [PubMed] 21. Martinez, J.; Keller, M. ..."
    # Insert a newline before " NN. Capital" when the prior token looks like
    # the end of a citation (closing bracket / period / lowercase letter).
    body = re.sub(
        r"(?<=[\]\.\)a-z0-9])\s+(\d{1,3})\.\s+(?=[A-Z][a-zA-ZáéíóúñÁÉÍÓÚÑ\-]{1,40}[,;\s])",
        r"\n\1. ",
        body,
    )
    
    # Try all 3 strategies and keep the one that produces the most entries
    strategies = {}
    
    # Strategy 1: bullets
    bullet_re = re.compile(r"^[-*]\s+(.+?)(?=\n[-*]\s+|\Z)", re.MULTILINE | re.DOTALL)
    bullet_refs = []
    for m in bullet_re.finditer(body):
        t = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(t) > 20:
            bullet_refs.append(_parse_one_ref(t, len(bullet_refs) + 1))
    strategies["bullets"] = bullet_refs
    
    # Strategy 2: numbered — preserve original numbering for accurate cite linking
    num_re = re.compile(
        r"(?:^|\n)\s*(?:\[(\d+)\]|(\d+)\.)\s+(.+?)(?=\n\s*(?:\[\d+\]|\d+\.)|\Z)",
        re.DOTALL,
    )
    num_refs = []
    seen_nums = set()
    for m in num_re.finditer(body):
        t = re.sub(r"\s+", " ", m.group(3)).strip()
        if len(t) <= 20:
            continue
        orig_num = int(m.group(1) or m.group(2))
        # Detect bogus matches where numbers don't proceed monotonically.
        # Accept duplicates only as next-sequence increment.
        if orig_num in seen_nums:
            continue
        seen_nums.add(orig_num)
        num_refs.append(_parse_one_ref(t, orig_num))
    # Quick sanity: numbers should be roughly contiguous from 1
    if num_refs:
        nums_extracted = sorted(int(r["id"].split("-")[1]) for r in num_refs)
        # Reject if first number > 5 (suggests garbage match)
        if nums_extracted[0] > 5:
            num_refs = []
    strategies["numbered"] = num_refs
    
    # Strategy 3: paragraph-per-ref
    para_refs = []
    for para in re.split(r"\n\s*\n", body):
        t = re.sub(r"\s+", " ", para).strip()
        if len(t) > 30 and re.search(r"\b(19|20)\d{2}\b", t):
            para_refs.append(_parse_one_ref(t, len(para_refs) + 1))
    strategies["paragraphs"] = para_refs
    
    # Pick the strategy with the most parsed entries
    best = max(strategies.items(), key=lambda kv: len(kv[1]))
    return best[1]


# ---------------------------------------------------------------------------
# In-text citation linking
# ---------------------------------------------------------------------------

# Support lowercased connectors that may appear in surnames: "de Torres", "van der Berg"
# Single author "name unit" = optional lowercase particle + surname
NAME_UNIT = rf"(?:(?:de|van|van\s+der|von|del|della|du|le|la|el|al-|el-)\s+)?{SURNAME}"

# Match parenthesized author-year citations, including multi-cite forms:
#   (Smith, 2024)
#   (Smith 2024)                  -- no comma
#   (Smith et al., 2024)
#   (Smith et al. 2024)           -- no comma
#   (Smith and Jones, 2024)
#   (Smith and Jones 2024)
#   (de Torres et al. 2018)       -- lowercase particle
#   (Smith 2018a, 2021)           -- same author multiple years
#   (Smith 2024; Jones 2025)      -- multi-cite separated by ;
#   (OECD, 2023[6])               -- OECD-style chapter-numbered ref (trailing [N])
INLINE_CITE_RE = re.compile(
    rf"""\(
        (
            (?:                                   # one citation
                {NAME_UNIT}
                (?:\s+(?:et\s+al\.?|and\s+{NAME_UNIT}|&\s+{NAME_UNIT}))?
                ,?\s+
                \d{{4}}[a-z]?
                (?:\[\d+\])?                      # optional OECD-style ref number
                (?:,\s*\d{{4}}[a-z]?(?:\[\d+\])?)*  # additional years for same author
            )
            (?:\s*;\s*                            # separator for multi-cite
                {NAME_UNIT}
                (?:\s+(?:et\s+al\.?|and\s+{NAME_UNIT}|&\s+{NAME_UNIT}))?
                ,?\s+
                \d{{4}}[a-z]?
                (?:\[\d+\])?
                (?:,\s*\d{{4}}[a-z]?(?:\[\d+\])?)*
            )*
        )
        \s*                                       # tolerate stray space before )
    \)""",
    re.VERBOSE,
)

# Prose citation: "Author et al. (Year)" / "Author (Year)" / "Author and Other (Year)"
PROSE_CITE_RE = re.compile(
    rf"({NAME_UNIT}(?:\s+(?:et\s+al\.?|and\s+{NAME_UNIT}|&\s+{NAME_UNIT}))?\s+\((\d{{4}})[a-z]?\))"
)

# Bracket numeric: [12], [12, 15], [12-15], [12-15, 17]
# CRITICAL: Only match when preceded by whitespace, punctuation, or start-of-line.
# This avoids matching math superscripts like "cm[-][1]" or "x[2]" which are
# attached directly to letters/digits.
BRACKET_CITE_RE = re.compile(r"(?:^|(?<=[\s\(,;]))\[(\d+(?:\s*[\-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[\-–]\s*\d+)?)*)\]")

# Parenthesized numeric: (12), (12, 15), (12-15), (12-15, 17)
# Some journals use this style. We need to be careful not to match years like "(2024)".
# Match only if numbers are SMALL (< 999, typical citation range).
PAREN_NUM_CITE_RE = re.compile(
    r"\((\d{1,3}(?:\s*[\-–]\s*\d{1,3})?(?:\s*,\s*\d{1,3}(?:\s*[\-–]\s*\d{1,3})?)*)\)"
)


def _build_ref_indexes(references):
    """Build lookup tables to match citations to ref IDs."""
    by_author_year = defaultdict(list)
    by_number = {}
    for i, r in enumerate(references):
        # By (surname.lower(), year)
        if r.get("first_author_surname") and r.get("year"):
            key = (r["first_author_surname"].lower(), r["year"])
            by_author_year[key].append(r["id"])
        # By the actual number embedded in the id (ref-NNN). For author-year
        # bibliographies this is also the 1-indexed position; for numbered
        # bibliographies it's the originally cited number.
        try:
            n = int(r["id"].split("-")[-1])
        except (ValueError, IndexError, KeyError):
            n = i + 1
        by_number[n] = r["id"]
        # Also keep position-based fallback so author-year style is unaffected
        by_number.setdefault(i + 1, r["id"])
    return by_author_year, by_number


def _link_one_cite(inner, by_author_year):
    """Given the inner content of a (...) citation, return linked markdown.
    Handles: 'Smith 2024', 'Smith, 2024', 'Smith 2024a, 2025', 
             'Smith et al. 2024', 'de Torres et al. 2018'."""
    pieces = [p.strip() for p in re.split(r"\s*;\s*", inner)]
    linked_pieces = []
    for piece in pieces:
        # Extract the first surname (possibly with lowercase particle) and a year
        m = re.match(
            rf"({NAME_UNIT})"
            rf"(?:\s+et\s+al\.?|\s+and\s+{NAME_UNIT}|\s*&\s*{NAME_UNIT})?"
            rf",?\s+(\d{{4}})[a-z]?"
            rf"(?:,\s*(\d{{4}})[a-z]?)*",  # optional additional year
            piece,
        )
        if not m:
            linked_pieces.append(piece)
            continue
        full_match = m.group(0)
        # Extract just the SURNAME from name_unit (drop particle)
        name_part = m.group(1)
        sm = re.search(SURNAME + r"$", name_part)
        surname = sm.group().lower() if sm else name_part.lower()
        year = int(m.group(2))
        ids = by_author_year.get((surname, year), [])
        if ids:
            # Link the matched citation; preserve any trailing text after the citation
            trailing = piece[len(full_match):]
            linked_pieces.append(f"[{full_match}](#{ids[0]}){trailing}")
        else:
            linked_pieces.append(piece)
    return "(" + "; ".join(linked_pieces) + ")"


def _expand_numeric_range(spec):
    """Expand '12-15' to [12,13,14,15], '12, 14' to [12, 14], etc."""
    nums = []
    for part in re.split(r'\s*,\s*', spec):
        rng = re.match(r'(\d+)\s*[\-\u2013]\s*(\d+)', part)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            if 0 < b - a < 50:
                nums.extend(range(a, b + 1))
        else:
            try:
                nums.append(int(part))
            except ValueError:
                pass
    return nums


def _references_are_numeric_style(references):
    """
    Heuristic: are the references a NUMBERED list (e.g. [1] Smith J., ...)
    or AUTHOR-YEAR (Smith J. (2024) ...)? If numbered, [N] in text should link;
    if author-year, [N] is probably a footnote marker, not a citation.
    
    Detection: if references' raw text frequently starts with a number-and-punct
    boundary that looks like list numbering, treat as numeric.
    """
    if not references:
        return False
    # We assigned IDs ref-001 etc.; we want to check the original text shape.
    # Numbered refs are often parsed from "[1] Smith..." or "1. Smith...". After
    # parsing, the text doesn't keep the "[1]"/"1." prefix, but a quick proxy is:
    # if MOST refs lack year-in-parens but have a year somewhere, they're often numbered.
    # Better: just check whether enough refs were parsed with valid surnames+years.
    n_with_author_year = sum(
        1 for r in references
        if r.get("first_author_surname") and r.get("year")
    )
    ratio = n_with_author_year / max(len(references), 1)
    # If 70%+ have a clear (Surname, Year) signature -> author-year style
    return ratio < 0.5


def link_citations(body_md, references):
    if not references:
        return body_md
    by_author_year, by_number = _build_ref_indexes(references)
    
    # ------------------------------------------------------------------
    # Protect image alt-text from citation linking. We swap every
    # `![alt](path)` for an opaque placeholder, run the linker, then
    # restore the originals. Otherwise `(1)` inside alt-text would
    # become `[1](#ref-001)` and break the image syntax.
    # ------------------------------------------------------------------
    _IMG_PLACEHOLDER_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
    _placeholders: list[str] = []
    def _stash(m):
        _placeholders.append(m.group(0))
        return f"\u2063IMG{len(_placeholders)-1}\u2063"
    body_md = _IMG_PLACEHOLDER_RE.sub(_stash, body_md)
    
    # 1. Inline cites
    def repl_inline(m):
        return _link_one_cite(m.group(1), by_author_year)
    body_md = INLINE_CITE_RE.sub(repl_inline, body_md)
    
    # 2. Prose cites: Surname (2024), Surname et al. (2024), de Torres et al. (2018)
    def repl_prose(m):
        full = m.group(1)
        year = int(m.group(2))
        # Extract surname: skip lowercase particles like "de", "van"
        # The first word that's capitalized is the surname.
        tokens = re.findall(r"\S+", full)
        surname = None
        for tok in tokens:
            if re.match(r"^[A-ZÁÉÍÓÚÑÀÈÌÒÙ]", tok):
                surname = re.match(SURNAME, tok)
                surname = surname.group() if surname else None
                break
        if not surname:
            return full
        ids = by_author_year.get((surname.lower(), year), [])
        if ids:
            return f"[{full}](#{ids[0]})"
        return full
    body_md = PROSE_CITE_RE.sub(repl_prose, body_md)
    
    # 3. & 4. Numeric cites — only safe to link in main body (skip author affiliations).
    # We define "front matter" as roughly the first 2000 chars (covers title block,
    # author list, abstract). Bracket cites linked everywhere AFTER that.
    # If there's an Abstract/Introduction heading sooner, use that as boundary.
    body_start_match = re.search(
        r"(?im)^#{1,3}\s*\**\s*(?:\d+(?:\.\d+)*\.?\s*)?\**\s*(introduction|background|overview|materials\s+and\s+methods|abstract)\b",
        body_md[:8000],   # only look in first 8K
    )
    if body_start_match:
        body_start = body_start_match.start()
    else:
        body_start = min(2000, len(body_md))
    
    front_matter = body_md[:body_start]
    main_body = body_md[body_start:]
    
    def _try_link_pattern(text, pattern):
        """Apply numeric link substitution if the pattern produces sensible matches."""
        all_nums = []
        for m in pattern.finditer(text):
            nums = _expand_numeric_range(m.group(1))
            all_nums.extend(nums)
        if len(all_nums) < 3:
            return text
        # 70% of numbers must be within reference range
        in_range = sum(1 for n in all_nums if n <= len(references))
        if in_range / len(all_nums) < 0.7:
            return text
        
        def repl(m):
            spec = m.group(1)
            nums = _expand_numeric_range(spec)
            if not nums:
                return m.group(0)
            # All numbers in range? Then link.
            all_in = all(by_number.get(n) for n in nums)
            if not all_in:
                return m.group(0)  # leave unmodified to be safe
            linked = []
            for n in nums:
                rid = by_number.get(n)
                if rid:
                    linked.append(f"[{n}](#{rid})")
                else:
                    linked.append(str(n))
            # Preserve original bracket type
            opener = m.group(0)[0]
            closer = m.group(0)[-1]
            return opener + ", ".join(linked) + closer
        
        return pattern.sub(repl, text)
    
    # Bracket-style first
    main_body = _try_link_pattern(main_body, BRACKET_CITE_RE)
    # Then paren-style (be careful: only if bracket didn't already pick up most of the cites)
    main_body = _try_link_pattern(main_body, PAREN_NUM_CITE_RE)
    
    body_md = front_matter + main_body
    # Restore image refs that were protected from citation linking
    def _restore(m):
        idx = int(m.group(1))
        return _placeholders[idx] if 0 <= idx < len(_placeholders) else m.group(0)
    body_md = re.sub(r"\u2063IMG(\d+)\u2063", _restore, body_md)
    return body_md


def render_references_section(references, ref_md=None, page_marker_re=None):
    """Render the bibliography as a markdown list.

    If `ref_md` (the original references markdown, possibly containing
    PDF page-boundary tokens) and `page_marker_re` are supplied, page
    markers are interleaved between the rendered reference entries at
    the positions where they fell in the raw ref_md.
    """
    if not references:
        return ""
    out = ["\n\n---\n\n## References\n"]

    # If we have ref_md + page-marker pattern, compute which page each
    # reference belongs to by scanning ref_md and tracking the
    # most-recently-seen page marker before each ref's first-author /
    # year signature.
    page_at_ref = {}
    last_marker = None
    if ref_md and page_marker_re:
        cursor = 0
        # Walk through ref_md, tracking page markers as we go
        all_tokens = []
        for m in page_marker_re.finditer(ref_md):
            all_tokens.append(("marker", m.start(), m.group(0)))
        # For each reference, try to find where its text appears in ref_md
        for r in references:
            # Use first 60 distinctive chars of the reference text as a probe
            probe = re.sub(r"\s+", " ", r["text"][:60]).strip()
            if not probe:
                continue
            # Normalise ref_md for matching too
            idx = ref_md.find(probe[:30])
            if idx < 0:
                continue
            # Find the most-recent marker before this position
            current_marker = None
            for tag, pos, txt in all_tokens:
                if pos < idx:
                    current_marker = txt
                else:
                    break
            page_at_ref[r["id"]] = current_marker

    emitted_markers = set()
    for r in references:
        # Emit a page-marker line just before the reference if the page
        # has changed since the previous reference.
        marker = page_at_ref.get(r["id"])
        if marker and marker not in emitted_markers:
            emitted_markers.add(marker)
            out.append("")
            out.append(marker)
            out.append("")
        anchor = f"<a id=\"{r['id']}\"></a>"
        text = r['text']
        if r.get('doi') and 'doi.org' not in text:
            doi_link = f"https://doi.org/{r['doi']}"
            text = text + f" [doi:{r['doi']}]({doi_link})"
        out.append(f"- {anchor}{text}")
    # Any page markers that were in ref_md but we couldn't tie to a
    # specific reference get emitted at the very end so the page count
    # still adds up. This typically happens for very-long bibliographies
    # (OECD 689-page book) where some refs straddle multiple pages.
    if ref_md and page_marker_re:
        all_in_ref_md = set(m.group(0) for m in page_marker_re.finditer(ref_md))
        orphans = sorted(all_in_ref_md - emitted_markers,
                         key=lambda s: int(page_marker_re.search(s).group(1)))
        if orphans:
            out.append("")
            for o in orphans:
                out.append(o)
                out.append("")
    return "\n".join(out) + "\n"
