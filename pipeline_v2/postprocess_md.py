"""
Markdown post-processing for pymupdf4llm output.

Fixes:
  1. BR-stacked tables (cells with embedded line-break tags) -> proper multi-row tables
  2. Title extraction (real title vs journal banner)
  3. Page header/footer deduplication (repeating text on each page)
  4. Soft-hyphen cleanup
  5. Collapse excess blank lines
  6. Author-line footnote-marker cleanup
"""
import re
from collections import Counter


SOFT_HYPHEN = "\u00ad"


def clean_soft_hyphens(text):
    return (text or "").replace(SOFT_HYPHEN, "")


# Unicode normalization: many PDFs use exotic hyphens/dashes/spaces. Normalize
# to ASCII equivalents so downstream regex works predictably.
UNICODE_NORMALIZE = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen  
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash (use ASCII for citation parsing; em dash kept as is)
    "\u2212": "-",  # minus sign
    "\u00a0": " ",  # non-breaking space
    "\u202f": " ",  # narrow no-break space
    "\u2009": " ",  # thin space
    "\u2007": " ",  # figure space
    "\u200b": "",   # zero-width space
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote (apostrophe)
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    # Presentation-form ligatures: expand back to plain letters so they
    # round-trip cleanly in markdown and search.
    "\ufb00": "ff",  # ﬀ
    "\ufb01": "fi",  # ﬁ
    "\ufb02": "fl",  # ﬂ
    "\ufb03": "ffi", # ﬃ
    "\ufb04": "ffl", # ﬄ
    "\ufb05": "st",  # ﬅ
    "\ufb06": "st",  # ﬆ
}


def normalize_unicode_punct(text):
    """Normalize exotic Unicode punctuation to ASCII equivalents."""
    if not text:
        return text
    for src, dst in UNICODE_NORMALIZE.items():
        text = text.replace(src, dst)
    text = fix_floating_diacritics(text)
    return text


# Floating-spacing-diacritic → combining-diacritic mapping. pymupdf sometimes
# emits diacritics as their spacing variants (e.g. ¨ U+00A8) placed BEFORE
# (or sometimes AFTER) the base vowel, so 'Böhm' is rendered as 'B¨ohm' or
# 'Bohm¨'. We recompose them into proper precomposed characters.
_COMBINING_DIACRITIC = {
    "\u00a8": "\u0308",  # ¨ → combining diaeresis
    "\u02d8": "\u0306",  # ˘ → combining breve
    "\u02d9": "\u0307",  # ˙ → combining dot above
    "\u02dc": "\u0303",  # ˜ → combining tilde
    "\u00b4": "\u0301",  # ´ → combining acute
    "\u0060": "\u0300",  # ` → combining grave (risky in code/quotes — handled below)
    "\u02c6": "\u0302",  # ˆ → combining circumflex
}
_VOWELS = set("aeiouyAEIOUY")

# Case 1: diacritic placed BEFORE the vowel — '¨a' → 'ä'
_DIACRITIC_BEFORE_RE = re.compile(
    r"([\u00a8\u02d8\u02d9\u02dc\u00b4\u02c6])([aeiouyAEIOUY])"
)
# Case 2: diacritic at the END of a word (typically after consonant) — 'Bohm¨' → 'Böhm'.
# We re-place it on the rightmost vowel of the word.
_DIACRITIC_TRAILING_RE = re.compile(
    r"([A-Za-zÀ-ÿĀ-ž]{2,})([\u00a8\u02d8\u02d9\u02dc\u02c6])(?![a-zA-Z])"
)


def fix_floating_diacritics(text):
    if not text:
        return text
    import unicodedata as _ud

    def _before(m):
        return _ud.normalize("NFC", m.group(2) + _COMBINING_DIACRITIC[m.group(1)])

    def _trailing(m):
        word, diac = m.group(1), m.group(2)
        for i in range(len(word) - 1, -1, -1):
            if word[i] in _VOWELS:
                return (
                    word[:i]
                    + _ud.normalize("NFC", word[i] + _COMBINING_DIACRITIC[diac])
                    + word[i + 1 :]
                )
        return word + diac

    text = _DIACRITIC_BEFORE_RE.sub(_before, text)
    text = _DIACRITIC_TRAILING_RE.sub(_trailing, text)
    # Known broken German/Spanish/French words from the corpus. The PDF text-
    # extractor sometimes drops the diacritic entirely on the base vowel and
    # emits a separate orphan diaeresis we cannot reliably re-attach.
    # We replace these explicit forms with the correct precomposed form.
    text = _apply_known_diacritic_words(text)
    # Any remaining orphan ¨/˘/˜ etc. that aren't adjacent to a vowel are
    # impossible to place back deterministically — they appear as stray
    # symbols in the text. Delete them, including any single space we left
    # on either side so '(Königslöw et al., 2021¨ )' → '(Königslöw et al., 2021)'.
    text = re.sub(r" ?[\u00a8\u02d8\u02d9\u02dc\u02c6] ?(?![a-zA-Z])", "", text)
    # If deletion left a stray space at the very end of a parenthesised
    # citation like '(Smith, 2021 )', squeeze it.
    text = re.sub(r"(\d{4}[a-z]?) \)", r"\1)", text)
    return text


# Hard-coded restorations for the most common German/Spanish/French names that
# show up across our agricultural-research corpus and lose their diacritic when
# pymupdf extracts them. Keep this list small and surname-only to avoid
# accidentally clobbering legitimate English words.
_DIACRITIC_KNOWN_WORDS = {
    "Schodl": "Schödl",
    "Konigslow": "Königslöw",
    "Konïgsl ow": "Königslöw",
    "Konïgslow": "Königslöw",
    "Universitat": "Universität",
    "Universitatsmedizin": "Universitätsmedizin",
    "Bathe": "Bäthe",
    "Bohm": "Böhm",
    "Hohenheim": "Hohenheim",  # not actually broken, kept for safety
    "Ernahrung": "Ernährung",
    "Bundesamt fur": "Bundesamt für",
    "fur Kartographie": "für Kartographie",
    "Geodasie": "Geodäsie",
    "Schaffer": "Schäffer",
    "Schafer": "Schäfer",
    "Kohler": "Köhler",
    "Mockel": "Möckel",
    "Muller": "Müller",
    "Danhardt": "Dänhardt",
    "Batary": "Batáry",
    "Baden-Bohm": "Baden-Böhm",
    "baden-bohm": "baden-böhm",
    "Pollath": "Polláth",
    "Tscharntke": "Tscharntke",  # no diacritic, kept for matching surrounding context
}

_DIACRITIC_KNOWN_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _DIACRITIC_KNOWN_WORDS) + r")\b"
)


def _apply_known_diacritic_words(text):
    return _DIACRITIC_KNOWN_RE.sub(lambda m: _DIACRITIC_KNOWN_WORDS[m.group(1)], text)


# Hard-coded fixes for `fl`/`fi`/`ffi`/`ffl` ligature dropouts: pymupdf4llm
# sometimes drops the 'l' or 'i' entirely when it can't render the ligature
# glyph, producing words like 'fower' (flower), 'signifcant' (significant),
# 'feld' (field). We only list combinations that aren't otherwise legal
# English words (so we never destroy real text like "fee" or "fight").
LIG_DROP_FIXES = {
    "fower": "flower", "fowers": "flowers", "fowering": "flowering",
    "fowered": "flowered", "Fower": "Flower", "Fowering": "Flowering",
    "Fowers": "Flowers", "wildfower": "wildflower",
    "Wildfower": "Wildflower", "wildfowers": "wildflowers",
    "Wildfowers": "Wildflowers", "sunfower": "sunflower",
    "sunfowers": "sunflowers", "mayfower": "mayflower",
    "fora": "flora", "foral": "floral", "forist": "florist",
    "fuid": "fluid", "fuids": "fluids", "fush": "flush",
    "fux": "flux", "fuxes": "fluxes",
    "fexible": "flexible", "fexibility": "flexibility",
    "fexed": "flexed", "fexing": "flexing", "fexion": "flexion",
    "fourishing": "flourishing", "fourish": "flourish",
    "fourished": "flourished",
    "overfow": "overflow", "underfow": "underflow",
    "signifcant": "significant", "signifcantly": "significantly",
    "signifcance": "significance", "insignifcant": "insignificant",
    "Signifcant": "Significant", "Signifcantly": "Significantly",
    "Signifcance": "Significance",
    "feld": "field", "felds": "fields", "feldwork": "fieldwork",
    "Field": "Field",  # noop for safety
    "frst": "first", "Frst": "First",
    "fnd": "find", "fnds": "finds",
    "fnal": "final", "fnally": "finally",
    "fnding": "finding", "fndings": "findings",
    "defne": "define", "defned": "defined",
    "defnition": "definition", "defnitions": "definitions",
    "defnitely": "definitely",
    "classifcation": "classification",
    "classifcations": "classifications",
    "identifcation": "identification",
    "identifcations": "identifications",
    "modifcation": "modification",
    "modifcations": "modifications",
    "verifcation": "verification", "unifcation": "unification",
    "magnifcation": "magnification",
    "simplifcation": "simplification",
    "amplifcation": "amplification",
    "specifc": "specific", "specifcs": "specifics",
    "specifcally": "specifically",
    "specifcation": "specification",
    "specifcations": "specifications",
    "scientifc": "scientific", "Scientifc": "Scientific",
    "pacifc": "pacific", "Pacifc": "Pacific",
    "terrifc": "terrific", "horrifc": "horrific",
    "efcient": "efficient", "efciency": "efficiency",
    "efcacy": "efficacy", "inefcient": "inefficient",
    "sufcient": "sufficient", "sufciently": "sufficiently",
    "difcult": "difficult", "difculty": "difficulty",
    "difculties": "difficulties",
    "profcient": "proficient", "profciency": "proficiency",
    "profle": "profile", "profles": "profiles",
    "benefcial": "beneficial", "benefciary": "beneficiary",
    "sacrifce": "sacrifice", "sacrifced": "sacrificed",
    "orifce": "orifice",
}

_LIG_DROP_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in LIG_DROP_FIXES) + r")\b"
)


def fix_dropped_ligatures(text):
    if not text:
        return text
    return _LIG_DROP_RE.sub(lambda m: LIG_DROP_FIXES[m.group(1)], text)


# Patterns for ligature fixes. The trick: PDFs lose ff/fi/fl/ffi ligatures
# and split words at that position:
#   "Effect"    -> "E ff ect"          (E + ff + ect)
#   "Effective" -> "E ff ective"       (E + ff + ective)
#   "file"      -> "fi le"             (fi + le)
#   "flow"      -> "fl ow"             (fl + ow)
#   "efficient" -> "e ffi cient"       (e + ffi + cient)
#
# Detection: the "fragment" word is short (1-3 chars) and consists of just ff/fi/fl/ffi/ffl.
# When we see that as a standalone word with both neighbors being word fragments,
# we can join them.

# Conservative: only fix when LEFT fragment is 1 char (very likely a broken word)
# OR when the produced word matches a common English word pattern.
# Plus: handle the "starts-with-ligature" case (fl ow, fi le) at start of word.

# Pattern 1: "X ligature yyy" where X is exactly 1 letter (very likely broken)
LIG_1CHAR_RE = re.compile(
    r"(?<![a-zA-Z])"
    r"([a-zA-Z])"               # exactly 1 letter prefix
    r"\s+(ff|fi|fl|ffi|ffl)"
    r"\s+([a-z][a-zA-Z]*)"
    r"(?![a-zA-Z])"
)

# Pattern 2: "ligature xxx" at start of word (no prefix). Examples: fl ow, fi le
LIG_START_RE = re.compile(
    r"(?<![a-zA-Z])"
    r"(ff|fi|fl|ffi|ffl)"
    r"\s+([a-z][a-zA-Z]*)"
    r"(?![a-zA-Z])"
)


def _lig_1char_repl(m):
    return m.group(1) + m.group(2) + m.group(3)


def _lig_start_repl(m):
    return m.group(1) + m.group(2)


HYPHEN_SPACE_RE = re.compile(r"(\w)-\s+(\w)")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:])")


PICTURE_PLACEHOLDER_RE = re.compile(
    r"\*?\*?==>\s*picture\s*\[[^\]]*\]\s*intentionally\s+omitted\s*<==\*?\*?",
    re.IGNORECASE,
)

# Pymupdf4llm sometimes wraps fragments of a word in bold markers, like
# "**The E** ff **ect of...**" for "The Effect of..." where "E" is a drop cap.
# Pattern: **...X** ligature **Y...** where X is a single letter and Y is lowercase.
BOLD_LIG_RE = re.compile(
    r"\*\*([^*]*?[A-Z])\*\*\s+(ff|fi|fl|ffi|ffl)\s+\*\*([a-z][^*]*?)\*\*"
)


def _bold_lig_repl(m):
    return "**" + m.group(1) + m.group(2) + m.group(3) + "**"


def fix_pdf_artifacts(text):
    """Fix common PDF extraction artifacts: broken ligatures, split hyphens, placeholders."""
    if not text:
        return text
    # Remove image-placeholder noise
    text = PICTURE_PLACEHOLDER_RE.sub("", text)
    # Fix bold-wrapped ligature splits: "**E** ff **ect**" -> "**Effect**"
    text = BOLD_LIG_RE.sub(_bold_lig_repl, text)
    # Apply 1-char prefix first (conservative), then start-of-word.
    # Iterate to handle compound cases.
    for _ in range(5):
        new = LIG_1CHAR_RE.sub(_lig_1char_repl, text)
        new = LIG_START_RE.sub(_lig_start_repl, new)
        if new == text:
            break
        text = new
    # Fix words where pymupdf4llm DROPPED the 'l'/'i' from fl/fi/ffi/ffl
    # ligatures entirely (e.g. 'fower' → 'flower', 'signifcant' → 'significant').
    text = fix_dropped_ligatures(text)
    text = HYPHEN_SPACE_RE.sub(r"\1-\2", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    # Fix italic underscore touching the previous word: 'chose_Bombus' → 'chose _Bombus'
    text = re.sub(r"(?<=[A-Za-z])_(?=[A-ZÁÉÍÓÚÑÀÈÌÒÙ][a-zA-Z])", " _", text)
    # Merge adjacent italic runs that pymupdf split mid-word, e.g. when a
    # diacritic intervened: '_F. Baden-B_ _ohm et al._' → '_F. Baden-Bohm et al._'.
    # Be permissive on the second run length (running headers often contain
    # the journal name) but conservative on the first run: it must end in a
    # letter and the second must begin with a lowercase letter (i.e. mid-word
    # split) to avoid joining unrelated italic phrases.
    text = re.sub(
        r"_([A-Za-zÀ-ÿĀ-ž][^_\n]{0,60}[A-Za-zÀ-ÿĀ-ž])_\s+_([a-zà-ÿā-ž][^_\n]{0,300})_",
        r"_\1\2_",
        text,
    )
    # Collapse runs of 3+ spaces into 2 (PDF column gaps)
    text = re.sub(r" {3,}", "  ", text)
    # Apply known-diacritic-word fixer AGAIN — joining italics may have produced
    # a name that now matches (e.g. 'Baden-Bohm' → 'Baden-Böhm').
    text = _apply_known_diacritic_words(text)
    # Strip stray spaces inside parentheses around italics: '( _Bombus_ )' → '(_Bombus_)'
    text = re.sub(r"\(\s+(?=[_*])", "(", text)
    text = re.sub(r"(?<=[_*])\s+\)", ")", text)
    # Missing space after year in citations: '2019for organic' → '2019 for organic'.
    # Note: don't allow the optional year-suffix letter ([a-z]?) here because
    # otherwise '2019for' eagerly matches '2019f' + 'or' → '2019f or'.
    text = re.sub(r"(\b(?:19|20)\d{2})([a-z]{2,})", r"\1 \2", text)
    # Missing space after comma before year: 'Olsson et al.,2015' → 'Olsson et al., 2015'
    text = re.sub(r"([A-Za-z.]),((?:19|20)\d{2})", r"\1, \2", text)
    # pymupdf4llm wraps the degree sign as a superscript bracket: '52[◦] 37′' → '52° 37′'.
    text = text.replace("[◦]", "°").replace("[°]", "°")
    # Strip stray spaces inside parentheses around capitalised words used as labels:
    # '( Bombus terrestris )' → '(Bombus terrestris)' (only when the inner content
    # is short and starts with a capital — avoid touching real prose).
    text = re.sub(r"\(\s+([A-Z][A-Za-z]{1,30}(?:\s+[a-zA-Z]{1,30}){0,3})\s+\)",
                  r"(\1)", text)
    return text


def collapse_blank_lines(text):
    return re.sub(r"\n{3,}", "\n\n", text)


# ---------------------------------------------------------------------------
# 1. BR-stacked table reshaping
# ---------------------------------------------------------------------------

BR_RE = re.compile(r"\s*<br\s*/?\s*>\s*")


def _split_br(cell):
    return [p.strip() for p in BR_RE.split(cell.strip())]


def _parse_md_row(line):
    s = line.strip()
    if not s.startswith("|"):
        return None
    if s.endswith("|"):
        s = s[1:-1]
    else:
        s = s[1:]
    cells = s.split("|")
    return [c.strip() for c in cells]


def _format_md_row(cells):
    safe = [(c or "").replace("|", "\\|").replace("\n", " ") for c in cells]
    return "| " + " | ".join(safe) + " |"


def _is_separator_row(cells):
    return any(cells) and all(
        re.fullmatch(r"-{2,}|:?-+:?", c.strip()) for c in cells if c.strip()
    )


def reshape_br_table_block(lines):
    if len(lines) < 2:
        return lines

    rows = []
    sep_idx = None
    for i, line in enumerate(lines):
        cells = _parse_md_row(line)
        if cells is None:
            return lines
        rows.append(cells)
        if _is_separator_row(cells) and sep_idx is None:
            sep_idx = i

    if sep_idx is None:
        sep_idx = 0

    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    header_rows = rows[:sep_idx]
    body_rows = rows[sep_idx + 1:] if sep_idx + 1 <= len(rows) else []

    def expand_row(row):
        split_cells = [_split_br(c) for c in row]
        max_n = max((len(s) for s in split_cells), default=1)
        split_cells = [s + [""] * (max_n - len(s)) for s in split_cells]
        return [list(t) for t in zip(*split_cells)]

    new_headers = []
    for r in header_rows:
        new_headers.extend(expand_row(r))
    new_body = []
    for r in body_rows:
        new_body.extend(expand_row(r))

    if not new_headers:
        new_headers = [[""] * n_cols]

    out = []
    for hr in new_headers:
        out.append(_format_md_row(hr))
    out.append(_format_md_row(["---"] * n_cols))
    for br in new_body:
        out.append(_format_md_row(br))
    return out


def reshape_article_info_abstract_table(markdown):
    """
    Elsevier-style journal pages have a 2-column 'A R T I C L E I N F O |
    A B S T R A C T' table on page 1. pymupdf4llm emits it as one giant
    table cell with <br>-separated lines:

        |A R T I C L E I N F O <br>_Keywords:_<br>kw1<br>kw2|A B S T R A C T|
        |---|---|
        ||Individual biodiversity measures…<br>diversity and abundance…|

    We detect this and rewrite it as a clean Abstract section so
    readers see the actual abstract text and the keywords list.
    """
    if not markdown or "A R T I C L E" not in markdown:
        return markdown
    # Find the first line that begins with a pipe-cell containing 'A R T I C L E'
    lines = markdown.split("\n")
    out = []
    i = 0
    consumed_to = -1
    while i < len(lines):
        line = lines[i]
        # Start: line is a pipe-table row whose first cell contains "A R T I C L E"
        if (
            line.lstrip().startswith("|")
            and "A R T I C L E" in line.upper().replace(".", "")
        ):
            # Collect the whole table block (consecutive pipe lines, possibly with
            # blank/separator rows).
            block = []
            j = i
            while j < len(lines) and (
                lines[j].lstrip().startswith("|") or lines[j].strip() == ""
            ):
                # Stop if blank line not immediately followed by another pipe
                if lines[j].strip() == "":
                    if j + 1 < len(lines) and lines[j + 1].lstrip().startswith("|"):
                        block.append(lines[j])
                        j += 1
                        continue
                    break
                block.append(lines[j])
                j += 1
            # Flatten block to one big string and split on <br>
            joined = " \n ".join(block)
            # Pull out all "cells" delimited by '|', strip the separator rows
            cells = []
            for ln in block:
                s = ln.strip()
                if not s.startswith("|"):
                    continue
                if s.endswith("|"):
                    s = s[1:-1]
                else:
                    s = s[1:]
                parts = [c.strip() for c in s.split("|")]
                # Skip separator rows
                if all(re.fullmatch(r":?-+:?", p) for p in parts if p):
                    continue
                cells.extend(parts)
            # Within each cell, split on <br>
            fragments = []
            br = re.compile(r"\s*<br\s*/?\s*>\s*")
            for c in cells:
                for f in br.split(c):
                    f = f.strip()
                    if f:
                        fragments.append(f)
            # Now classify: keywords list and abstract paragraph(s).
            keywords = []
            abstract_lines = []
            in_keywords = False
            in_abstract = False
            for f in fragments:
                f_clean = re.sub(r"[*_]", "", f).strip()
                # Headings
                if re.fullmatch(r"A\s*R\s*T\s*I\s*C\s*L\s*E\s*I\s*N\s*F\s*O\.?",
                                f_clean, re.IGNORECASE):
                    in_keywords = False
                    in_abstract = False
                    continue
                if re.fullmatch(r"A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\.?",
                                f_clean, re.IGNORECASE):
                    in_keywords = False
                    in_abstract = True
                    continue
                if re.match(r"^_?Keywords?_?\s*:?\s*$", f, re.IGNORECASE):
                    in_keywords = True
                    in_abstract = False
                    continue
                # If this fragment starts with 'Keywords:' inline, split it
                m_kw = re.match(r"^_?Keywords?_?\s*:\s*(.*)$", f, re.IGNORECASE)
                if m_kw:
                    in_keywords = True
                    in_abstract = False
                    rest = m_kw.group(1).strip()
                    if rest:
                        keywords.append(rest)
                    continue
                if in_keywords:
                    # Short fragments are keywords; once we hit a long sentence
                    # the keywords block is over and we drop into "other".
                    if len(f.split()) <= 8:
                        keywords.append(f)
                        continue
                    in_keywords = False
                if in_abstract:
                    abstract_lines.append(f)
                    continue
                # Fragments before we hit ABSTRACT — likely keywords too
                if not abstract_lines and len(f.split()) <= 8:
                    keywords.append(f)
            # Only rewrite if we got at least an abstract paragraph
            if abstract_lines:
                # Join abstract: most pymupdf4llm <br> splits are mid-sentence
                # line wraps. We join with a space, then collapse multi-spaces.
                abstract = " ".join(abstract_lines)
                abstract = re.sub(r"\s+", " ", abstract).strip()
                # The leading "fower"/"signifcant" will be fixed by the ligature
                # fixer that runs in fix_pdf_artifacts.
                out.append("## Abstract\n")
                out.append(abstract + "\n")
                if keywords:
                    # De-duplicate while preserving order
                    seen = set()
                    kw_clean = []
                    for k in keywords:
                        kk = re.sub(r"[*_]", "", k).strip().rstrip(",;")
                        if kk and kk.lower() not in seen:
                            kw_clean.append(kk)
                            seen.add(kk.lower())
                    if kw_clean:
                        out.append("**Keywords:** " + ", ".join(kw_clean) + "\n")
                out.append("")
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def unwrap_reference_tables(markdown):
    """
    Some publishers (esp. MDPI when text is rendered in columns near page bottom)
    cause pymupdf4llm to detect the reference list as a 2-column markdown table:
        |27.|Author, X.Y. Title. _Journal_ **2024**, _15_, 802-817.|
        |---|---|
        ||continuation text|
        |28.|Next author...|
    We unwrap these back to plain numbered lines so reference parsing works
    and the (subsequent) junk-table filter doesn't drop them.
    """
    lines = markdown.split("\n")
    out_lines = []
    i = 0
    NUM_PREFIX = re.compile(r"^\s*\|\s*\d{1,3}\.\s*\|")
    SEP_ROW = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")
    EMPTY_FIRST = re.compile(r"^\s*\|\s*\|")
    while i < len(lines):
        line = lines[i]
        # Detect start of a table block whose first non-separator data row starts with N. |
        if line.lstrip().startswith("|"):
            block = []
            j = i
            while j < len(lines) and (
                lines[j].lstrip().startswith("|") or lines[j].strip() == ""
            ):
                # Stop at blank line that breaks the block
                if lines[j].strip() == "":
                    # peek: if next line still part of table, continue; else stop
                    if j + 1 < len(lines) and lines[j + 1].lstrip().startswith("|"):
                        block.append(lines[j])
                        j += 1
                        continue
                    break
                block.append(lines[j])
                j += 1
            # Decide: is this a reference table?
            num_rows = sum(1 for l in block if NUM_PREFIX.search(l))
            if num_rows >= 3:
                # Unwrap
                current = None
                for bl in block:
                    s = bl.strip()
                    if not s or SEP_ROW.match(s):
                        continue
                    # Split cells
                    if s.startswith("|"):
                        s = s[1:]
                    if s.endswith("|"):
                        s = s[:-1]
                    cells = [c.strip() for c in s.split("|")]
                    m = re.match(r"^(\d{1,3})\.$", cells[0]) if cells else None
                    if m:
                        if current is not None:
                            out_lines.append(current)
                        num = m.group(1)
                        rest = " ".join(c for c in cells[1:] if c)
                        current = f"{num}. {rest}"
                    else:
                        # Continuation: append non-empty cells
                        cont = " ".join(c for c in cells if c)
                        if cont and current is not None:
                            current = current.rstrip() + " " + cont
                if current is not None:
                    out_lines.append(current)
                out_lines.append("")
                i = j
                continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


def find_and_reshape_br_tables(markdown):
    out_lines = []
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            has_br = any("<br" in l for l in block)
            if has_br:
                out_lines.extend(reshape_br_table_block(block))
            else:
                out_lines.extend(block)
            continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# 2. Page header/footer deduplication
# ---------------------------------------------------------------------------

def dedupe_page_headers_footers(per_page_texts):
    """Remove lines that repeat on most pages (running headers/footers, page numbers)."""
    if len(per_page_texts) < 4:
        return per_page_texts

    line_pages = Counter()
    for t in per_page_texts:
        page_lines = [l.strip() for l in t.split("\n") if l.strip()]
        # Increase upper bound to 200 chars to catch long running titles
        candidates = [l for l in page_lines if 2 <= len(l) <= 200]
        for l in set(candidates):
            line_pages[l] += 1

    # Lower threshold to 30% so we catch running titles that don't appear on EVERY page
    threshold = max(3, int(0.3 * len(per_page_texts)))
    repeats = {l for l, c in line_pages.items() if c >= threshold}

    # Also detect "fuzzy" repeats: lines that are almost-identical but differ in
    # truncation (e.g. running titles end with ellipsis on some pages).
    # Build a set of "stems" (first 60 chars) that appear repeatedly.
    stem_counts = Counter()
    for t in per_page_texts:
        for l in t.split("\n"):
            s = l.strip()
            if 30 <= len(s) <= 200:
                stem_counts[s[:60]] += 1
    repeating_stems = {st for st, c in stem_counts.items() if c >= threshold}

    page_num_re = re.compile(r"^[|\s]*(\d+\s*(?:of|/)\s*\d+|\d+)\s*[|\s]*$")
    # Vol/page banner artifacts (Springer)
    banner_artifact_re = re.compile(r"^(Vol\.?:?\(?[\d]+\)?|\d+\s+\d+\s*$)")

    cleaned = []
    for t in per_page_texts:
        kept = []
        for line in t.split("\n"):
            s = line.strip()
            if s in repeats:
                continue
            if page_num_re.match(s):
                continue
            if banner_artifact_re.match(s):
                continue
            # Check fuzzy repeat
            if len(s) >= 30 and s[:60] in repeating_stems:
                continue
            kept.append(line)
        cleaned.append("\n".join(kept))
    return cleaned


# ---------------------------------------------------------------------------
# 3. Title extraction
# ---------------------------------------------------------------------------

BANNER_RE = re.compile(
    r"(received:|accepted:|published:|^doi:|copyright|contents lists|"
    r"open access|journal homepage|sciencedirect|elsevier|wiley|springer|"
    r"frontiers|mdpi|cite this|article number|how to cite|"
    r"available\s+online|peer\s*review|"
    # ISSN, Volume markers
    r"\b\d{4}-\d{4}\b|"
    r"\bvol\.?\s*\d|issue\s+\d|\d+\s*\(\d+\):)", re.IGNORECASE
)

# Strict pattern: matches if a STRING IS A JOURNAL BANNER (as opposed to just
# containing a banner word). Used by title extraction.
JOURNAL_TITLE_RE = re.compile(
    r"^("
    r"agriculture,?\s*ecosystems(?:\s+and\s+environment)?|"
    r"soil\s*(?:&|and)\s*tillage(?:\s+research)?|"
    r"land\s+use\s+policy|"
    r"environmental\s+research(?:\s+letters)?|"
    r"nature|science|cell|"
    r"applied\s+soil(?:\s+ecology)?|"
    r"geoderma|"
    r"soil\s+(?:science|biology|use)|"
    r"crop\s+(?:protection|science)|"
    r"renewable\s+agriculture|"
    r"cambridge\.org|"
    r"sustainability|"
    r"agronomy(?:\s+for\s+sustainable\s+development)?"
    r")\b\s*\.?\s*$", re.IGNORECASE
)

GENERIC_TITLE_RE = re.compile(
    r"^(article|original research|review article|research article|"
    r"practice and policy|opinion|short communication|editorial|"
    r"perspective|brief report|letter|"
    r"funding\s+information|correspondence|abstract|resumen|"
    r"keywords?|acknowledgments?|conflicts?\s+of\s+interest|"
    r"author\s+contributions?|data\s+availability|appendix|"
    r"received|accepted|published|conclusion|conclusions|"
    r"introduction|methods?|results?|discussion|"
    r"open access|original article|table of contents)$",
    re.IGNORECASE,
)


def extract_title(markdown, max_lookahead=80):
    lines = [l for l in markdown.split("\n")[:max_lookahead] if l.strip()]
    
    # Special case: "Cite this article: ... (year). Title. JournalName ..."
    cite_match = re.search(
        r"Cite\s+this\s+article:?[^.]{0,400}?\(\d{4}\)\.\s+([^.]+?)\.\s*[A-Z][^.]+?\d",
        markdown[:3000], re.DOTALL,
    )
    if cite_match:
        candidate = re.sub(r"\s+", " ", cite_match.group(1)).strip()
        if 15 < len(candidate) < 300:
            return candidate
    
    # Special case: Corrigendum / Erratum titles
    err_match = re.search(
        r"(?:Corrigendum|Erratum)\s+to\s+[\"\u201c]([^\"\u201d]+)[\"\u201d]",
        markdown[:2000],
    )
    if err_match:
        title = err_match.group(1).strip()
        return f"Corrigendum: {title}"
    
    # Walk in order, return the FIRST title-like heading or bold paragraph
    def is_good_title(t):
        t = t.strip()
        # Normalize spaced-out letters (e.g. "R E S E A R C H  A R T I C L E" -> "research article")
        # Detect: each "word" is a single letter
        tokens = t.split()
        if len(tokens) >= 4 and all(len(tok) <= 2 for tok in tokens):
            normalized = "".join(tokens).lower()
        else:
            normalized = re.sub(r"\s+", " ", t).lower()
        
        return (
            30 <= len(t) <= 400
            and not BANNER_RE.search(t)
            and not GENERIC_TITLE_RE.match(t)
            and not GENERIC_TITLE_RE.match(normalized)
            and not JOURNAL_TITLE_RE.match(t)
            # Must contain at least 4 words AND most words must be > 2 chars
            and len(tokens) >= 4
            and sum(1 for tok in tokens if len(tok) > 2) >= 4
            # Not just a year/number/marker
            and not re.fullmatch(r"[\d\s\W]+", t)
        )
    
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Markdown heading
        m = re.match(r"^#{1,3}\s+(.+)", s)
        if m:
            title = re.sub(r"\*+|_+", "", m.group(1)).strip()
            if is_good_title(title):
                return title
        # Bold paragraph (whole line wrapped in **...**)
        b = re.match(r"^\*{2}([^*].*?)\*{2}\s*$", s)
        if b:
            title = b.group(1).strip()
            # Clean any internal formatting
            title = re.sub(r"\*+|_+", "", title).strip()
            if is_good_title(title):
                return title
    
    # Fallback: first long non-banner line
    fallback = [s.strip() for s in lines if 30 < len(s.strip()) < 300 and not BANNER_RE.search(s)]
    return fallback[0] if fallback else ""


# ---------------------------------------------------------------------------
# 4. Author-list cleanup
# ---------------------------------------------------------------------------

def clean_author_line(text):
    text = re.sub(r"\[[\d,\*\s\-]+\]", "", text)
    text = re.sub(r"\s*\*\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------

def postprocess_full(markdown):
    text = clean_soft_hyphens(markdown)
    text = find_and_reshape_br_tables(text)
    text = collapse_blank_lines(text)
    return text


def strip_stray_br(text):
    """Remove standalone <br> tags left over from picture-text processing."""
    # Remove <br> followed only by whitespace and newlines
    text = re.sub(r"\s*<br\s*/?\s*>\s*\n", "\n", text)
    # Remove <br> at end of lines
    text = re.sub(r"<br\s*/?\s*>\s*$", "", text, flags=re.MULTILINE)
    return text


# Boilerplate lines that should be stripped from body text wherever they appear.
# These are typical journal banner artifacts that pymupdf4llm bleeds through.
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*-?\s*\*?\s*Corresponding\s+author", re.IGNORECASE),
    re.compile(r"^\s*Corresponding\s+author[\s:]", re.IGNORECASE),
    # E-mail address footnote (Elsevier corresponding-author line)
    re.compile(r"^\s*-?\s*_?E-?mail\s+address(?:es)?:?_?\s+", re.IGNORECASE),
    # Lone fragment like "_F. Baden-B_ ¨" left over from author footer
    re.compile(r"^\s*_[A-Z]\.\s+[A-Z][A-Za-zÀ-ÿĀ-ž\-']+_\s*[\u00a8\u02d8\u02d9\u02dc\u00b4]?\s*$"),
    re.compile(r"^\s*https?://doi\.org/10\.\d", re.IGNORECASE),
    re.compile(r"^\s*Received\s+\d{1,2}\s+\w+\s+\d{4}", re.IGNORECASE),
    re.compile(r"^\s*Received:?\s+\d{1,2}\s+\w+\s+\d{4}", re.IGNORECASE),
    re.compile(r"^\s*\d{4}-\d{4}/", ),  # ISSN line
    re.compile(r"^\s*©\s*\d{4}\s+(?:The\s+)?[Aa]uthors?", ),
    re.compile(r"^\s*This\s+is\s+an\s+open\s+access\s+article", re.IGNORECASE),
    re.compile(r"^\s*Available\s+online\s+\d", re.IGNORECASE),
    re.compile(r"^\s*Published\s+by\s+(Copernicus|Elsevier|Wiley|Springer|MDPI|Frontiers)", re.IGNORECASE),
    re.compile(r"^\s*wileyonlinelibrary\.com", re.IGNORECASE),
    re.compile(r"^\s*www\.(?:mdpi|elsevier|sciencedirect)\.com", re.IGNORECASE),
    re.compile(r"^\s*Vol\.?:?\s*\([\d:]+\)", re.IGNORECASE),  # Springer "Vol.:(0123456789)"
    # Common journal footer like "_Journal Name_ **2020**, _10_, 580"  
    re.compile(r"^\s*_[A-Z][\w\s&]+_\s+\*\*\d{4}\*\*\s*,\s*_\d", ),
    # Page references like "_Soil Use Manage._ 2024;40:e13066"
    re.compile(r"^\s*_[A-Z][\w\s&\.]+_\s+\d{4};?\s*\d", ),
    # Article number lines like "Soil & Tillage Research 244 (2024) 106215"
    re.compile(r"^[A-Z][\w\s&]{3,40}\s+\d+\s*\(\d{4}\)\s+\d", ),
    # MDPI sidebar metadata that bleeds into the body after the Introduction:
    re.compile(r"^\s*Academic\s+Editors?[:\s]", re.IGNORECASE),
    re.compile(r"^\s*Guest\s+Editor[:\s]", re.IGNORECASE),
    re.compile(r"^\s*Handled\s+by\b", re.IGNORECASE),
    re.compile(r"^\s*\*?\*?Publisher['\u2019]s\s+Note:?\*?\*?", re.IGNORECASE),
    re.compile(r"^\s*\*?\*?Copyright:?\*?\*?\s*©", re.IGNORECASE),
    re.compile(r"^\s*\*?\*?Citation:?\*?\*?\s+[A-Z]", re.IGNORECASE),
    re.compile(r"^\s*\*?\*?Funding(?:\s+information)?:?\*?\*?\s+[A-Z]", re.IGNORECASE),
    # Wiley/MDPI license blurb
    re.compile(r"^\s*This\s+article\s+is\s+an\s+open\s+access\s+article", re.IGNORECASE),
    re.compile(r"^\s*Creative\s+Commons\s+Attribution\b", re.IGNORECASE),
    # "Received: …; Accepted: …; Published: …" all-in-one variants
    re.compile(r"^\s*Received[:\s]+\d.*Accepted[:\s]+\d", re.IGNORECASE),
    # MDPI extra metadata
    re.compile(r"^\s*Extended\s+author\s+information", re.IGNORECASE),
    re.compile(r"^\s*Specialty\s+section:", re.IGNORECASE),
    re.compile(r"^\s*These\s+authors\s+(?:contributed|have\s+contributed)", re.IGNORECASE),
    re.compile(r"^\s*\*?Correspondence:?\*?\s+[A-Z]", re.IGNORECASE),
    # Springer/Wiley footer references like:
    #   "**36** Page 2 of 17"   (even pages — issue number then page x of y)
    #   "Page 3 of 17 **36**"   (odd pages — page x of y then issue number)
    re.compile(r"^\s*\*\*\d+\*\*\s+Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s+\*\*\d+\*\*\s*$", re.IGNORECASE),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    # MDPI journal-URL footer like "https://www.mdpi.com/journal/agronomy"
    re.compile(r"^\s*https?://www\.mdpi\.com/journal/", re.IGNORECASE),
    # Wiley DOI URL footer
    re.compile(r"^\s*https?://onlinelibrary\.wiley\.com/", re.IGNORECASE),
]


def strip_boilerplate(text):
    """Remove journal-banner boilerplate lines that bleed into body."""
    lines = text.split("\n")
    kept = []
    for line in lines:
        is_boilerplate = any(p.match(line) for p in BOILERPLATE_PATTERNS)
        if is_boilerplate:
            continue
        kept.append(line)
    return "\n".join(kept)


# Convert pymupdf4llm's [N] superscript notation to Unicode superscripts
# (or just plain notation) BEFORE citation linking.
# Patterns: [1], [2], [3], ... up to a few characters; only digits/+/-/=/().

# Unicode superscript map
SUPERSCRIPT_MAP = str.maketrans({
    "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3", "4": "\u2074",
    "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079",
    "+": "\u207a", "-": "\u207b", "=": "\u207c", "(": "\u207d", ")": "\u207e",
    "\u2212": "\u207b",  # Unicode minus
    "n": "\u207f",
})


def convert_superscripts(text):
    """
    Convert pymupdf4llm's [N] superscript brackets (used for math notation like
    'cm⋅min[−][1]' = cm·min⁻¹) to actual Unicode superscripts.
    
    Detection heuristic: a [bracket] is a superscript (NOT a citation) if:
      - The bracketed content is short (1-3 chars) AND
      - Consists of only digits/+/-/=/n
      - AND it's preceded by a letter, digit, or another superscript bracket
        (i.e., not at the start of a word or after a space)
    
    Citations are typically preceded by a space, period, or comma.
    """
    # Pattern: [N] where N is a digit/sign and the character right before is
    # a non-space (i.e., directly attached to previous text).
    # Also handle Greek letters, underscore (for italic markup like _R_[2]), 
    # and other math symbols as "preceding char"
    SUP_RE = re.compile(r"(?<=[A-Za-z0-9_\)\]\u00b0\u00b5\u0370-\u03ff])\[([+\-=\u2212\d]{1,3})\]")
    
    def repl(m):
        inner = m.group(1)
        # Try to convert to unicode superscripts
        try:
            converted = inner.translate(SUPERSCRIPT_MAP)
            return converted
        except Exception:
            return m.group(0)
    
    # Apply repeatedly to handle multi-bracket cases like [−][1] -> ⁻¹
    for _ in range(3):
        new = SUP_RE.sub(repl, text)
        if new == text:
            break
        text = new
    return text


# ---------------------------------------------------------------------------
# 6. Picture-text block reformatter
# ---------------------------------------------------------------------------

PICTURE_TEXT_FULL_RE = re.compile(
    r"\*\*-{2,} Start of picture text -{2,}\*\*"
    r".*?"
    r"\*\*-{2,} End of picture text -{2,}\*\*",
    re.DOTALL,
)

# For papers where pymupdf4llm only emits the End marker, capture everything from
# the preceding paragraph break (or image ref) up to and including the End marker.
# Important constraint: the content between the prefix and the End marker MUST
# contain <br> tags (typical of OCR-flattened picture text). We also cap the
# inner content to 2_000 chars so we don't accidentally pull in many pages of
# the document when the End marker is far away (e.g. OECD page 1 → page 30).
PICTURE_TEXT_END_ONLY_RE = re.compile(
    r"(?:!\[[^\]]*\]\([^)]+\)|\n\n)"          # preceded by image ref (any alt) or blank line
    r"((?:(?!\*\*-{2,} End of picture text)[\s\S]){0,2000}?<br[\s\S]{0,1500}?)"
    r"\*\*-{2,} End of picture text -{2,}\*\*\s*(?:<br\s*/?>)?",
    re.DOTALL,
)


def _picture_block_to_code(content):
    """Split picture-text content on <br> and render as a code block."""
    rows = re.split(r"\s*<br\s*/?\s*>\s*", content)
    rows = [r.strip() for r in rows if r.strip()]
    # Drop any leftover marker fragments
    rows = [r for r in rows if "picture text" not in r.lower()]
    if not rows:
        return ""
    return "\n```text\n" + "\n".join(rows) + "\n```\n"


def reformat_picture_text_blocks(markdown):
    """
    pymupdf4llm wraps OCR'd table-image content in:
        **----- Start of picture text -----**<br>
        row1<br>row2<br>row3<br>
        **----- End of picture text -----**<br>
    
    Some papers only emit the End marker. We handle both cases.
    """
    # First pass: full blocks (start + end)
    def repl_full(m):
        block = m.group(0)
        inner = re.sub(r".*?Start of picture text -+\*\*", "", block, count=1, flags=re.DOTALL)
        inner = re.sub(r"\*\*-+ End of picture text.*", "", inner, count=1, flags=re.DOTALL)
        return _picture_block_to_code(inner)
    markdown = PICTURE_TEXT_FULL_RE.sub(repl_full, markdown)
    
    # Second pass: orphan End markers — capture content between preceding image/blank
    # line and the End marker.
    def repl_end_only(m):
        prefix_match = m.group(0)
        content = m.group(1)
        # Preserve the prefix (image ref or blank line) but replace the content+End
        prefix = prefix_match[:prefix_match.find(content)] if content else prefix_match
        # Only convert if the captured content actually has <br> tags (looks like OCR output)
        if "<br" not in content:
            return prefix_match  # unchanged
        code = _picture_block_to_code(content)
        return prefix + code
    markdown = PICTURE_TEXT_END_ONLY_RE.sub(repl_end_only, markdown)
    
    return markdown
