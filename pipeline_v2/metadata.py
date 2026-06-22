"""
Extract structured metadata from a paper:
  - title
  - authors (list)
  - year
  - DOI
  - journal (best-effort)
"""
import re
from postprocess_md import extract_title, BANNER_RE


DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)


def extract_metadata(markdown):
    title = extract_title(markdown)
    
    doi = None
    m = DOI_PATTERN.search(markdown[:4000])
    if m:
        doi = m.group(1).rstrip(".,;")
    else:
        m = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", markdown)
        if m:
            candidate = m.group(1).rstrip(".,;")
            if 8 < len(candidate) < 100:
                doi = candidate
    
    # Year: extract a year that is clearly NOT followed by 's' (decade marker like 1960s)
    year = None
    head = markdown[:4000]
    for pat in [
        r"(?:published|copyright)\D{0,20}((?:19|20)\d{2})(?!\d)(?!s)",
        r"\(((?:19|20)\d{2})\)",
        r"\u00a9\s*((?:19|20)\d{2})(?!\d)(?!s)",
        r"\b((?:19|20)\d{2})\b(?!s)",
    ]:
        m = re.search(pat, head, re.IGNORECASE)
        if m:
            try:
                year = int(m.group(1))
                break
            except (ValueError, IndexError):
                continue
    
    # Journal: look for journal-name patterns near top (banner usually first lines).
    # Be strict — must look like a journal heading, not a sentence/funding line.
    journal = None
    for line in markdown.split("\n")[:25]:
        s = line.strip()
        # Strip markdown markers
        s_clean = re.sub(r"[#*_>]+", "", s).strip()
        # Patterns where journal name appears with vol/year/issue notation
        # e.g. "Agriculture, Ecosystems and Environment 356 (2023) 108649"
        # or   "Soil & Tillage Research 244 (2024) 106215"
        # or   "Earth Systems and Environment (2022) 6:29-44"
        m = re.match(
            r"^([A-Z][A-Za-z &,]{4,70}?(?:Journal|Research|Science|Studies|Letters|"
            r"Reports|Review|Ecology|Agronomy|Biology|Soil|Environment|"
            r"Sustainability|Management|Systems|Conservation|Tillage|Use))"
            r"\s+(?:\d+|\(\d{4}\))",
            s_clean,
        )
        if m:
            journal = m.group(1).strip().rstrip(",")
            break
    # Fallback: title-of-section pattern in Frontiers / MDPI papers
    if not journal:
        m = re.search(r"published\s+(?:in|by)\s+([A-Z][A-Za-z &,]+(?:Journal|Research|Science|Letters|Frontiers in [A-Z][a-z]+))", markdown[:3000], re.IGNORECASE)
        if m:
            journal = m.group(1).strip()
    
    # Authors: walk lines after title; reject titles, headings, banners, affiliations
    KEYWORD_HINTS = re.compile(
        r"\b(keywords?|palabras\s*clave|abstract|resumen|introduction|"
        r"acknowledgments|funding\s+information|correspondence|"
        r"received|accepted|published|copyright|article\s+number|"
        r"editor|reviewed\s+by|department\s+of|university\s+of|institute|"
        r"laboratory|research\s+(?:group|center|institute|article|paper|note)|"
        r"original\s+(?:article|paper|research)|"
        r"review\s+(?:article|paper)|"
        r"short\s+(?:communication|note)|"
        r"perspective|commentary|case\s+report|"
        r"note\s+and\s+comment|brief\s+(?:report|communication))\b",
        re.IGNORECASE,
    )
    
    # Geographic/institutional names that are NOT authors (often appear in OECD-style
    # disclaimers or institutional documents).
    NON_AUTHOR_PHRASES = re.compile(
        r"\b(Golan\s+Heights|East\s+Jerusalem|West\s+Bank|Northern\s+Cyprus|"
        r"European\s+Union|United\s+Nations|Turkish\s+Republic|"
        r"Member\s+States|Republic\s+of\s+Cyprus|OECD\s+Member|"
        r"International\s+Law|Climate\s+Change|Common\s+Agricultural|"
        r"Green\s+Deal|Conservation\s+Biology|Sustainable\s+Development|"
        r"Soil\s+(?:Use|Science|Erosion|Tillage)|"
        r"Cover\s+Crops?|Olive\s+(?:Groves?|Orchards?|Oil)|"
        r"Mediterranean\s+(?:Olive|Region|Basin|Vineyard|Almond))\b",
        re.IGNORECASE,
    )
    
    title_lower = (title or "").lower()
    authors = []
    
    for line in markdown.split("\n")[:60]:
        s = line.strip()
        if not s or len(s) > 600:
            continue
        # First check if line uses footnote-marker separators between authors
        # (e.g. "Smith[1] Jones[1,2] Brown[2]") -- common in some journals.
        # If we see 3+ such markers, split on them and treat each segment as one author.
        markers = re.findall(r"\[[\d,\*\s\-]+\]", s)
        if len(markers) >= 3:
            # Strip ** and split on the marker pattern
            no_format = re.sub(r"\*+|_+", "", s).strip()
            # Split on [...] sequences, keeping the non-bracket parts
            segments = re.split(r"\[[\d,\*\s\-]+\]", no_format)
            # Strip leading/trailing punctuation including separators (commas,
            # semicolons, ampersands, middle-dot, bullet) used between authors.
            # Also strip leading '#' headings — sometimes the author line is
            # rendered as `## Author Name[1], ...`.
            segments = [
                re.sub(r"^[\s#,;&·•\.]+|[\s,;&·•\.]+$", "", seg).strip()
                for seg in segments
                if seg.strip()
            ]
            # Also collapse 'and ' / 'y ' / '&' prefixes used by some publishers
            segments = [re.sub(r"^(?:and|y|&)\s+", "", s, flags=re.IGNORECASE) for s in segments]
            # Filter to ones that look like 2-4 word names (allowing
            # hyphenated surnames and middle initials like 'M.').
            single_name = (
                r"[A-Z\u00C0-\u017F][a-z\u00C0-\u017F\u2019']+"
                r"(?:-[A-Z\u00C0-\u017F][a-z\u00C0-\u017F\u2019']*)*"
            )
            initial = r"[A-Z]\."
            tok = rf"(?:{single_name}|{initial})"
            name_re = re.compile(rf"^{tok}(?:\s+{tok}){{1,4}}$")
            valid_names = [seg for seg in segments if name_re.match(seg)]
            if 2 <= len(valid_names) <= 25:
                authors = valid_names[:15]
                break
        
        clean = re.sub(r"\*+|_+", "", s).strip()
        clean = re.sub(r"\[[\d,\*\s\-]+\]", "", clean).strip()
        clean = re.sub(r"^#+\s*", "", clean).strip()
        if not clean:
            continue
        clean_lower = clean.lower()
        # Skip the title line
        if title and (title_lower == clean_lower or title_lower in clean_lower):
            continue
        if KEYWORD_HINTS.search(clean):
            continue
        if BANNER_RE.search(clean):
            continue
        # Skip lines containing non-author institutional / geographic phrases
        if NON_AUTHOR_PHRASES.search(clean):
            continue
        if clean.isupper():
            continue
        if sum(c.isdigit() for c in clean) > 8:
            continue
        # Skip lines that look like sentences (contain prepositions like "of", "the", "by")
        # in a long list — common in disclaimer paragraphs
        # If the line has many lowercase function words, it's prose not author list
        words = clean.split()
        if len(words) > 15:
            function_words = sum(1 for w in words if w.lower() in {
                "of", "the", "and", "by", "to", "for", "in", "on", "is", "are",
                "with", "from", "this", "that", "have", "has", "be", "or"
            })
            if function_words >= 4:
                continue
        # Ampersand-separated author list (Springer style: "O. M. Nieto & J.
        # Castro & E. Fernández-Ondoño"). Detect early so we don't fall into
        # the regular name-token finder which mis-parses initials.
        if "&" in clean and clean.count("&") >= 1:
            single = (
                r"[A-Z\u00C0-\u017F][a-z\u00C0-\u017F\u2019']+"
                r"(?:-[A-Z\u00C0-\u017F][a-z\u00C0-\u017F\u2019']*)*"
            )
            initial = r"[A-Z]\."
            tok = rf"(?:{single}|{initial})"
            full_name_re = re.compile(rf"^{tok}(?:\s+{tok}){{1,5}}$")
            amp_segments = [seg.strip() for seg in re.split(r"\s*&\s*", clean) if seg.strip()]
            amp_names = [seg for seg in amp_segments if full_name_re.match(seg)]
            if 2 <= len(amp_names) <= 25 and len(amp_names) == len(amp_segments):
                authors = amp_names[:15]
                break
        # Name component allows internal hyphens, but the post-hyphen part may
        # start with an uppercase letter (e.g. "Perera-Fernández").
        name_token = (
            r"[A-Z\u00C0-\u017F][a-z\u00C0-\u017F\u2019']+"
            r"(?:-[A-Za-z\u00C0-\u017F][a-z\u00C0-\u017F\u2019']*)*"
        )
        # Match 2-5 consecutive name tokens (handles "Víctor Hugo Durán Zuazo")
        # with optional Spanish/Dutch/French particles in the middle.
        particle = r"(?:de|van|van\s+der|von|del|della|du|la|le)"
        names = re.findall(
            rf"({name_token}"
            rf"(?:\s+(?:{particle}\s+)?{name_token}){{1,4}})",
            clean,
        )
        if 2 <= len(names) <= 25:
            authors = names[:15]
            break
        # Single-author papers: the line must be JUST the name (possibly with a comma)
        # and short — not a sentence with a person's name in it.
        if len(names) == 1 and 10 <= len(clean) <= 80 and clean.count(" ") <= 6:
            # The line should be mostly the name itself
            if len(names[0]) >= len(clean) * 0.7:
                authors = names
                break
    
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "journal": journal,
    }
