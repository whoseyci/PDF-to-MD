"""Strip redundant front matter (banner, duplicate title, Received/Accepted etc.)
that appears in the body markdown after the curated metadata block."""
import re


def strip_front_matter(body_md, known_title=None):
    """
    Strip the redundant front matter block at the start of the body.
    Stops at:
      - "## Abstract" / "## ABSTRACT" / "## Introduction"
      - "## **1." or "## 1." (numbered section)
    Whichever comes first.
    """
    if not body_md:
        return body_md
    
    # Find the boundary where real content begins
    boundary_patterns = [
        # Main section start
        r"(?im)^#{1,3}\s*\**\s*(abstract|resumen)\b",
        r"(?im)^#{1,3}\s*\**\s*(introduction|background|overview|1\s*\.)\b",
        r"(?im)^#{1,3}\s*\**\s*\d+\s*\.\s*\**\s*[A-Z][a-z]+",
        # Numbered section as plain markdown
        r"(?im)^#{1,3}\s*\**\s*1\b",
    ]
    
    earliest_boundary = None
    for pat in boundary_patterns:
        m = re.search(pat, body_md)
        if m and (earliest_boundary is None or m.start() < earliest_boundary):
            earliest_boundary = m.start()
    
    if earliest_boundary is None:
        return body_md  # can't find boundary, leave as-is
    
    # If the boundary is more than ~4500 chars in, that's the real text already
    # (we probably weren't dealing with a front matter block).
    # 4500 (vs 3500) allows for journals with a large multi-column abstract table.
    if earliest_boundary > 4500:
        return body_md
    
    # Strip everything before the boundary, BUT preserve any PDF page-
    # boundary markers (invisible U+2063 tokens of the form \u2063\u2063\u2063PB<N>/<T>\u2063\u2063\u2063)
    # found in the stripped region. We want every PDF page to be represented
    # by a visible separator in the final output.
    stripped = body_md[:earliest_boundary]
    preserved_markers = re.findall(r"\u2063{3}PB\d+/\d+\u2063{3}", stripped)
    suffix = body_md[earliest_boundary:]
    if preserved_markers:
        return "\n\n".join(preserved_markers) + "\n\n" + suffix
    return suffix
