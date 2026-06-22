"""
Table handling: use pymupdf4llm's inline tables only (post-reshape) and
filter out junk tables that look like header/footer/title artifacts.
"""
import re


def is_junk_table(table_lines):
    """
    Heuristic detection of misclassified 'tables' that are really:
      - Title block / author affiliations (cells contain DOI, Received:, etc.)
      - One-column tables (no actual columns)
      - Tables with all cells empty
    Returns True if this table should be removed.
    """
    if not table_lines or len(table_lines) < 2:
        return True
    
    # Parse cells
    rows = []
    for line in table_lines:
        s = line.strip()
        if not s.startswith("|"):
            continue
        if s.endswith("|"):
            s = s[1:-1]
        else:
            s = s[1:]
        cells = [c.strip() for c in s.split("|")]
        rows.append(cells)
    
    if not rows:
        return True
    
    # Detect separator (and discard for analysis)
    data_rows = [r for r in rows if not all(
        re.fullmatch(r"-{2,}|:?-+:?", c.strip()) for c in r if c.strip()
    )]
    if not data_rows:
        return True
    
    n_cols = max(len(r) for r in data_rows)
    if n_cols < 2:
        return True
    
    # Check cell content
    all_cells = [c for r in data_rows for c in r]
    nonempty = [c for c in all_cells if c.strip()]
    if not nonempty:
        return True
    
    total_text = " ".join(nonempty).lower()
    
    # Banner indicators - if these are prominent, it's title/affil block
    BANNER_HITS = [
        "doi:", "received:", "accepted:", "published:", "https://", "http://",
        "copyright", "creative commons", "issn", "journal of", "@", "elsevier",
        "wiley", "springer", "frontiers", "mdpi", "correspondence",
    ]
    hits = sum(1 for kw in BANNER_HITS if kw in total_text)
    if hits >= 2:
        return True
    
    # If avg cell length is huge (>300 chars), it's probably prose-as-table
    avg_len = sum(len(c) for c in nonempty) / len(nonempty)
    if avg_len > 250:
        return True
    
    # If one cell holds 80%+ of total text, it's a single-cell pseudo-table
    if nonempty:
        max_cell = max(len(c) for c in nonempty)
        total_len = sum(len(c) for c in nonempty)
        if total_len > 0 and max_cell / total_len > 0.8:
            return True
    
    return False


def remove_junk_tables(markdown):
    """Walk markdown, identify table blocks, drop junk ones."""
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
            if not is_junk_table(block):
                out_lines.extend(block)
            # else drop silently
            continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)
