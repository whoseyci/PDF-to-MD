"""
Per-kind output validators.

The model returns arbitrary text. The validators check whether the
output actually looks like what we asked for (a Mermaid diagram, a
markdown table, a LaTeX equation, a short alt-sentence) and clean it
up. Output that doesn't validate is rejected — we'd rather have NO
alt text than wrong alt text.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_chat_wrapper(text: str) -> str:
    """Strip common chat-template artefacts (e.g. SmolVLM's `Assistant:` prefix)."""
    if not text:
        return ""
    # Take everything after the LAST `Assistant:` if present
    m = list(re.finditer(r"(?im)^\s*Assistant\s*:\s*", text))
    if m:
        text = text[m[-1].end():]
    # Strip leading "User: …\n" if it leaked through
    text = re.sub(r"(?ims)^\s*User\s*:.*?\n\s*Assistant\s*:\s*", "", text)
    return text.strip()


def _extract_code_block(text: str, lang: Optional[str] = None) -> Optional[str]:
    """Find a fenced code block (optionally of a given language) and return its body."""
    if lang:
        pat = re.compile(rf"```{lang}\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    else:
        pat = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)
    m = pat.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Sentence / alt-text validation
# ---------------------------------------------------------------------------

_BAD_PREFIXES = re.compile(
    r"^(sure|certainly|of course|here is|here's|i'?ll|i can|let me|this image|this figure)\b",
    re.IGNORECASE,
)

# Phrases that small VLMs love to parrot back from the prompt itself —
# we treat them as zero-information output and reject.
_PROMPT_ECHO_PATTERNS = [
    re.compile(r"(?i)\bcolumns?\s*[`:]+\s*Category\s*[`,]"),
    re.compile(r"(?i)\bValue\s*\(units\)"),
    re.compile(r"(?i)\bMERMAID_UNAVAILABLE\b"),
    re.compile(r"(?i)\bTABLE_UNAVAILABLE\b"),
    re.compile(r"(?i)\bLATEX_UNAVAILABLE\b"),
    re.compile(r"(?i)\bin one short sentence\b"),
    re.compile(r"(?i)\breproduce them as a markdown table\b"),
    re.compile(r"(?i)\bdo not invent\b"),
]


def _is_prompt_echo(s: str) -> bool:
    return any(p.search(s) for p in _PROMPT_ECHO_PATTERNS)


def validate_short_sentence(raw: str, max_words: int = 60) -> Optional[str]:
    """
    Trim and validate a "one short sentence" model output.

    Rejects:
    - empty / whitespace-only
    - very long blobs that don't look like a single sentence
    - leading filler phrases ("Sure! Here is…")

    Returns the cleaned sentence (a string ending in `.`), or None.
    """
    text = _strip_chat_wrapper(raw)
    if not text:
        return None

    # Take just the first sentence-ish chunk.
    # Cut at first newline (multiple lines → almost always model rambled).
    text = text.split("\n", 1)[0].strip()
    # Cut at first period+space pair to keep it one sentence.
    m = re.match(r"^(.+?[\.\!\?])(?:\s|$)", text)
    if m:
        text = m.group(1).strip()

    text = re.sub(r"^[\-\*\u2022]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    if _BAD_PREFIXES.match(text):
        return None
    if _is_prompt_echo(text):
        return None
    words = text.split()
    if not (3 <= len(words) <= max_words):
        return None
    # End with a period if missing
    if not text.endswith((".", "!", "?")):
        text = text + "."
    return text


# ---------------------------------------------------------------------------
# Mermaid validation
# ---------------------------------------------------------------------------

_MERMAID_HEADER_RE = re.compile(
    r"^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|journey|gitGraph)\b",
    re.IGNORECASE | re.MULTILINE,
)


def validate_mermaid(raw: str) -> Optional[str]:
    """
    Pull a Mermaid diagram body out of the model output and sanity-check it.

    Returns the diagram text (no fence) or None on failure.
    """
    text = _strip_chat_wrapper(raw)
    if not text or "MERMAID_UNAVAILABLE" in text:
        return None

    # 1) Prefer a fenced ```mermaid block
    block = _extract_code_block(text, lang="mermaid")
    # 2) Or any fenced block whose first line looks like a Mermaid header
    if not block:
        block = _extract_code_block(text)
        if block and not _MERMAID_HEADER_RE.search(block):
            block = None
    # 3) Or treat the whole text as mermaid if it starts with a header
    if not block and _MERMAID_HEADER_RE.search(text.lstrip()):
        block = text.strip()

    if not block:
        return None

    block = block.strip()
    # Require at least one edge / arrow to be plausible
    if not re.search(r"-->|---|==>|-\.->", block):
        return None
    # Reject pathologically long / runaway output
    if block.count("\n") > 60 or len(block) > 4000:
        return None
    return block


# ---------------------------------------------------------------------------
# Markdown-table validation
# ---------------------------------------------------------------------------

_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|\-]+\|\s*$", re.MULTILINE)


def validate_markdown_table(raw: str) -> Optional[str]:
    """
    Find a well-formed Github-flavored markdown table in the model output.

    Returns the table block, or None if not present / not well-formed.
    """
    text = _strip_chat_wrapper(raw)
    if not text or "TABLE_UNAVAILABLE" in text:
        return None

    # First try fenced code block
    block = _extract_code_block(text)
    candidates = [block] if block else []
    # And the raw text
    candidates.append(text)

    for cand in candidates:
        if not cand:
            continue
        if not _TABLE_SEP_RE.search(cand):
            continue
        # Extract the contiguous block of pipe-lines starting at the first one
        lines = cand.splitlines()
        block_lines = []
        in_table = False
        for line in lines:
            if _TABLE_ROW_RE.match(line):
                block_lines.append(line.strip())
                in_table = True
            elif in_table:
                break
        if len(block_lines) < 3:  # header + separator + ≥1 data row
            continue
        # Sanity: all rows should have the same column count
        ncols = [l.count("|") for l in block_lines]
        if max(ncols) - min(ncols) > 1:
            continue
        return "\n".join(block_lines)
    return None


# ---------------------------------------------------------------------------
# LaTeX equation validation
# ---------------------------------------------------------------------------

def validate_latex(raw: str) -> Optional[str]:
    """Pull a `$$…$$` or `$…$` LaTeX expression out of the model output."""
    text = _strip_chat_wrapper(raw)
    if not text or "LATEX_UNAVAILABLE" in text:
        return None
    # Prefer block math
    m = re.search(r"\$\$([^$]+)\$\$", text, re.DOTALL)
    if m:
        body = m.group(1).strip()
        if 2 <= len(body) <= 1000:
            return f"$${body}$$"
    # Fall back to inline math
    m = re.search(r"\$([^$\n]{2,200})\$", text)
    if m:
        body = m.group(1).strip()
        return f"${body}$"
    return None
