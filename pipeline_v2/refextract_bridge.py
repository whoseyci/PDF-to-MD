"""
inspirehep/refextract bridge.

Augments our regex-based `references_v2.py` with structured DOI /
journal / volume / year fields parsed by inspirehep/refextract.

`refextract.extract_references_from_file(pdf)` returns a list of
dicts where each ref has keys like `author`, `doi`, `journal_title`,
`journal_volume`, `journal_year`, `linemarker`, `raw_ref`.

We map those onto the same shape `references_v2.py` already uses
(`{id, raw, authors, year, doi, journal, ...}`) so the citation
linker doesn't need to change. When a ref appears in both, we
prefer refextract's structured fields and keep our regex matcher's
ID assignment (since IDs are what the in-text citations point to).

Usage:

    from pipeline_v2.refextract_bridge import enrich_references
    refs = enrich_references(refs, pdf_path)   # in-place enrichment

Returns the same list, with each dict gaining a `refextract` key
holding the raw refextract dict for that entry (or None if not
matched).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def is_available() -> bool:
    """True if refextract can be imported."""
    try:
        import refextract  # noqa: F401
        return True
    except ImportError:
        return False


def extract_refs(pdf_path: Path) -> List[Dict[str, Any]]:
    """Run refextract on a PDF; return its raw structured-ref list.

    Each entry is a dict with optional keys: `author`, `doi`,
    `journal_title`, `journal_volume`, `journal_year`, `journal_page`,
    `linemarker`, `raw_ref`, `texkey`, `year`. All values are lists
    of strings (refextract's idiom).
    """
    try:
        import refextract
    except ImportError:
        log.debug("refextract not installed; returning empty list")
        return []
    try:
        result = refextract.extract_references_from_file(str(pdf_path))
    except Exception as e:
        log.warning("refextract failed on %s: %s", pdf_path, e)
        return []
    # `result` may be a list directly OR a dict containing 'references'
    if isinstance(result, dict) and "references" in result:
        return result["references"] or []
    if isinstance(result, list):
        return result
    return []


def enrich_references(refs: List[Dict[str, Any]],
                       pdf_path: Path) -> List[Dict[str, Any]]:
    """
    For each entry in our `refs` list, look up a matching refextract
    record (by linemarker / raw text similarity) and attach it.

    Returns the same list, mutated.
    """
    if not is_available():
        return refs
    rex = extract_refs(pdf_path)
    if not rex:
        return refs

    # Build lookup tables. linemarker is the most reliable key when both
    # systems see the same numbering.
    by_linemarker: Dict[str, Dict[str, Any]] = {}
    for entry in rex:
        lms = entry.get("linemarker") or []
        for lm in lms:
            by_linemarker[str(lm).strip()] = entry

    # Fallback: match by trimmed raw-text prefix (cheap fuzzy)
    by_prefix: Dict[str, Dict[str, Any]] = {}
    for entry in rex:
        raws = entry.get("raw_ref") or []
        for raw in raws:
            key = _norm_prefix(raw)
            if key:
                by_prefix.setdefault(key, entry)

    n_matched = 0
    for ref in refs:
        rid = str(ref.get("id") or ref.get("number") or "").strip()
        matched = by_linemarker.get(rid)
        if matched is None:
            raw = ref.get("raw") or ref.get("text") or ""
            matched = by_prefix.get(_norm_prefix(raw))
        if matched is not None:
            ref["refextract"] = _flatten_refextract(matched)
            # Promote useful fields if our side doesn't have them
            re_flat = ref["refextract"]
            if not ref.get("doi") and re_flat.get("doi"):
                ref["doi"] = re_flat["doi"]
            if not ref.get("year") and re_flat.get("year"):
                ref["year"] = re_flat["year"]
            if not ref.get("journal") and re_flat.get("journal_title"):
                ref["journal"] = re_flat["journal_title"]
            n_matched += 1
    log.info("refextract: matched %d / %d references in %s",
              n_matched, len(refs), pdf_path.name)
    return refs


def _norm_prefix(raw: str) -> str:
    """A normalised key for fuzzy ref matching: first ~80 alnum chars."""
    if not raw:
        return ""
    import re
    s = re.sub(r"[^a-zA-Z0-9]", "", raw).lower()
    return s[:80]


def _flatten_refextract(entry: Dict[str, Any]) -> Dict[str, Any]:
    """refextract gives list-of-string values; pick the first item each."""
    out: Dict[str, Any] = {}
    for k, v in entry.items():
        if isinstance(v, list) and v:
            out[k] = v[0]
        elif isinstance(v, list):
            continue
        else:
            out[k] = v
    return out
