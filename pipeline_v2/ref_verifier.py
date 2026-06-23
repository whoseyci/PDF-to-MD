"""
Reference verifier inspired by refchecker (markrussinovich/refchecker).

Verifies each reference against external bibliographic databases:

  * **Crossref** (api.crossref.org/works/<doi>) -- canonical for DOIs
  * **OpenAlex** (api.openalex.org/works) -- DOI or title search
  * **Semantic Scholar** (api.semanticscholar.org) -- title search
    (used only as backup; rate-limited without API key)

For each ref, we record a verdict:
  * `verified`: a database returned a hit AND the title/author
    overlaps with what we extracted
  * `mismatch`: a database returned a hit but the metadata diverges
    (possible OCR error or wrong citation)
  * `not_found`: no database returned a hit (possibly fabricated, or
    just non-indexed -- e.g. grey literature)
  * `skipped`: not enough info to query (no DOI, no title, no year)
  * `error`: network error or API failure

Verification is **opt-in** because it makes external network calls
and burns API quota. Run it as a post-processing step:

    from pipeline_v2.ref_verifier import verify_references
    for ref in refs:
        verify_references(ref)   # mutates ref in place

We're conservative on network calls: 1 Crossref hit per ref with
DOI (cheap), otherwise 1 OpenAlex search by title. We do NOT call
all three for every ref.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# --------------------------------------------------------------------
# HTTP helpers (stdlib only, no `requests` dep)
# --------------------------------------------------------------------

_USER_AGENT = "PDF-to-MD-RefVerifier/1.0 (https://github.com/whoseyci/PDF-to-MD)"


def _http_get_json(url: str, *, timeout: float = 10.0
                    ) -> Optional[Dict[str, Any]]:
    req = urllib.request.Request(url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        log.debug("HTTP %s on %s", e.code, url)
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.debug("network error on %s: %s", url, e)
        return None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------
# Crossref
# --------------------------------------------------------------------

def lookup_crossref_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Query Crossref for a known DOI; returns the `message` block or None."""
    doi = (doi or "").strip().lstrip("https://doi.org/").lstrip("doi:")
    if not doi:
        return None
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    data = _http_get_json(url)
    if not data or "message" not in data:
        return None
    return data["message"]


# --------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------

def lookup_openalex_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    doi = (doi or "").strip().lstrip("https://doi.org/").lstrip("doi:")
    if not doi:
        return None
    url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}"
    return _http_get_json(url)


def search_openalex_by_title(title: str, *, year: Optional[int] = None
                              ) -> Optional[Dict[str, Any]]:
    title = (title or "").strip()
    if len(title) < 8:
        return None
    params = {"search": title, "per-page": "3"}
    if year:
        params["filter"] = f"publication_year:{year}"
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = _http_get_json(url)
    if not data or not data.get("results"):
        return None
    return data["results"][0]


# --------------------------------------------------------------------
# Compare returned metadata to what we extracted
# --------------------------------------------------------------------

def _norm_title(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"[^a-zA-Z0-9 ]", " ", t).lower()
    return re.sub(r"\s+", " ", t).strip()


def _title_overlap_score(a: str, b: str) -> float:
    """Jaccard over word sets, computed on normalised titles."""
    aw = set(_norm_title(a).split())
    bw = set(_norm_title(b).split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / max(1, len(aw | bw))


def _extract_year_from_ref(ref: Dict[str, Any]) -> Optional[int]:
    for k in ("year", "publication_year"):
        v = ref.get(k)
        if v:
            try:
                y = int(str(v)[:4])
                if 1500 < y < 2100:
                    return y
            except ValueError:
                pass
    # try mining from raw
    raw = ref.get("raw") or ref.get("text") or ""
    m = re.search(r"\b(19|20)\d{2}\b", raw)
    if m:
        return int(m.group(0))
    return None


_JOURNAL_LIKE_RE = re.compile(
    # patterns like "Foo Bar 12, 345" or "J. Biol. 99 (3), 12-34" --
    # these are journal-volume-page-citation fragments, NOT titles.
    r"\b\d+\s*(?:\(\d+\))?\s*[,;]\s*[ep]?\d", re.IGNORECASE)


def _looks_like_journal_citation(s: str) -> bool:
    """True if a string looks like a journal name + volume + page citation."""
    if not s: return True
    if _JOURNAL_LIKE_RE.search(s): return True
    # All-uppercase abbreviated journal blobs ("J BIOL CHEM 999, 12345")
    words = s.split()
    if len(words) <= 3 and any(w.endswith(".") and len(w) <= 6 for w in words):
        return True
    return False


def _extract_title_from_ref(ref: Dict[str, Any]) -> Optional[str]:
    for k in ("title", "article_title"):
        v = ref.get(k)
        if v and len(str(v)) > 8:
            return str(v)
    raw = ref.get("raw") or ref.get("text") or ""
    if not raw:
        return None
    # Heuristic 1: text between year and next sentence-ending period.
    m = re.search(
        r"\b(?:19|20)\d{2}\b[).,]?\s*(.{15,250}?)(?<=[a-z])\.", raw)
    if m:
        title = m.group(1).strip().strip(".")
        if (len(title) > 12 and " " in title
                and not _looks_like_journal_citation(title)):
            return title
    # Heuristic 2: longest sentence-shaped run.
    candidates = [
        s.strip() for s in re.split(r"(?<=[a-z])\.\s+", raw)
        if 15 < len(s.strip()) < 250]
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        if _looks_like_journal_citation(c): continue
        words = c.split()
        if len(words) >= 4 and not re.match(r"^[A-Z][a-z]+,\s", c):
            return c
    return None


# --------------------------------------------------------------------
# Main verify function
# --------------------------------------------------------------------

def verify_reference(ref: Dict[str, Any], *,
                      use_crossref: bool = True,
                      use_openalex: bool = True,
                      sleep_between: float = 0.1
                      ) -> Dict[str, Any]:
    """Verify a single reference; returns the verification record.

    Also writes the record into ref['verification'] for downstream use.
    """
    verdict = {"verdict": "skipped", "source": None, "note": "",
                "match_score": 0.0}
    doi = (ref.get("doi") or "").strip()

    # Path 1: Crossref by DOI (highest signal)
    if use_crossref and doi:
        time.sleep(sleep_between)
        msg = lookup_crossref_doi(doi)
        if msg:
            crossref_title = ""
            tlist = msg.get("title") or []
            if tlist:
                crossref_title = tlist[0]
            ref_title = _extract_title_from_ref(ref) or ""
            score = _title_overlap_score(ref_title, crossref_title)
            # "verified" if title overlap is good OR if our title is
            # too short to be meaningful (e.g. refextract returned
            # only "DOI Foo Bar" without the title). A DOI that
            # resolves IS strong evidence on its own.
            if not ref_title or len(ref_title) < 15:
                verdict = {
                    "verdict": "verified", "source": "crossref",
                    "note": f"DOI resolves (no extractable title to cross-check)",
                    "match_score": 0.5,
                    "crossref_title": crossref_title,
                }
            elif score >= 0.4:
                verdict = {
                    "verdict": "verified", "source": "crossref",
                    "note": f"DOI resolves; title overlap {score:.2f}",
                    "match_score": round(score, 3),
                    "crossref_title": crossref_title,
                }
            else:
                verdict = {
                    "verdict": "mismatch", "source": "crossref",
                    "note": (f"DOI resolves but title differs (overlap "
                              f"{score:.2f}): expected {ref_title!r} "
                              f"vs got {crossref_title!r}"),
                    "match_score": round(score, 3),
                    "crossref_title": crossref_title,
                }
            ref["verification"] = verdict
            return verdict
        # fall through to OpenAlex

    # Path 2: OpenAlex search
    if use_openalex:
        time.sleep(sleep_between)
        oa = None
        if doi:
            oa = lookup_openalex_by_doi(doi)
        if oa is None:
            title = _extract_title_from_ref(ref)
            year = _extract_year_from_ref(ref)
            if title:
                oa = search_openalex_by_title(title, year=year)
        if oa:
            oa_title = oa.get("title", "") or oa.get("display_name", "")
            ref_title = _extract_title_from_ref(ref) or ""
            score = _title_overlap_score(ref_title, oa_title)
            if score >= 0.4 or not ref_title:
                verdict = {
                    "verdict": "verified", "source": "openalex",
                    "note": f"OpenAlex hit; title overlap {score:.2f}",
                    "match_score": round(score, 3),
                    "openalex_title": oa_title,
                    "openalex_id": oa.get("id"),
                }
            else:
                verdict = {
                    "verdict": "mismatch", "source": "openalex",
                    "note": (f"OpenAlex hit but title differs (overlap "
                              f"{score:.2f}): expected {ref_title!r} "
                              f"vs got {oa_title!r}"),
                    "match_score": round(score, 3),
                    "openalex_title": oa_title,
                }
            ref["verification"] = verdict
            return verdict

    if not doi and not _extract_title_from_ref(ref):
        verdict["note"] = "no DOI and no parseable title"
    else:
        verdict = {"verdict": "not_found", "source": None,
                    "note": "no Crossref / OpenAlex match",
                    "match_score": 0.0}
    ref["verification"] = verdict
    return verdict


def verify_references(refs: List[Dict[str, Any]], *,
                       use_crossref: bool = True,
                       use_openalex: bool = True,
                       sleep_between: float = 0.1,
                       progress: bool = False
                       ) -> Dict[str, int]:
    """Verify a batch of refs; returns a summary count by verdict."""
    summary = {"verified": 0, "mismatch": 0, "not_found": 0,
                "skipped": 0, "error": 0}
    for i, ref in enumerate(refs):
        try:
            v = verify_reference(ref,
                                   use_crossref=use_crossref,
                                   use_openalex=use_openalex,
                                   sleep_between=sleep_between)
            summary[v["verdict"]] = summary.get(v["verdict"], 0) + 1
        except Exception as e:
            ref["verification"] = {
                "verdict": "error", "source": None,
                "note": f"{type(e).__name__}: {e}",
                "match_score": 0.0,
            }
            summary["error"] += 1
        if progress and (i + 1) % 10 == 0:
            print(f"  verified {i+1}/{len(refs)}: {summary}", flush=True)
    return summary
