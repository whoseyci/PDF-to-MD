"""
Selective LLM-boost dispatcher (pattern from VikParuchuri/marker).

`marker --use_llm` runs the classical pipeline first then selectively
re-processes *only* the document blocks with low confidence using an
LLM. This avoids the slowness of LLM-everything while still catching
the few blocks where the classical pipeline made a wrong call.

We implement the same idea here as a small policy module that decides,
per block, whether to invoke the VLM. The actual VLM call happens in
the caller — we only return the decision so the dispatch logic is
testable without a model.

Decisions:

  * `skip`     — accept the classical output, don't call the LLM
  * `validate` — call the LLM with a "is this right?" prompt; only
                 replace the classical output if the LLM flags it
  * `replace`  — call the LLM and replace the classical output unconditionally
  * `extract`  — classical pipeline produced nothing; call the LLM
                 as the primary extractor

Block types this knows about:
  * `figure`      — uses `chart_extraction.confidence` and `.status`
  * `table`       — uses `tables_v2.py` confidence
  * `reference`   — uses `verification.verdict` from ref_verifier
  * `equation`    — uses tesseract-confidence of an OCR fallback
  * `paragraph`   — uses ligature / soft-hyphen residue heuristics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


Decision = Literal["skip", "validate", "replace", "extract"]


@dataclass
class BoostDecision:
    decision: Decision
    reason: str
    cost_estimate_seconds: float = 0.0


# --------------------------------------------------------------------
# Per-block-type policies
# --------------------------------------------------------------------

def decide_for_figure(chart_extraction: Optional[Dict[str, Any]]
                       ) -> BoostDecision:
    """Decide whether to invoke the VLM on a figure.

    Inputs:
      `chart_extraction`: the dict from `chart_extract` (may be None
        if no classical extractor was tried).
    """
    if chart_extraction is None:
        return BoostDecision("extract", "no classical extractor available",
                              cost_estimate_seconds=120.0)
    status = chart_extraction.get("status")
    conf = float(chart_extraction.get("confidence") or 0.0)
    if status == "ok" and conf >= 0.85:
        return BoostDecision("validate",
                              f"OK with confidence {conf:.2f}; spot-check",
                              cost_estimate_seconds=40.0)
    if status == "ok":
        return BoostDecision("validate",
                              f"OK but low confidence {conf:.2f}",
                              cost_estimate_seconds=40.0)
    if status == "partial":
        return BoostDecision("validate",
                              f"PARTIAL ({conf:.2f}); ask VLM to fill gaps",
                              cost_estimate_seconds=60.0)
    if status in ("no_axis", "no_bars", "ocr_failed"):
        return BoostDecision("extract",
                              f"classical extractor returned {status}",
                              cost_estimate_seconds=120.0)
    if status == "unsupported":
        return BoostDecision("extract",
                              "no classical extractor for this kind",
                              cost_estimate_seconds=120.0)
    # error / unknown
    return BoostDecision("extract",
                          f"classical extractor error: {status}",
                          cost_estimate_seconds=120.0)


def decide_for_reference(ref: Dict[str, Any]) -> BoostDecision:
    """When should the LLM re-extract a reference?"""
    raw = (ref.get("raw") or ref.get("text") or "").strip()
    verification = (ref.get("verification") or {})
    verdict = verification.get("verdict")

    if not raw:
        return BoostDecision("skip", "empty raw text")
    if verdict == "verified":
        return BoostDecision("skip", "already verified externally")
    if verdict == "mismatch":
        return BoostDecision("replace",
                              "external metadata disagrees -- LLM rewrite",
                              cost_estimate_seconds=20.0)
    if verdict == "not_found" and len(raw) < 30:
        return BoostDecision("extract",
                              "too-short raw text; LLM may parse better",
                              cost_estimate_seconds=20.0)
    # Heavy OCR noise -> retry with LLM
    alpha_ratio = (sum(c.isalpha() for c in raw) / max(1, len(raw)))
    if alpha_ratio < 0.55:
        return BoostDecision("extract",
                              f"high non-alpha ratio ({alpha_ratio:.2f}); "
                              "raw text looks like OCR garbage",
                              cost_estimate_seconds=20.0)
    return BoostDecision("skip", "regex parse looks clean")


def decide_for_table(table_meta: Dict[str, Any]) -> BoostDecision:
    """Decide for table extraction."""
    if not table_meta or not table_meta.get("rows"):
        return BoostDecision("extract", "no rows extracted",
                              cost_estimate_seconds=60.0)
    conf = float(table_meta.get("confidence") or 0.5)
    if conf >= 0.85:
        return BoostDecision("skip", f"high confidence {conf:.2f}")
    if conf >= 0.55:
        return BoostDecision("validate", f"medium confidence {conf:.2f}",
                              cost_estimate_seconds=40.0)
    return BoostDecision("replace",
                          f"low confidence {conf:.2f}; redo with LLM",
                          cost_estimate_seconds=80.0)


def decide_for_paragraph(text: str) -> BoostDecision:
    """Decide for body paragraphs (heavy OCR / ligature residue)."""
    if not text or len(text) < 20:
        return BoostDecision("skip", "too short to bother")
    import re
    # Real artifact signals -- soft hyphens, unconverted ligatures,
    # punctuation runs of >= 3 (".....", "—————" etc.), and 3+ ALL-CAPS
    # short runs ("ABC DEF GHI" patterns from OCR scrambling).
    bad = 0
    bad += text.count("\u00ad") * 3        # soft hyphens weight heavily
    bad += text.count("ﬁ") + text.count("ﬂ")
    bad += text.count("ﬀ") + text.count("ﬃ") + text.count("ﬄ")
    bad += len(re.findall(r"[^\w\s\.\,\;\:\?\!\(\)\-\—\"'/\u00b0\%\$]{3,}",
                            text))
    # Single-character orphan words that aren't real (excluding 'a' / 'I')
    bad += len(re.findall(r"\b(?![aAI]\b)[a-zA-Z]\b", text))
    ratio = bad / max(1, len(text))
    if ratio > 0.04:
        return BoostDecision("replace",
                              f"heavy artifact ratio {ratio:.2%}",
                              cost_estimate_seconds=30.0)
    if ratio > 0.015:
        return BoostDecision("validate",
                              f"some artifacts ({ratio:.2%}); spot-check",
                              cost_estimate_seconds=20.0)
    return BoostDecision("skip", f"clean (artifact ratio {ratio:.2%})")


# --------------------------------------------------------------------
# Top-level dispatcher
# --------------------------------------------------------------------

def estimate_corpus_cost(decisions: list[BoostDecision]) -> Dict[str, Any]:
    """Sum decisions into a cost & action summary for a corpus run."""
    counts = {"skip": 0, "validate": 0, "replace": 0, "extract": 0}
    total_s = 0.0
    for d in decisions:
        counts[d.decision] += 1
        total_s += d.cost_estimate_seconds
    return {
        "decisions_by_action": counts,
        "total_estimated_seconds": round(total_s, 1),
        "total_estimated_hours": round(total_s / 3600, 2),
        "n_blocks": len(decisions),
        "frac_skipped": round(counts["skip"] / max(1, len(decisions)), 3),
    }
