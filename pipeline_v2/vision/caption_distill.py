"""E16 — Distillation pattern for caption extraction.

When a figure has no chart_extract result and no mermaid diagram,
today we either invoke Gemma 4 to generate alt-text (~16 min/figure)
or fall back to using the raw caption.

This module adds a **student-then-teacher** policy:

  * Student (rule-based, ~milliseconds): If the caption already has
    a complete sentence describing the figure (>= 12 words, ends with
    a period, contains a noun + verb pattern), use it directly. No
    LLM call needed.

  * Teacher (Gemma 4, slow): Only invoked when the student flags
    low-confidence. The teacher gets a tighter prompt that includes
    the student's draft as a starting point.

Decision policy is local and transparent — no LLM call to decide
whether to call the LLM.

API:
    from pipeline_v2.vision.caption_distill import distill_alt_text
    result = distill_alt_text(caption=fig.caption,
                                 fig_ocr_text=fig.ocr_text,
                                 image_path=fig_path,
                                 teacher_fn=gemma4_describe)
    # result.alt_text, result.source ('student' | 'teacher' | 'caption_only')
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DistillResult:
    alt_text: str
    source: str = "unknown"          # 'student' | 'teacher' | 'caption_only' | 'empty'
    student_confidence: float = 0.0
    teacher_invoked: bool = False
    teacher_elapsed_s: Optional[float] = None
    reason: str = ""


# ----------------------------------------------------------------------
# Student: cheap rule-based caption assessment
# ----------------------------------------------------------------------

_GENERIC_OPENERS = {
    "fig", "figure", "fig.", "image", "graphic", "plot",
    "chart", "diagram",
}

_INFORMATIVE_VERBS = {
    "shows", "depicts", "illustrates", "presents", "displays",
    "compares", "summarises", "summarizes", "describes",
    "indicates", "represents", "demonstrates", "highlights",
    "plots", "visualises", "visualizes",
}


def assess_caption(caption: Optional[str]) -> float:
    """Return [0..1] confidence that the caption is self-sufficient
    alt-text. Higher = less likely to need the VLM."""
    if not caption:
        return 0.0
    c = caption.strip()
    if not c:
        return 0.0
    words = c.split()
    n_words = len(words)
    # Tiny captions are not useful as alt-text
    if n_words < 6:
        return 0.0
    # Strip leading "Figure 3." / "Fig. 1:" then check substance
    body = re.sub(
        r"^\s*(?:figure|fig\.?|fig)\s*\d+[a-z]?\s*[:.\-—)]?\s*",
        "", c, flags=re.IGNORECASE).strip()
    if not body:
        return 0.0
    body_words = body.split()
    if len(body_words) < 5:
        return 0.0
    body_lower = body.lower()
    # Signal 1: caption ends with a period / question mark / exclamation
    has_terminator = body.rstrip().endswith((".", "?", "!"))
    # Signal 2: contains an informative verb
    has_verb = any(v in body_lower for v in _INFORMATIVE_VERBS)
    # Signal 3: contains specific nouns (avoid pure "Figure showing X")
    starts_generic = body_lower.split()[0] in _GENERIC_OPENERS
    # Signal 4: contains at least one number (units, year, sample size)
    has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", body))

    score = 0.0
    if len(body_words) >= 12: score += 0.30
    elif len(body_words) >= 8: score += 0.15
    if has_terminator: score += 0.20
    if has_verb: score += 0.20
    if not starts_generic: score += 0.15
    if has_number: score += 0.15
    return round(min(1.0, score), 3)


def truncate_alt_text(text: str, max_chars: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


# ----------------------------------------------------------------------
# Distillation top-level
# ----------------------------------------------------------------------

def distill_alt_text(*,
                       caption: Optional[str] = None,
                       fig_ocr_text: Optional[str] = None,
                       image_path: Optional[Path] = None,
                       teacher_fn: Optional[Callable[[Path, str], str]] = None,
                       student_threshold: float = 0.55,
                       max_chars: int = 280,
                       ) -> DistillResult:
    """Decide which alt-text to emit for a figure.

    Order of preference:
      1. Caption alone is good enough (student_confidence >= threshold)
      2. Teacher (VLM) is available and student says low-confidence
      3. Caption alone, even if mediocre, beats no alt-text
      4. Empty result
    """
    student_conf = assess_caption(caption)

    if student_conf >= student_threshold:
        return DistillResult(
            alt_text=truncate_alt_text(caption or "", max_chars),
            source="student",
            student_confidence=student_conf,
            reason=f"caption is self-sufficient ({student_conf:.2f})",
        )

    # Try the teacher if we have one and the figure is actually loadable
    if teacher_fn is not None and image_path is not None \
            and Path(image_path).exists():
        prompt = _build_teacher_prompt(caption, fig_ocr_text)
        import time
        t0 = time.time()
        try:
            out = teacher_fn(Path(image_path), prompt)
        except Exception as e:
            return DistillResult(
                alt_text=truncate_alt_text(caption or "", max_chars),
                source="caption_only",
                student_confidence=student_conf,
                teacher_invoked=True,
                teacher_elapsed_s=round(time.time() - t0, 2),
                reason=f"teacher failed: {type(e).__name__}: {e}",
            )
        teacher_t = round(time.time() - t0, 2)
        out = (out or "").strip()
        if out:
            return DistillResult(
                alt_text=truncate_alt_text(out, max_chars),
                source="teacher",
                student_confidence=student_conf,
                teacher_invoked=True,
                teacher_elapsed_s=teacher_t,
                reason="teacher answer accepted",
            )
        # Teacher returned nothing -- fall through

    # Last resort: caption alone (possibly empty)
    if caption and caption.strip():
        return DistillResult(
            alt_text=truncate_alt_text(caption, max_chars),
            source="caption_only",
            student_confidence=student_conf,
            reason="no teacher; using raw caption",
        )
    return DistillResult(
        alt_text="",
        source="empty",
        student_confidence=student_conf,
        reason="no caption, no teacher",
    )


def _build_teacher_prompt(caption: Optional[str],
                            ocr_text: Optional[str]) -> str:
    parts = ["<image>",
              "Describe this figure in ONE complete English sentence "
              "suitable as alt-text (under 60 words). "
              "Mention the figure type (bar chart, map, photo, etc.) "
              "and the main quantity / subject it shows. "
              "Do not include the words 'figure' or 'image' at the start."]
    if caption and caption.strip():
        parts.append(f"\nFor context, the published caption reads: "
                      f"{caption.strip()[:200]}")
    if ocr_text and ocr_text.strip() and len(ocr_text.strip()) < 200:
        parts.append(f"\nThe figure contains this text: "
                      f"{ocr_text.strip()}")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Stats reporter
# ----------------------------------------------------------------------

@dataclass
class DistillStats:
    n_total: int = 0
    n_student: int = 0
    n_teacher: int = 0
    n_caption_only: int = 0
    n_empty: int = 0
    teacher_seconds_total: float = 0.0

    def add(self, r: DistillResult):
        self.n_total += 1
        if r.source == "student": self.n_student += 1
        elif r.source == "teacher":
            self.n_teacher += 1
            self.teacher_seconds_total += r.teacher_elapsed_s or 0
        elif r.source == "caption_only": self.n_caption_only += 1
        else: self.n_empty += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "student": self.n_student,
            "teacher": self.n_teacher,
            "caption_only": self.n_caption_only,
            "empty": self.n_empty,
            "teacher_seconds_total": round(self.teacher_seconds_total, 2),
            "teacher_calls_saved_pct": (
                round(100 * self.n_student / max(1, self.n_total), 1)
                if self.n_total else 0
            ),
        }
