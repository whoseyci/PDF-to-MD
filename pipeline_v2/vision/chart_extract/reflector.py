"""E17 — Deliberation Reflector for chart_extract.

When a chart extractor returns PARTIAL or low-confidence OK, today we
just accept the result. The Reflector pattern (from RecursiveMAS's
Deliberation style) asks: "given this extraction + its warnings,
should we retry with different parameters before giving up?"

The Reflector is pure-Python — no LLM call. It inspects:

  1. The extractor result (status, confidence, warnings, reasons)
  2. The image features from the Mixture classifier (if available)
  3. The expected vs actual output shape (e.g. simple_bars returned
     1 bar on an image with 8 saturated columns → almost certainly
     wrong)

…and emits a `ReflectionDecision`:

  * accept                 → use the result as-is
  * retry_with_params(p)   → re-run the extractor with tighter knobs
  * fallback_to_next_kind  → walk the Mixture ladder to the next-best
                              extractor
  * give_up                → return PARTIAL/UNSUPPORTED

The Reflector is **bounded**: at most one retry per extractor per
figure, so worst-case cost is 2× extraction time, not unbounded.

Used by:
    pipeline_v2/vision/chart_extract/reflective_runner.py
    or any code that wants self-correcting extraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import ChartExtractionResult, ExtractionStatus


class ReflectionAction(str, Enum):
    ACCEPT = "accept"
    RETRY_WITH_PARAMS = "retry_with_params"
    FALLBACK_TO_NEXT_KIND = "fallback_to_next_kind"
    GIVE_UP = "give_up"


@dataclass
class ReflectionDecision:
    action: ReflectionAction
    reason: str = ""
    retry_params: Dict[str, Any] = field(default_factory=dict)
    suggested_kind: Optional[str] = None  # for FALLBACK_TO_NEXT_KIND
    confidence_delta: float = 0.0          # how much we expect to gain


def reflect(result: ChartExtractionResult,
             *,
             expected_n_bars: Optional[int] = None,
             expected_n_circles: Optional[int] = None,
             image_features: Optional[Any] = None,
             fallback_ladder: Optional[List[str]] = None,
             already_retried: bool = False,
             ) -> ReflectionDecision:
    """Decide what to do with `result`.

    Args:
      expected_n_bars: from image features, how many vertical bar
        strips were detected (mixture classifier feature)
      expected_n_circles: how many circles Hough detected
      image_features: full ImageFeatures dataclass (optional)
      fallback_ladder: ordered list of FigureKind names to try next
      already_retried: if True, never recommend another retry
        (bounded loop)
    """
    status = result.status
    n_bars_got = len(result.bar_boxes or [])
    n_slices_got = len(result.pie_slices or [])
    n_series_got = len(result.line_series or [])

    # --- Hard-OK case: accept ---
    if status == ExtractionStatus.OK and result.confidence >= 0.7:
        return ReflectionDecision(
            action=ReflectionAction.ACCEPT,
            reason="high-confidence OK")

    # --- Hard-OK but low confidence: maybe retry ---
    if status == ExtractionStatus.OK and result.confidence < 0.7:
        # Did we miss bars?
        if (expected_n_bars is not None and n_bars_got > 0
                and expected_n_bars >= n_bars_got + 2):
            if not already_retried:
                return ReflectionDecision(
                    action=ReflectionAction.RETRY_WITH_PARAMS,
                    reason=(f"got {n_bars_got} bars but image has "
                              f"{expected_n_bars} strips; loosening "
                              "bar-detection thresholds"),
                    retry_params={"min_bar_aspect": 0.20,
                                    "min_fill_ratio": 0.40,
                                    "min_bar_count_pixels": 0.0005},
                    confidence_delta=0.2,
                )
        return ReflectionDecision(
            action=ReflectionAction.ACCEPT,
            reason="low-conf OK, no obvious fix")

    # --- PARTIAL: try to figure out why ---
    if status == ExtractionStatus.PARTIAL:
        reason_lower = (result.reason or "").lower()
        if "label" in reason_lower or "missing" in reason_lower:
            if not already_retried:
                return ReflectionDecision(
                    action=ReflectionAction.RETRY_WITH_PARAMS,
                    reason="missing labels; widening OCR band",
                    retry_params={"ocr_band_tolerance_frac": 0.10},
                    confidence_delta=0.1,
                )
        return ReflectionDecision(
            action=ReflectionAction.ACCEPT,
            reason=f"partial accepted: {result.reason}")

    # --- NO_BARS: maybe wrong extractor kind ---
    if status == ExtractionStatus.NO_BARS:
        if fallback_ladder:
            for k in fallback_ladder:
                # Don't recommend the same kind we're already doing
                if not (result.extractor or "").startswith(k.split("/")[0]):
                    return ReflectionDecision(
                        action=ReflectionAction.FALLBACK_TO_NEXT_KIND,
                        reason=f"no bars found; trying {k}",
                        suggested_kind=k,
                    )
        return ReflectionDecision(
            action=ReflectionAction.GIVE_UP,
            reason="no bars and no fallback kind to try")

    # --- NO_AXIS: maybe rotated, or maybe a chart kind without axes ---
    if status == ExtractionStatus.NO_AXIS:
        if fallback_ladder:
            # Pie/diagram don't need axes
            for k in fallback_ladder:
                if "pie" in k or "diagram" in k or "schematic" in k:
                    return ReflectionDecision(
                        action=ReflectionAction.FALLBACK_TO_NEXT_KIND,
                        reason="no axis; trying axis-free kind",
                        suggested_kind=k,
                    )
        return ReflectionDecision(
            action=ReflectionAction.GIVE_UP,
            reason="no axis, no axis-free fallback")

    # --- OCR_FAILED: retry with image upscaling? ---
    if status == ExtractionStatus.OCR_FAILED:
        if not already_retried:
            return ReflectionDecision(
                action=ReflectionAction.RETRY_WITH_PARAMS,
                reason="OCR returned nothing; will upscale image",
                retry_params={"image_scale": 2.0},
                confidence_delta=0.15,
            )
        return ReflectionDecision(
            action=ReflectionAction.GIVE_UP,
            reason="OCR still empty after retry")

    # --- UNSUPPORTED: walk fallback ladder ---
    if status == ExtractionStatus.UNSUPPORTED:
        if fallback_ladder:
            return ReflectionDecision(
                action=ReflectionAction.FALLBACK_TO_NEXT_KIND,
                reason="extractor declared UNSUPPORTED",
                suggested_kind=fallback_ladder[0],
            )
        return ReflectionDecision(
            action=ReflectionAction.GIVE_UP,
            reason="unsupported and no fallback")

    # --- ERROR: usually irrecoverable ---
    return ReflectionDecision(
        action=ReflectionAction.GIVE_UP,
        reason=f"error status: {result.reason}")
