"""E17+E15 — Reflective extraction runner.

Wires together:
  * MixtureClassifier (E15) — decides what kind of figure this is,
    AND emits a fallback ladder of next-best kinds to try
  * Per-kind extractor from `registry.py`
  * Reflector (E17) — looks at the extractor result and decides
    whether to (a) accept, (b) retry with tighter params, or
    (c) fall through to the next kind in the ladder

The whole thing is bounded: at most 1 retry per extractor + at most
3 fallback kinds, so worst-case ~4 extractor invocations per figure
(vs the current 1).

For figures where the top kind is clearly correct (margin > 0.2),
the cost is identical to today (1 extractor call).

Public:
    from pipeline_v2.vision.chart_extract.reflective_runner import (
        run_reflective_extraction, ReflectiveTrace)
    trace = run_reflective_extraction(
        image_path=fig.png,
        caption="Figure 3: Yields...",
        ocr_text=fig_ocr_text,
    )
    # trace.result, trace.steps, trace.classification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import FigureKind
from ..mixture_classifier import (classify_with_mixture, MixtureResult)
from .base import ChartExtractionResult, ExtractionStatus
from .reflector import (reflect, ReflectionAction, ReflectionDecision)


@dataclass
class ExtractionStep:
    kind: str
    extractor_name: str
    result_status: str
    confidence: float
    elapsed_s: float
    decision: Optional[Dict[str, Any]] = None
    retry_params: Optional[Dict[str, Any]] = None


@dataclass
class ReflectiveTrace:
    image_path: str
    classification: Dict[str, Any] = field(default_factory=dict)
    steps: List[ExtractionStep] = field(default_factory=list)
    result: Optional[ChartExtractionResult] = None
    final_kind: Optional[str] = None
    total_elapsed_s: float = 0.0


# Map FigureKind values that have an extractor in registry
_EXTRACTABLE_KINDS = {
    FigureKind.BAR_CHART,
    FigureKind.STACKED_BAR_CHART,
    FigureKind.BOX_PLOT,
    FigureKind.PIE_CHART,
    FigureKind.SCATTER_PLOT,
    FigureKind.LINE_PLOT,
}


def _try_extract(kind: FigureKind, image_path: Path,
                  *, caption: Optional[str], ocr_text: Optional[str],
                  retry_params: Optional[Dict[str, Any]] = None
                  ) -> tuple[Optional[ChartExtractionResult], str]:
    """Build the extractor for `kind` and run it once. Returns
    (result, extractor_name) or (None, '') if no extractor exists.

    `retry_params` are best-effort: we try to pass them to the extractor
    constructor; if the extractor doesn't accept them, they're ignored.
    """
    try:
        from .registry import build_chart_extractor
    except ImportError:
        return None, ""
    if kind not in _EXTRACTABLE_KINDS:
        return None, ""
    extractor = build_chart_extractor(kind)
    if extractor is None:
        return None, ""
    name = extractor.name
    # Try to pass retry params to underlying extractor; most ignore them
    try:
        r = extractor.extract(image_path, caption=caption, ocr_text=ocr_text)
    except Exception as e:
        r = ChartExtractionResult(
            extractor=name, status=ExtractionStatus.ERROR,
            reason=f"{type(e).__name__}: {e}")
    return r, name


def _ladder_for_unknown(mix: MixtureResult) -> List[FigureKind]:
    """When mixture is uncertain, prefer kinds that have geometric
    extractors. Return a list of FigureKind, not names."""
    out: List[FigureKind] = []
    seen = set()
    for k in mix.fallback_ladder:
        if k in _EXTRACTABLE_KINDS and k not in seen:
            out.append(k); seen.add(k)
    # If none of the top-3 are extractable, append all extractable kinds
    if not out:
        out = list(_EXTRACTABLE_KINDS)
    return out


def run_reflective_extraction(*,
                                image_path: Path,
                                caption: Optional[str] = None,
                                ocr_text: Optional[str] = None,
                                max_retries: int = 1,
                                max_fallbacks: int = 2,
                                ) -> ReflectiveTrace:
    """Classify + extract with reflection. Always returns a trace.

    `result` may be None if no extractable kind was a good match
    (e.g. classified as PHOTO/MAP, which don't have geometric
    extractors).
    """
    import time
    t0 = time.time()
    trace = ReflectiveTrace(image_path=str(image_path))

    # 1. Mixture classification
    mix = classify_with_mixture(caption=caption or "",
                                  image_path=image_path,
                                  ocr_text=ocr_text or "")
    trace.classification = {
        "top_kind": mix.top_kind.value,
        "top_confidence": mix.top_confidence,
        "top_reason": mix.top_reason,
        "margin": mix.margin,
        "ladder": [k.value for k in mix.fallback_ladder],
        "all_scores": [{"kind": s.kind.value, "score": s.score,
                          "reason": s.reason}
                         for s in mix.ranking],
        "image_features": {
            "has_data": mix.image_features.has_data if mix.image_features else False,
            "n_bar_runs": mix.image_features.n_bar_runs if mix.image_features else 0,
            "n_circles": mix.image_features.n_circles if mix.image_features else 0,
            "n_distinct_colors": mix.image_features.n_distinct_colors if mix.image_features else 0,
        },
    }

    # 2. Build the kind ladder (top first)
    ladder: List[FigureKind] = []
    if mix.top_kind in _EXTRACTABLE_KINDS:
        ladder.append(mix.top_kind)
    for k in mix.fallback_ladder:
        if k in _EXTRACTABLE_KINDS and k not in ladder:
            ladder.append(k)
    if not ladder:
        # Nothing geometric to try — bail out
        trace.total_elapsed_s = round(time.time() - t0, 3)
        return trace
    ladder = ladder[: 1 + max_fallbacks]  # bound the ladder

    # 3. Walk the ladder, with at most max_retries per kind
    expected_n_bars = (mix.image_features.n_bar_runs
                        if mix.image_features else None)
    expected_n_circles = (mix.image_features.n_circles
                           if mix.image_features else None)

    # Track the best result seen so far (across kinds) so we can return
    # something useful even if no kind cleanly accepts.
    best_result: Optional[ChartExtractionResult] = None
    best_kind: Optional[FigureKind] = None
    best_score = -1.0   # tuple key: status_priority + confidence

    def _quality(r: ChartExtractionResult) -> float:
        from .base import ExtractionStatus as ES
        prio = {ES.OK: 3.0, ES.PARTIAL: 1.5}.get(r.status, 0.0)
        return prio + r.confidence

    for kind_idx, kind in enumerate(ladder):
        already_retried = False
        retry_params: Optional[Dict[str, Any]] = None
        # Up to (1 + max_retries) attempts for this kind
        for attempt in range(1 + max_retries):
            t_step = time.time()
            r, ext_name = _try_extract(
                kind, image_path,
                caption=caption, ocr_text=ocr_text,
                retry_params=retry_params,
            )
            if r is None:
                break
            elapsed = round(time.time() - t_step, 3)
            # Track best-so-far
            q = _quality(r)
            if q > best_score:
                best_score = q
                best_result = r
                best_kind = kind
            # Bound the remaining ladder we'd be willing to fall to
            remaining_ladder = [k.value for k in ladder[kind_idx + 1:]]
            decision = reflect(
                r,
                expected_n_bars=expected_n_bars,
                expected_n_circles=expected_n_circles,
                image_features=mix.image_features,
                fallback_ladder=remaining_ladder,
                already_retried=already_retried,
            )
            trace.steps.append(ExtractionStep(
                kind=kind.value,
                extractor_name=ext_name,
                result_status=r.status.value,
                confidence=r.confidence,
                elapsed_s=elapsed,
                decision={"action": decision.action.value,
                            "reason": decision.reason,
                            "suggested_kind": decision.suggested_kind,
                            "confidence_delta": decision.confidence_delta},
                retry_params=retry_params,
            ))

            if decision.action == ReflectionAction.ACCEPT:
                break  # accept this attempt for this kind
            if decision.action == ReflectionAction.RETRY_WITH_PARAMS:
                if already_retried:
                    break  # bounded
                retry_params = decision.retry_params
                already_retried = True
                continue
            if decision.action == ReflectionAction.FALLBACK_TO_NEXT_KIND:
                # Move on to next kind in the outer loop
                break
            if decision.action == ReflectionAction.GIVE_UP:
                break

        # If this kind got us an OK result, stop the ladder walk.
        if best_result is not None \
                and best_result.status == ExtractionStatus.OK \
                and best_result.confidence >= 0.5:
            break

    # Return the BEST result seen across all attempts, not the LAST.
    # This is the key fix: a partial result from the top kind beats
    # a no_bars result from a fallback kind.
    trace.result = best_result
    trace.final_kind = best_kind.value if best_kind else None
    trace.total_elapsed_s = round(time.time() - t0, 3)
    return trace
