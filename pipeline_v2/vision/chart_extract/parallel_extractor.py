"""ParallelExtractor -- run every chart extractor against every figure
and arbitrate by quality, not by classifier prediction.

Why this exists
---------------
The classifier is unreliable on weak/no-caption figures (~37%
accuracy in our stress bench). The reflective runner walks a
kind-ladder that depends on the classifier; when the classifier is
wrong, the right kind is buried in the ladder and we pay 2x
extraction time just to find it.

The ParallelExtractor flips the architecture:

  1. Run EVERY enabled extractor on the figure (bar, stacked, pie,
     line, scatter, box, diagram, equation). Each runs independently.
  2. Each extractor self-reports a status (OK, PARTIAL, NO_AXIS,
     NO_BARS, OCR_FAILED, UNSUPPORTED, ERROR). The status IS the
     self-rejection signal -- a bar extractor knows when it sees no
     bars. We've tightened self-rejection in simple_bars, pie_chart,
     scatter_plot, stacked_bars so they don't false-positive each
     other's input.
  3. Arbitrate by quality score = status_priority * confidence.
     OK_with_conf_0.95 always beats PARTIAL_with_conf_0.4 regardless
     of what the classifier said.
  4. Cross-check: if multiple extractors return OK at close quality,
     prefer the one whose kind matches the classifier hint (mild
     tiebreaker only). If ALL extractors failed, prefer the classifier
     hint over the arbitrary first-failed result.

Cost: ~N extractors per figure instead of 1-4. On 8 chart extractors
that's bounded -- each runs in 0.05-1.5s on CPU, so a full sweep is
~4-6s per figure regardless of classifier accuracy.

This buys reliability: an extractor that gets a figure it's not
specialised for self-rejects via its status code, and the system
picks the one extractor that actually succeeded.

Public API
----------
  run_parallel_extraction(image_path, caption, ocr_text,
                            classifier_hint, ...) -> ParallelExtractionTrace
  run_smart_extraction(image_path, caption, ocr_text)
      -> uses reflective path when keyword classifier is decisive,
         else falls back to parallel-all with classifier hint as
         tiebreaker. Saves ~30% time on caption-rich corpora.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import FigureKind
from .base import ChartExtractionResult, ExtractionStatus


# Status priority weights (higher = better self-reported quality)
_STATUS_PRIORITY = {
    ExtractionStatus.OK:           4.0,
    ExtractionStatus.PARTIAL:      2.0,
    ExtractionStatus.NO_BARS:      0.3,
    ExtractionStatus.NO_AXIS:      0.3,
    ExtractionStatus.OCR_FAILED:   0.1,
    ExtractionStatus.UNSUPPORTED:  0.0,
    ExtractionStatus.ERROR:        0.0,
}


def _structural_credibility(r: ChartExtractionResult) -> float:
    """[-1..+1] bonus/penalty based on structural quality of the
    extracted data.

    Penalises extractors that returned 'OK' with trivially small
    or implausible output (1 bar, 1 box, all medians equal, etc).
    Rewards extractors that returned coherent multi-item data.
    """
    bonus = 0.0
    # Bar chart: >=3 bars is healthy, 1-2 is suspect
    if r.bar_boxes:
        n = len(r.bar_boxes)
        if n >= 3: bonus += 0.3
        elif n <= 1: bonus -= 0.5
    # Pie: >=3 wedges plausible
    if r.pie_slices:
        n = len(r.pie_slices)
        if n >= 3: bonus += 0.3
        elif n < 2: bonus -= 0.5
        # Check wedge angle sum is ~360°
        total_pct = sum(s.get("percent", 0) for s in r.pie_slices)
        if 95 <= total_pct <= 105:
            bonus += 0.2
        else:
            bonus -= 0.3
    # Box: medians should vary; whiskers must extend beyond box
    if r.box_stats:
        n = len(r.box_stats)
        if n >= 3: bonus += 0.2
        medians = [b.get("median", 0) for b in r.box_stats]
        # If medians are very close together (cluster), penalise
        if medians:
            med_spread = max(medians) - min(medians)
            med_mag = max(abs(m) for m in medians) or 1.0
            rel_spread = med_spread / med_mag
            if rel_spread < 0.15:
                bonus -= 0.6  # near-identical medians = fake boxes
            elif rel_spread >= 0.50:
                bonus += 0.3  # well-spread medians = real boxes
        # Whisker check: in a real box plot, whisker_low STRICTLY
        # less than q1 AND whisker_high STRICTLY greater than q3.
        # A "bar chart misread as box plot" has q1 == whisker_low
        # (the bar bottom IS the whisker bottom) -- no whiskers
        # extending beyond the rectangle.
        n_real_whiskers = 0
        for b in r.box_stats:
            wl = b.get("whisker_low", 0); q1 = b.get("q1", 0)
            q3 = b.get("q3", 0); wh = b.get("whisker_high", 0)
            box_size = abs(q3 - q1) or 1.0
            top_extension = abs(wh - q3) / box_size
            bot_extension = abs(q1 - wl) / box_size
            # Real boxplot whisker extends by >= 20% of box size
            if top_extension > 0.20 or bot_extension > 0.20:
                n_real_whiskers += 1
        if n_real_whiskers >= max(1, n // 2):
            bonus += 0.3
        else:
            bonus -= 0.6  # no real whiskers = not a box plot
    # Line: at least 1 series with >=4 samples
    if r.line_series:
        if any(len(s.get("samples", [])) >= 4 for s in r.line_series):
            bonus += 0.2
        else:
            bonus -= 0.3
    # Scatter: total markers should be substantial; clusters should
    # have spread across both x and y axes.
    if r.scatter_summary:
        total_markers = sum(c.get("n_points", 0) for c in r.scatter_summary)
        if total_markers >= 15:
            bonus += 0.4
        elif total_markers >= 5:
            bonus += 0.2
        # Reward genuine 2D spread: x_max - x_min AND y_max - y_min
        # should both be > 0 for at least one cluster
        for c in r.scatter_summary:
            x_span = c.get("x_max", 0) - c.get("x_min", 0)
            y_span = c.get("y_max", 0) - c.get("y_min", 0)
            if x_span > 0 and y_span > 0:
                bonus += 0.1
                break
        # Penalty if all clusters have ~same xy mean (no spread)
        means = [(round(c.get("x_mean", 0), 2),
                   round(c.get("y_mean", 0), 2))
                 for c in r.scatter_summary]
        if len(set(means)) < len(means):
            bonus -= 0.3
    # Stacked: matrix needs >= 2 rows AND >= 2 cols of non-zero values
    if r.matrix:
        n_nonzero_cells = sum(1 for row in r.matrix for v in row if v)
        if len(r.matrix) >= 2 and n_nonzero_cells >= 4:
            bonus += 0.3
        else:
            bonus -= 0.3
    return max(-1.0, min(1.0, bonus))


def _quality_score(r: ChartExtractionResult) -> float:
    """Map a result to a quality score for arbitration.

    Combines self-reported status (+confidence) with structural
    credibility (how plausible the extracted data actually is).
    Range roughly [-1..6].
    """
    prio = _STATUS_PRIORITY.get(r.status, 0.0)
    base = prio + min(1.0, r.confidence or 0.0)
    # Only apply structural bonus/penalty to results that claim OK
    # or PARTIAL (others have no data to evaluate)
    if r.status in (ExtractionStatus.OK, ExtractionStatus.PARTIAL):
        return base + _structural_credibility(r)
    return base


@dataclass
class ParallelExtractionTrace:
    image_path: str
    n_extractors_run: int = 0
    classifier_hint: Optional[str] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[ChartExtractionResult] = None
    winner_extractor: Optional[str] = None
    winner_kind: Optional[str] = None
    arbitration_reason: str = ""
    total_elapsed_s: float = 0.0
    decision_log: List[str] = field(default_factory=list)


_DEFAULT_ENABLED_KINDS: List[FigureKind] = [
    FigureKind.BAR_CHART,
    FigureKind.STACKED_BAR_CHART,
    FigureKind.BOX_PLOT,
    FigureKind.PIE_CHART,
    FigureKind.SCATTER_PLOT,
    FigureKind.LINE_PLOT,
    FigureKind.FLOW_DIAGRAM,
    FigureKind.EQUATION,
]


def _try_one(kind: FigureKind, image_path: Path,
              caption: Optional[str], ocr_text: Optional[str]
              ) -> Tuple[Optional[ChartExtractionResult], str]:
    """Adapter that knows about all three extractor families
    (chart_extract registry, diagram_extract, equation_extract)."""
    from .reflective_runner import _try_extract
    return _try_extract(kind, image_path,
                         caption=caption, ocr_text=ocr_text)


def run_parallel_extraction(*,
                              image_path: Path,
                              caption: Optional[str] = None,
                              ocr_text: Optional[str] = None,
                              classifier_hint: Optional[FigureKind] = None,
                              enabled_kinds: Optional[List[FigureKind]] = None,
                              auto_ocr: bool = True,
                              decorative_skip: bool = True,
                              early_stop_on_high_conf: float = 0.85,
                              ) -> ParallelExtractionTrace:
    """Run every enabled extractor; pick the best by quality score."""
    t0 = time.time()
    trace = ParallelExtractionTrace(
        image_path=str(image_path),
        classifier_hint=(classifier_hint.value
                          if classifier_hint else None))

    # 0. Auto-OCR
    if auto_ocr and (ocr_text is None or not ocr_text.strip()):
        try:
            import pytesseract
            from PIL import Image
            ocr_text = pytesseract.image_to_string(Image.open(image_path))
            trace.decision_log.append(f"auto-ocr: {len(ocr_text)} chars")
        except Exception as e:
            ocr_text = ocr_text or ""
            trace.decision_log.append(f"auto-ocr failed: {e}")

    # 1. Decorative shortcut
    if decorative_skip and classifier_hint == FigureKind.DECORATIVE:
        trace.winner = ChartExtractionResult(
            extractor="mixture/decorative",
            status=ExtractionStatus.UNSUPPORTED,
            reason="classified as decorative",
            confidence=0.0,
        )
        trace.winner_extractor = "mixture/decorative"
        trace.winner_kind = "decorative"
        trace.arbitration_reason = "decorative skip"
        trace.total_elapsed_s = round(time.time() - t0, 3)
        return trace

    # 2. Order extractors -- classifier hint first if provided
    enabled = list(enabled_kinds or _DEFAULT_ENABLED_KINDS)
    if classifier_hint and classifier_hint in enabled:
        enabled.remove(classifier_hint)
        enabled.insert(0, classifier_hint)
        trace.decision_log.append(
            f"classifier hint placed {classifier_hint.value} first")

    # 3. Run extractors in order. Store full results so we don't
    # have to re-run the winner.
    results_by_kind: Dict[str, Tuple[ChartExtractionResult, str]] = {}
    for kind in enabled:
        t_step = time.time()
        result, ext_name = _try_one(
            kind, image_path,
            caption=caption, ocr_text=ocr_text)
        elapsed = round(time.time() - t_step, 3)
        if result is None:
            trace.decision_log.append(
                f"{kind.value}: no extractor wired")
            continue
        q = _quality_score(result)
        trace.candidates.append({
            "kind": kind.value, "extractor": ext_name,
            "status": result.status.value,
            "confidence": result.confidence,
            "quality": round(q, 3),
            "reason": result.reason,
            "elapsed_s": elapsed,
        })
        results_by_kind[kind.value] = (result, ext_name)
        trace.n_extractors_run += 1
        # Early stop: if a HINTED extractor returns high-conf OK
        if (classifier_hint is not None
                and kind == classifier_hint
                and result.status == ExtractionStatus.OK
                and (result.confidence or 0.0) >= early_stop_on_high_conf):
            trace.decision_log.append(
                f"early-stop: hinted {kind.value} ok at "
                f"conf={result.confidence}")
            break

    # 4. Arbitrate
    if not trace.candidates:
        trace.winner = ChartExtractionResult(
            extractor="parallel/none",
            status=ExtractionStatus.UNSUPPORTED,
            reason="no extractor returned a result",
        )
        trace.arbitration_reason = "no candidates"
        trace.total_elapsed_s = round(time.time() - t0, 3)
        return trace

    ranked = sorted(trace.candidates, key=lambda c: -c["quality"])
    top = ranked[0]
    # If everyone failed (top quality < 0.5), prefer the classifier
    # hint over the arbitrary first-failed result.
    if classifier_hint and top["quality"] < 0.5:
        hint_str = classifier_hint.value
        for c in ranked:
            if c["kind"] == hint_str:
                if c is not top:
                    trace.arbitration_reason = (
                        f"all-failed: chose {hint_str} (classifier hint) "
                        f"over {top['kind']} (both q<0.5)")
                    top = c
                break
    # Close-call tiebreaker
    elif (len(ranked) >= 2 and classifier_hint
            and (top["quality"] - ranked[1]["quality"]) < 0.3):
        hint_str = classifier_hint.value
        for c in ranked[:3]:
            if c["kind"] == hint_str:
                if c is not top:
                    trace.arbitration_reason = (
                        f"close-call: chose {hint_str} (classifier hint) "
                        f"over {top['kind']} (q={top['quality']} vs "
                        f"{c['quality']})")
                    top = c
                break

    # Recover full winner result from cache (no re-run)
    cached = results_by_kind.get(top["kind"])
    if cached is not None:
        trace.winner, trace.winner_extractor = cached
        trace.winner_kind = top["kind"]
    if not trace.arbitration_reason:
        trace.arbitration_reason = (
            f"highest quality ({top['quality']}) was "
            f"{top['kind']}/{top['status']}")

    trace.total_elapsed_s = round(time.time() - t0, 3)
    return trace


def run_smart_extraction(*,
                          image_path: Path,
                          caption: Optional[str] = None,
                          ocr_text: Optional[str] = None,
                          ) -> ParallelExtractionTrace:
    """Best of both worlds: trust the keyword classifier when the
    caption is informative; fall back to parallel-all when it's not.

    Decision rule:
      * Compute keyword classifier score on caption
      * If keyword classifier scores >= 2 hits, run the reflective
        path (cheap, ~1.7s)
      * Otherwise (no caption / weak caption), run all extractors in
        parallel with the image-based Mixture top as tiebreaker
        (more expensive ~4-5s but reliable)

    Bench (PARALLEL_REPORT.md): smart matches parallel-hinted at
    88.2% accuracy on the synthetic stress bench but is ~12% faster
    on average because caption-decisive cases skip the full sweep.
    """
    # Quick keyword hit count
    try:
        from ..classifier import _KEYWORDS as _KW
    except ImportError:
        _KW = {}
    text = ((caption or "") + " " + (ocr_text or "")).lower()
    best_hits = 0
    best_kind: Optional[FigureKind] = None
    for kind, kws in _KW.items():
        hits = sum(1 for kw in kws if kw.lower() in text)
        if hits > best_hits:
            best_hits = hits; best_kind = kind

    # Strong caption -> reflective path (cheap, classifier-trusted)
    if best_hits >= 2 and best_kind is not None:
        from .reflective_runner import run_reflective_extraction
        refl = run_reflective_extraction(
            image_path=image_path, caption=caption, ocr_text=ocr_text)
        out = ParallelExtractionTrace(
            image_path=str(image_path),
            n_extractors_run=len(refl.steps),
            classifier_hint=best_kind.value,
            winner=refl.result,
            winner_extractor=(refl.result.extractor
                              if refl.result else None),
            winner_kind=refl.final_kind,
            arbitration_reason=(
                f"caption-decisive: {best_hits} keyword hits for "
                f"{best_kind.value}; used reflective path"),
            total_elapsed_s=refl.total_elapsed_s,
        )
        for s in refl.steps:
            out.candidates.append({
                "kind": s.kind, "extractor": s.extractor_name,
                "status": s.result_status,
                "confidence": s.confidence,
                "quality": (_quality_score(refl.result)
                             if (refl.result and s.kind == refl.final_kind)
                             else 0.0),
                "reason": "",
                "elapsed_s": s.elapsed_s,
            })
        return out

    # Weak/no caption -> parallel-all with mixture hint as tiebreaker
    from ..mixture_classifier import classify_with_mixture
    try:
        mix = classify_with_mixture(caption=caption,
                                      image_path=image_path,
                                      ocr_text=ocr_text)
        hint = (mix.top_kind
                if mix.top_kind != FigureKind.UNKNOWN else None)
    except Exception:
        hint = None
    return run_parallel_extraction(
        image_path=image_path, caption=caption, ocr_text=ocr_text,
        classifier_hint=hint,
    )
