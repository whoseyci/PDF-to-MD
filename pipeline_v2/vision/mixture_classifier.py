"""E15 — Mixture-of-specialists figure classifier.

The default classifier (``vision/classifier.py``) is a keyword scorer
over the caption text. It works when the caption explicitly names the
figure type ("Figure 3. Bar chart of yield by treatment") but fails on
short or generic captions, and it ignores the IMAGE itself.

This module adds a Mixture pattern: each ``FigureKind`` has a small
**specialist** that scores its own confidence by looking at:

  * the caption text (re-using the keyword classifier)
  * cheap image properties (color richness, line density, circular
    regions, aspect ratio, …)
  * the OCR text in the figure (axis labels are a strong bar/line
    plot signal, % signs are a pie signal, …)

A ``Summarizer`` combines per-specialist scores and emits a ranked
list. Downstream (the runner) can short-circuit on the top kind, OR
walk the list when the top specialist's extractor fails.

This is the RecursiveMAS Mixture pattern adapted to a no-GPU,
no-shared-latent-space world: specialists communicate via numeric
confidences, not hidden states. Cheap, transparent, debuggable.

API:
    from pipeline_v2.vision.mixture_classifier import (
        classify_with_mixture, MixtureResult)
    result = classify_with_mixture(
        caption="Figure 3. Yields...",
        image_path=Path("figures/fig-003.png"),
        ocr_text="Treatment  Yield (kg/ha)  Control...",
    )
    # result.top_kind, result.top_confidence, result.ranking
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import FigureKind


# ---------------------------------------------------------------------
# Per-specialist features
# ---------------------------------------------------------------------

@dataclass
class ImageFeatures:
    """Cheap pixel-level features. None = couldn't be computed."""
    has_data: bool = False
    width: int = 0
    height: int = 0
    aspect: float = 0.0           # w/h
    n_distinct_colors: int = 0    # quantised colour histogram count
    line_density: float = 0.0     # fraction of pixels that are part of detected line segments
    n_circles: int = 0            # Hough-detected circles (pie wedges / nodes)
    n_rect_components: int = 0    # rectangular CC count
    n_bar_runs: int = 0           # number of vertical-saturated-column runs (bars)
    text_pixel_frac: float = 0.0  # rough estimate of text area
    is_grayscale: bool = False


def compute_image_features(image_path: Optional[Path]) -> ImageFeatures:
    """Run quick OpenCV checks on the figure. Always returns a struct
    (with has_data=False on error)."""
    f = ImageFeatures()
    if image_path is None or not Path(image_path).exists():
        return f
    try:
        import cv2
        import numpy as np
    except ImportError:
        return f
    try:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return f
        h, w = bgr.shape[:2]
        f.width, f.height = int(w), int(h)
        f.aspect = round(w / max(1, h), 3)
        f.has_data = True

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Distinct color buckets (hue/16 × sat/64)
        h_buckets = (hsv[:, :, 0] // 16) * 4 + (hsv[:, :, 1] // 64)
        f.n_distinct_colors = int(len(np.unique(h_buckets)))

        # Grayscale-ish detection
        sat = hsv[:, :, 1]
        f.is_grayscale = bool((sat > 30).sum() < 0.05 * sat.size)

        # Line density via Canny + line-segments
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        f.line_density = round(float((edges > 0).mean()), 4)

        # Hough circles
        try:
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.4,
                minDist=int(0.1 * min(w, h)),
                param1=80, param2=40,
                minRadius=int(0.05 * min(w, h)),
                maxRadius=int(0.45 * min(w, h)),
            )
            f.n_circles = int(circles.shape[1]) if circles is not None else 0
        except Exception:
            f.n_circles = 0

        # Rectangular CCs (potential bars or node boxes)
        # Threshold dark pixels + count CCs whose aspect roughly matches a rectangle
        _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        n_comp, _, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
        rect_count = 0
        for i in range(1, n_comp):
            x, y, ww, hh, area = stats[i]
            if ww < 15 or hh < 15: continue
            if ww > 0.6 * w or hh > 0.6 * h: continue
            fill = area / max(1, ww * hh)
            if fill > 0.25:
                rect_count += 1
        f.n_rect_components = rect_count

        # Bar-column runs (saturated vertical strips)
        sat_mask = (hsv[:, :, 1] > 60).astype(np.uint8)
        col_count = sat_mask.sum(axis=0)
        thresh = max(8, 0.20 * h)
        runs = 0
        i = 0
        while i < w:
            if col_count[i] > thresh:
                j = i
                while j < w and col_count[j] > thresh:
                    j += 1
                if 6 <= (j - i) < 0.4 * w:
                    runs += 1
                i = j
            else:
                i += 1
        f.n_bar_runs = runs

        # Text-area estimate: dark fragmented regions
        dark_frac = float((gray < 100).mean())
        f.text_pixel_frac = round(dark_frac, 4)
    except Exception:
        # Best-effort; if anything blows up, just return what we have
        pass
    return f


# ---------------------------------------------------------------------
# Caption-keyword scorer (re-uses classifier.py vocab)
# ---------------------------------------------------------------------

def _caption_keyword_score(caption: str, ocr_text: str,
                            target: FigureKind) -> float:
    """Return [0..1] keyword-fitness for `target` against caption/OCR."""
    try:
        from .classifier import _KEYWORDS
    except ImportError:
        return 0.0
    if not _KEYWORDS.get(target):
        return 0.0
    text = (caption or "") + " " + (ocr_text or "")
    text = text.lower()
    hits = sum(1 for kw in _KEYWORDS[target] if kw.lower() in text)
    # Normalize: cap at ~3 hits → 1.0
    return min(1.0, hits / 3.0)


# ---------------------------------------------------------------------
# Specialists
# ---------------------------------------------------------------------

@dataclass
class SpecialistScore:
    kind: FigureKind
    score: float                # 0..1
    reason: str = ""
    components: Dict[str, float] = field(default_factory=dict)


def _scale(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def specialist_bar_chart(caption: str, ocr_text: str,
                          img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.BAR_CHART)
    bar = _scale(img.n_bar_runs, 2, 12) if img.has_data else 0.0
    has_axis = 1.0 if (ocr_text and re.search(r"\d", ocr_text)) else 0.3
    score = 0.45 * bar + 0.35 * cap + 0.20 * has_axis
    return SpecialistScore(
        kind=FigureKind.BAR_CHART, score=round(score, 3),
        reason=f"{img.n_bar_runs} vertical strips, cap={cap:.2f}",
        components={"bar_strips": bar, "caption_kw": cap,
                     "has_numeric": has_axis})


def specialist_stacked_bar(caption: str, ocr_text: str,
                            img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text,
                                   FigureKind.STACKED_BAR_CHART)
    bar = _scale(img.n_bar_runs, 2, 12) if img.has_data else 0.0
    multi_color = _scale(img.n_distinct_colors, 6, 30) if img.has_data else 0.0
    score = 0.40 * bar + 0.30 * cap + 0.30 * multi_color
    return SpecialistScore(
        kind=FigureKind.STACKED_BAR_CHART, score=round(score, 3),
        reason=f"{img.n_bar_runs} strips, {img.n_distinct_colors} colors",
        components={"bar_strips": bar, "caption_kw": cap,
                     "multi_color": multi_color})


def specialist_pie(caption: str, ocr_text: str,
                    img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.PIE_CHART)
    # A real pie chart usually has EXACTLY one big circle that
    # dominates the figure. Hough often finds spurious 1-2 small
    # circles in bar/scatter charts -- guard against that.
    if not img.has_data:
        circle_signal = 0.0
    elif img.n_circles == 1:
        circle_signal = 1.0
    elif img.n_circles == 2 and img.n_bar_runs == 0:
        circle_signal = 0.7      # maybe a donut or labeled pie
    elif img.n_circles >= 3:
        circle_signal = 0.2      # noisy detection
    else:
        circle_signal = 0.0
    # Penalise if many vertical bars are visible (definitely not a pie)
    if img.has_data and img.n_bar_runs >= 3:
        circle_signal *= 0.2
    has_pct = 1.0 if (ocr_text and "%" in ocr_text) else 0.0
    # Require either circle signal OR explicit caption -- not just OCR %
    score = 0.55 * circle_signal + 0.35 * cap + 0.10 * has_pct
    return SpecialistScore(
        kind=FigureKind.PIE_CHART, score=round(score, 3),
        reason=f"circles={img.n_circles}, bars={img.n_bar_runs}, cap={cap:.2f}",
        components={"circle_signal": circle_signal, "caption_kw": cap,
                     "has_pct": has_pct})


def specialist_line(caption: str, ocr_text: str,
                     img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.LINE_PLOT)
    # Lines: high edge density, few rectangular CCs, moderate color
    line_d = _scale(img.line_density, 0.005, 0.05) if img.has_data else 0.0
    few_rects = 1.0 - _scale(img.n_rect_components, 0, 6) if img.has_data else 0.0
    no_bars = 1.0 - _scale(img.n_bar_runs, 0, 6) if img.has_data else 0.0
    score = 0.4 * line_d + 0.25 * cap + 0.20 * few_rects + 0.15 * no_bars
    return SpecialistScore(
        kind=FigureKind.LINE_PLOT, score=round(score, 3),
        reason=f"edges={img.line_density:.3f}",
        components={"line_density": line_d, "caption_kw": cap,
                     "few_rects": few_rects, "no_bars": no_bars})


def specialist_scatter(caption: str, ocr_text: str,
                        img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.SCATTER_PLOT)
    # Scatter: many small CCs (markers), few vertical bar strips,
    # axis tick labels present in OCR.
    many_small = _scale(img.n_rect_components, 10, 100) if img.has_data else 0.0
    few_bars = 1.0 - _scale(img.n_bar_runs, 0, 4) if img.has_data else 0.0
    has_axis_nums = 0.0
    if ocr_text:
        import re as _re
        # axis ticks usually have multiple isolated short numbers
        nums = _re.findall(r"\b\d+(?:\.\d+)?\b", ocr_text)
        has_axis_nums = _scale(len(nums), 4, 20)
    score = 0.40 * many_small + 0.25 * cap + 0.20 * few_bars + 0.15 * has_axis_nums
    return SpecialistScore(
        kind=FigureKind.SCATTER_PLOT, score=round(score, 3),
        reason=f"small_ccs={img.n_rect_components}, bars={img.n_bar_runs}",
        components={"many_small": many_small, "caption_kw": cap,
                     "few_bars": few_bars,
                     "has_axis_nums": has_axis_nums})


def specialist_box(caption: str, ocr_text: str,
                    img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.BOX_PLOT)
    # Box plot: a few small rectangles roughly equally spaced
    few_boxes = _scale(img.n_rect_components, 2, 8) if img.has_data else 0.0
    score = 0.35 * few_boxes + 0.55 * cap
    return SpecialistScore(
        kind=FigureKind.BOX_PLOT, score=round(score, 3),
        reason=f"rect_ccs={img.n_rect_components}, cap={cap:.2f}",
        components={"few_boxes": few_boxes, "caption_kw": cap})


def specialist_flow_diagram(caption: str, ocr_text: str,
                              img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text,
                                   FigureKind.FLOW_DIAGRAM)
    boxes = _scale(img.n_rect_components, 3, 15) if img.has_data else 0.0
    edges_ok = _scale(img.line_density, 0.005, 0.03) if img.has_data else 0.0
    score = 0.30 * boxes + 0.40 * cap + 0.30 * edges_ok
    return SpecialistScore(
        kind=FigureKind.FLOW_DIAGRAM, score=round(score, 3),
        reason=f"rect_ccs={img.n_rect_components}, edges={img.line_density:.3f}",
        components={"boxes": boxes, "caption_kw": cap, "edges_ok": edges_ok})


def specialist_schematic(caption: str, ocr_text: str,
                          img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.SCHEMATIC)
    boxes = _scale(img.n_rect_components, 3, 15) if img.has_data else 0.0
    score = 0.40 * boxes + 0.60 * cap
    return SpecialistScore(
        kind=FigureKind.SCHEMATIC, score=round(score, 3),
        reason=f"boxes={img.n_rect_components}",
        components={"boxes": boxes, "caption_kw": cap})


def specialist_map(caption: str, ocr_text: str,
                    img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.MAP)
    many_colors = _scale(img.n_distinct_colors, 10, 80) if img.has_data else 0.0
    not_bars = 1.0 - _scale(img.n_bar_runs, 0, 4) if img.has_data else 0.0
    score = 0.55 * cap + 0.25 * many_colors + 0.20 * not_bars
    return SpecialistScore(
        kind=FigureKind.MAP, score=round(score, 3),
        reason=f"caption={cap:.2f}, colors={img.n_distinct_colors}",
        components={"caption_kw": cap, "many_colors": many_colors})


def specialist_photo(caption: str, ocr_text: str,
                      img: ImageFeatures) -> SpecialistScore:
    # Photos are hard to detect from features alone; we require an
    # explicit caption signal OR very rich colour + very low text.
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.PHOTO)
    many_colors = _scale(img.n_distinct_colors, 60, 250) if img.has_data else 0.0
    low_text = 1.0 - _scale(img.text_pixel_frac, 0.03, 0.20) if img.has_data else 0.0
    no_bars = 1.0 if img.n_bar_runs <= 1 else 0.0
    # Require both visual cues to be high to even score; this prevents
    # the photo specialist from beating chart specialists on charts.
    visual = many_colors * low_text * no_bars
    score = 0.65 * cap + 0.35 * visual
    return SpecialistScore(
        kind=FigureKind.PHOTO, score=round(score, 3),
        reason=f"colors={img.n_distinct_colors}, text_frac={img.text_pixel_frac:.3f}",
        components={"caption_kw": cap, "visual_product": visual})


def specialist_equation(caption: str, ocr_text: str,
                          img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.EQUATION)
    # Wide-and-short aspect + lots of small dark glyphs
    wide = 1.0 if (img.has_data and img.aspect > 2.5) else 0.0
    has_eq_glyph = 0.0
    if ocr_text:
        if re.search(r"[=∑∫√≤≥±·×÷]", ocr_text):
            has_eq_glyph = 1.0
    score = 0.40 * cap + 0.35 * wide + 0.25 * has_eq_glyph
    return SpecialistScore(
        kind=FigureKind.EQUATION, score=round(score, 3),
        reason=f"aspect={img.aspect}, has_math={bool(has_eq_glyph)}",
        components={"caption_kw": cap, "wide_aspect": wide,
                     "has_math_glyph": has_eq_glyph})


def specialist_table_as_image(caption: str, ocr_text: str,
                                img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text,
                                   FigureKind.TABLE_AS_IMAGE)
    # Grid-like horizontal lines, lots of digits, low color
    grid = _scale(img.line_density, 0.008, 0.04) if img.has_data else 0.0
    digit_density = 0.0
    if ocr_text:
        digits = sum(c.isdigit() for c in ocr_text)
        total = max(1, len(ocr_text))
        digit_density = _scale(digits / total, 0.05, 0.30)
    grayish = 1.0 if (img.has_data and img.is_grayscale) else 0.0
    score = 0.30 * cap + 0.30 * grid + 0.25 * digit_density + 0.15 * grayish
    return SpecialistScore(
        kind=FigureKind.TABLE_AS_IMAGE, score=round(score, 3),
        reason=f"grid={img.line_density:.3f}, digits={digit_density:.2f}",
        components={"grid_density": grid, "caption_kw": cap,
                     "digit_density": digit_density})


SPECIALISTS = [
    specialist_bar_chart,
    specialist_stacked_bar,
    specialist_pie,
    specialist_line,
    specialist_scatter,
    specialist_box,
    specialist_flow_diagram,
    specialist_schematic,
    specialist_map,
    specialist_photo,
    specialist_equation,
    specialist_table_as_image,
]


# ---------------------------------------------------------------------
# Hybrid classifier: keyword when caption is informative, mixture otherwise
# ---------------------------------------------------------------------

def classify_figure_hybrid(*,
                             caption: str = "",
                             image_path: Optional[Path] = None,
                             ocr_text: str = "",
                             keyword_trust_min_hits: int = 2,
                             ) -> "MixtureResult":
    """Smart dispatcher: use the cheap keyword classifier when the
    caption clearly names the figure type; only run the (more
    expensive, image-loading) Mixture when keywords are weak/missing.

    This is the recommended entry point for the runner. It avoids
    paying the Mixture cost (~25 ms + cv2.imread) on the 80% of
    figures where the caption is already decisive.
    """
    # Quick keyword scan — copied from classifier.py logic, kept local
    # so we don't import the full classifier module
    try:
        from .classifier import _KEYWORDS as _KW
    except ImportError:
        _KW = {}
    text = ((caption or "") + " " + (ocr_text or "")).lower()
    best_kind = FigureKind.UNKNOWN
    best_hits = 0
    for kind, kws in _KW.items():
        hits = sum(1 for kw in kws if kw.lower() in text)
        if hits > best_hits:
            best_hits = hits
            best_kind = kind

    if best_hits >= keyword_trust_min_hits:
        # Keyword classifier is confident -- skip the image work
        score = min(1.0, best_hits / 3.0)
        return MixtureResult(
            top_kind=best_kind,
            top_confidence=round(score, 3),
            top_reason=f"keyword classifier ({best_hits} kw hits)",
            margin=score,
            ranking=[SpecialistScore(kind=best_kind, score=score,
                                       reason="keyword decisive")],
            image_features=None,
            fallback_ladder=[],
        )

    # Caption weak -- pay for the full Mixture pass with image features
    return classify_with_mixture(caption=caption,
                                   image_path=image_path,
                                   ocr_text=ocr_text)


# ---------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------

@dataclass
class MixtureResult:
    top_kind: FigureKind = FigureKind.UNKNOWN
    top_confidence: float = 0.0
    top_reason: str = ""
    margin: float = 0.0           # top - 2nd
    ranking: List[SpecialistScore] = field(default_factory=list)
    image_features: Optional[ImageFeatures] = None
    fallback_ladder: List[FigureKind] = field(default_factory=list)


def classify_with_mixture(*,
                            caption: str = "",
                            image_path: Optional[Path] = None,
                            ocr_text: str = "",
                            trust_caption_threshold: float = 0.66
                            ) -> MixtureResult:
    """Run all specialists, return ranked + summary.

    **Hybrid policy:** if the caption keyword classifier scores a kind
    above ``trust_caption_threshold`` (≥2 keyword hits by default), we
    trust it as the top kind and use image features only to build the
    *fallback* ladder. This avoids the case where image-feature noise
    overrides an explicit "Figure 3: bar chart of…" caption.

    Image features only override the keyword score when the keyword
    signal is weak (no/short caption).
    """
    img = compute_image_features(image_path)
    scores = [fn(caption or "", ocr_text or "", img)
              for fn in SPECIALISTS]
    ranking = sorted(scores, key=lambda s: -s.score)

    # Compute pure caption-keyword scores in parallel
    cap_scores: List[Tuple[FigureKind, float]] = []
    for s in scores:
        cap_only = _caption_keyword_score(caption or "", ocr_text or "",
                                            s.kind)
        cap_scores.append((s.kind, cap_only))
    cap_scores.sort(key=lambda kv: -kv[1])
    cap_top_kind, cap_top_score = cap_scores[0]

    if cap_top_score >= trust_caption_threshold:
        # Caption is decisive. Use it as top; reorder ranking accordingly.
        promoted = next((s for s in ranking if s.kind == cap_top_kind), None)
        if promoted is not None:
            # Boost promoted entry to be top, preserve its actual score
            ranking = [promoted] + [s for s in ranking if s is not promoted]
        top = promoted or ranking[0]
        top_kind = cap_top_kind
        top_conf = max(top.score, cap_top_score)
        top_reason = (f"caption keyword decisive "
                        f"(kw_score={cap_top_score:.2f}); "
                        f"img features: {top.reason}")
        second = ranking[1] if len(ranking) > 1 else None
        margin = round(top_conf - (second.score if second else 0.0), 3)
    else:
        top = ranking[0]
        top_kind = top.kind if top.score >= 0.15 else FigureKind.UNKNOWN
        top_conf = top.score
        top_reason = top.reason
        second = ranking[1] if len(ranking) > 1 else None
        margin = round(top.score - (second.score if second else 0.0), 3)

    # Fallback ladder: distinct kinds whose mixture score > 0.15
    # (the ladder is for cases where the top extractor fails)
    ladder: List[FigureKind] = []
    if top_kind != FigureKind.UNKNOWN:
        ladder.append(top_kind)
    for s in ranking:
        if s.kind in ladder:
            continue
        if s.score < 0.15:
            continue
        ladder.append(s.kind)
        if len(ladder) >= 3:
            break

    return MixtureResult(
        top_kind=top_kind,
        top_confidence=round(top_conf, 3),
        top_reason=top_reason,
        margin=margin,
        ranking=ranking,
        image_features=img,
        fallback_ladder=ladder[1:],  # exclude top from ladder, it's already chosen
    )
