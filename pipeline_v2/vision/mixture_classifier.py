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
    # Universal "is this a chart" gate -- 2 perpendicular thick straight
    # lines suggest axes. Used as a hard prior to suppress chart
    # specialists on schematics/photos/maps.
    has_chart_axes: bool = False
    h_axis_length_frac: float = 0.0  # longest horizontal black line / image width
    v_axis_length_frac: float = 0.0  # longest vertical black line / image height
    # Round 3 features (Sep 2026 specialist polish)
    n_panels: int = 1             # if multipanel detected, number of panels (1=single)
    panel_layout: str = ""        # "1x1", "2x1", "1x2", "2x2", "3x3", "Nx?" etc
    is_decorative: bool = False   # tiny image OR very thin banner OR very monochrome
    decorative_reason: str = ""
    has_math_glyphs: bool = False # equation/formula likely
    has_legend_box: bool = False  # detected boxed legend (small rect with text near top-right)
    n_tick_marks_h: int = 0       # short perpendicular ticks on the horizontal axis
    n_tick_marks_v: int = 0
    edge_pixels_at_border: float = 0.0  # frame-detection: dark pixels along the outer edge


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

        # Chart-axis detection. Charts have two long perpendicular
        # near-black straight lines. Schematics, photos, maps don't.
        # We use morphological opening with a long horizontal/vertical
        # kernel to highlight axis lines specifically.
        try:
            dark = (gray < 120).astype(np.uint8) * 255
            # First: strip a thin border (1-3 px) so a page frame
            # doesn't dominate. The plot axis is usually a few px
            # inside the image edge.
            BORDER = max(3, min(w, h) // 50)
            inner_mask = np.zeros_like(dark)
            inner_mask[BORDER:-BORDER, BORDER:-BORDER] = 1
            dark_inner = dark * inner_mask
            kx = max(20, int(0.30 * w))
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
            h_lines = cv2.morphologyEx(dark_inner, cv2.MORPH_OPEN, h_kernel)
            ky = max(20, int(0.30 * h))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
            v_lines = cv2.morphologyEx(dark_inner, cv2.MORPH_OPEN, v_kernel)
            h_per_row = h_lines.sum(axis=1) / 255
            v_per_col = v_lines.sum(axis=0) / 255
            longest_h = float(h_per_row.max()) / max(1, w)
            longest_v = float(v_per_col.max()) / max(1, h)
            f.h_axis_length_frac = round(longest_h, 3)
            f.v_axis_length_frac = round(longest_v, 3)
            # Axes test: BOTH a long horizontal AND vertical line.
            if longest_h >= 0.25 and longest_v >= 0.25:
                # Find ALL strong horizontal lines and ALL strong
                # vertical lines, not just the longest. The plot's
                # actual axis (with ticks) may not be the longest row
                # (e.g., a boxed plot has 4 spines of equal length;
                # the bottom one is the x-axis).
                strong_threshold = longest_h * 0.85 * w
                candidate_rows = [int(r) for r in
                                    np.where(h_per_row >= strong_threshold)[0]]
                strong_threshold_v = longest_v * 0.85 * h
                candidate_cols = [int(c) for c in
                                    np.where(v_per_col >= strong_threshold_v)[0]]
                # Plausible position
                candidate_rows = [r for r in candidate_rows
                                    if (h * 0.05) < r < (h * 0.95)]
                candidate_cols = [c for c in candidate_cols
                                    if (w * 0.05) < c < (w * 0.95)]
                # Use full (non-masked) dark mask for tick counting
                # so the inner-border erosion doesn't kill faint ticks.
                dark_full = (gray < 130).astype(np.uint8) * 255
                # For each candidate H-line, count ticks BELOW it.
                # The real x-axis usually has ticks just below.
                best_h_ticks = 0
                for r in candidate_rows:
                    # Strip BELOW this row (max 15 px), avoid going off-img
                    end = min(h, r + 15)
                    if end - r < 3: continue
                    strip = dark_full[r + 1:end, :]
                    col_active = strip.sum(axis=0) > 0
                    ticks = int(np.sum(np.diff(col_active.astype(int)) == 1))
                    if ticks > best_h_ticks:
                        best_h_ticks = ticks
                f.n_tick_marks_h = best_h_ticks
                # For each candidate V-line, count ticks LEFT of it.
                best_v_ticks = 0
                for c in candidate_cols:
                    end = max(0, c - 15)
                    if c - end < 3: continue
                    strip = dark_full[:, end:c]
                    row_active = strip.sum(axis=1) > 0
                    ticks = int(np.sum(np.diff(row_active.astype(int)) == 1))
                    if ticks > best_v_ticks:
                        best_v_ticks = ticks
                f.n_tick_marks_v = best_v_ticks
                # Real chart axes have many tick marks. Tighten the
                # threshold to filter out spurious "border" matches.
                if (f.n_tick_marks_h + f.n_tick_marks_v) >= 4:
                    f.has_chart_axes = True
        except Exception:
            pass

        # Border / frame detection (a thin dark outer rectangle).
        # Many "figures" are just decorative banners with a dark border.
        try:
            border_px = (gray[:3, :].mean() < 200) + \
                        (gray[-3:, :].mean() < 200) + \
                        (gray[:, :3].mean() < 200) + \
                        (gray[:, -3:].mean() < 200)
            f.edge_pixels_at_border = float(border_px) / 4
        except Exception:
            pass

        # Math-glyph detection from a quick Tesseract pass.
        # (We don't always have ocr_text available at feature time --
        # this is best-effort; skip silently on failure.)
        try:
            import pytesseract
            from PIL import Image
            txt = pytesseract.image_to_string(Image.fromarray(gray),
                                                 config="--psm 6")
            if re.search(r"[=∑∫√≤≥±·×÷∞αβγδλμπρσφω]|\\frac|\\sqrt", txt):
                f.has_math_glyphs = True
        except Exception:
            pass

        # Decorative detection: very thin banner, OR
        # tiny absolute size, OR uniform color with NO structure.
        # "Structure" = chart axes OR enough rect components OR text.
        try:
            very_thin = (h < 40 or w < 40)
            very_wide_banner = (f.aspect > 8 or f.aspect < 0.125)
            n_pixels = w * h
            tiny = n_pixels < 6000  # <80x75
            very_monochrome = f.n_distinct_colors < 4
            # Has structure = something a real figure would have
            has_structure = (f.has_chart_axes
                              or f.n_rect_components >= 3
                              or f.n_bar_runs >= 2
                              or f.n_circles >= 1
                              or f.text_pixel_frac > 0.04)
            if very_thin:
                f.is_decorative = True
                f.decorative_reason = f"very thin ({w}x{h})"
            elif tiny:
                f.is_decorative = True
                f.decorative_reason = f"tiny ({w}x{h})"
            elif very_wide_banner and not has_structure:
                f.is_decorative = True
                f.decorative_reason = f"banner-aspect {f.aspect}, no structure"
            elif very_monochrome and not has_structure:
                f.is_decorative = True
                f.decorative_reason = (
                    f"monochrome ({f.n_distinct_colors} colors), no structure")
        except Exception:
            pass

        # Multipanel detection: a chart figure often has 2x2/3x3 sub-plots.
        # Real multipanel splits have a CLEAR white gap PLUS content
        # blocks on both sides. We require:
        #   1. The gap is at least 2% of the image dim wide (not just
        #      a single blank row)
        #   2. Both sides of the gap have substantial dark content
        # Plus we look at the *interior* of the image, not the
        # whole-image row mean (which a blank-background plot would
        # also satisfy).
        try:
            BORDER_P = max(10, int(0.05 * min(w, h)))
            interior = gray[BORDER_P:-BORDER_P, BORDER_P:-BORDER_P]
            ih, iw = interior.shape
            row_mean_i = interior.mean(axis=1)
            col_mean_i = interior.mean(axis=0)
            # A "white" row in a panel grid is mean > 245 (very white)
            white_rows = row_mean_i > 245
            white_cols = col_mean_i > 245
            def _strong_gaps(white_arr, dim, min_gap_frac=0.02):
                """Find white runs of length >= min_gap_frac * dim that
                are NOT at the start/end (those are margins not gaps)."""
                n = len(white_arr)
                min_gap = max(4, int(min_gap_frac * dim))
                # Drop edge runs (first 15% and last 15%)
                edge = int(0.15 * n)
                gaps = []
                i = edge
                while i < n - edge:
                    if white_arr[i]:
                        j = i
                        while j < n and white_arr[j]:
                            j += 1
                        if (j - i) >= min_gap:
                            # Verify both sides have content
                            left_dark = (~white_arr[max(0, i - 20):i]).sum()
                            right_dark = (~white_arr[j:j + 20]).sum()
                            if left_dark >= 5 and right_dark >= 5:
                                gaps.append((i + j) // 2)
                        i = j
                    else:
                        i += 1
                # Merge gaps closer than 15% of dim
                merged = []
                for g in gaps:
                    if merged and g - merged[-1] < n * 0.15:
                        continue
                    merged.append(g)
                return merged
            # Use a higher min_gap_frac for serious multipanel
            # detection (real subplot gaps are >8% of dim).
            h_gaps = _strong_gaps(white_rows, ih, min_gap_frac=0.08)
            v_gaps = _strong_gaps(white_cols, iw, min_gap_frac=0.08)
            n_rows = 1 + len(h_gaps)
            n_cols = 1 + len(v_gaps)
            # Real multipanel REQUIRES the gap to look like a true
            # row-of-panels split: BOTH dimensions must split (a 2x2
            # or 3x3 grid). One-dimensional "splits" are bar/scatter
            # gaps, not panels.
            if n_rows >= 2 and n_cols >= 2:
                n_rows = min(n_rows, 4)
                n_cols = min(n_cols, 4)
                f.n_panels = n_rows * n_cols
                f.panel_layout = f"{n_rows}x{n_cols}"
            else:
                f.panel_layout = "1x1"
        except Exception:
            f.panel_layout = "1x1"

        # Legend-box detection: a small bordered rect in top-right
        # corner with text in it.
        try:
            top_right = gray[: int(h * 0.40), int(w * 0.55):]
            tr_dark = (top_right < 130).astype(np.uint8) * 255
            ny, ncc, st, _ = cv2.connectedComponentsWithStats(tr_dark, 8)
            for i in range(1, ncc):
                x, y, ww, hh, area = st[i]
                if 0.06 * top_right.size < area < 0.40 * top_right.size:
                    aspect = ww / max(1, hh)
                    if 0.5 < aspect < 4:
                        # Hollow-ish? sample interior darkness
                        interior_mean = top_right[
                            y + 2:y + hh - 2, x + 2:x + ww - 2].mean()
                        if 100 < interior_mean < 230:
                            f.has_legend_box = True
                            break
        except Exception:
            pass
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
    # Hard chart-axis gate: bar charts MUST have axes. If no axes
    # detected AND no caption signal, score plummets.
    axis_prior = 1.0 if (img.has_data and img.has_chart_axes) else 0.0
    if not img.has_data:
        axis_prior = 0.5  # unknown -- don't penalise
    score = 0.30 * bar + 0.30 * cap + 0.15 * has_axis + 0.25 * axis_prior
    return SpecialistScore(
        kind=FigureKind.BAR_CHART, score=round(score, 3),
        reason=f"{img.n_bar_runs} strips, axes={img.has_chart_axes}, cap={cap:.2f}",
        components={"bar_strips": bar, "caption_kw": cap,
                     "has_numeric": has_axis, "axis_prior": axis_prior})


def specialist_stacked_bar(caption: str, ocr_text: str,
                            img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text,
                                   FigureKind.STACKED_BAR_CHART)
    bar = _scale(img.n_bar_runs, 2, 12) if img.has_data else 0.0
    multi_color = _scale(img.n_distinct_colors, 6, 30) if img.has_data else 0.0
    axis_prior = 1.0 if (img.has_data and img.has_chart_axes) else 0.0
    if not img.has_data: axis_prior = 0.5
    score = 0.30 * bar + 0.25 * cap + 0.20 * multi_color + 0.25 * axis_prior
    return SpecialistScore(
        kind=FigureKind.STACKED_BAR_CHART, score=round(score, 3),
        reason=f"{img.n_bar_runs} strips, {img.n_distinct_colors} colors, axes={img.has_chart_axes}",
        components={"bar_strips": bar, "caption_kw": cap,
                     "multi_color": multi_color,
                     "axis_prior": axis_prior})


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
    axis_prior = 1.0 if (img.has_data and img.has_chart_axes) else 0.0
    if not img.has_data: axis_prior = 0.5
    score = (0.25 * line_d + 0.20 * cap + 0.15 * few_rects
               + 0.10 * no_bars + 0.30 * axis_prior)
    return SpecialistScore(
        kind=FigureKind.LINE_PLOT, score=round(score, 3),
        reason=f"edges={img.line_density:.3f}, axes={img.has_chart_axes}",
        components={"line_density": line_d, "caption_kw": cap,
                     "few_rects": few_rects, "no_bars": no_bars,
                     "axis_prior": axis_prior})


def specialist_scatter(caption: str, ocr_text: str,
                        img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.SCATTER_PLOT)
    many_small = _scale(img.n_rect_components, 10, 100) if img.has_data else 0.0
    few_bars = 1.0 - _scale(img.n_bar_runs, 0, 4) if img.has_data else 0.0
    has_axis_nums = 0.0
    if ocr_text:
        import re as _re
        nums = _re.findall(r"\b\d+(?:\.\d+)?\b", ocr_text)
        has_axis_nums = _scale(len(nums), 4, 20)
    axis_prior = 1.0 if (img.has_data and img.has_chart_axes) else 0.0
    if not img.has_data: axis_prior = 0.5
    score = (0.30 * many_small + 0.20 * cap + 0.15 * few_bars
               + 0.10 * has_axis_nums + 0.25 * axis_prior)
    return SpecialistScore(
        kind=FigureKind.SCATTER_PLOT, score=round(score, 3),
        reason=f"small_ccs={img.n_rect_components}, axes={img.has_chart_axes}",
        components={"many_small": many_small, "caption_kw": cap,
                     "few_bars": few_bars,
                     "has_axis_nums": has_axis_nums,
                     "axis_prior": axis_prior})


def specialist_box(caption: str, ocr_text: str,
                    img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.BOX_PLOT)
    few_boxes = _scale(img.n_rect_components, 2, 8) if img.has_data else 0.0
    axis_prior = 1.0 if (img.has_data and img.has_chart_axes) else 0.0
    if not img.has_data: axis_prior = 0.5
    score = 0.25 * few_boxes + 0.45 * cap + 0.30 * axis_prior
    return SpecialistScore(
        kind=FigureKind.BOX_PLOT, score=round(score, 3),
        reason=f"rect_ccs={img.n_rect_components}, axes={img.has_chart_axes}, cap={cap:.2f}",
        components={"few_boxes": few_boxes, "caption_kw": cap,
                     "axis_prior": axis_prior})


def specialist_flow_diagram(caption: str, ocr_text: str,
                              img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text,
                                   FigureKind.FLOW_DIAGRAM)
    boxes = _scale(img.n_rect_components, 3, 15) if img.has_data else 0.0
    edges_ok = _scale(img.line_density, 0.005, 0.03) if img.has_data else 0.0
    # Diagrams DON'T have plot axes. Penalise if axes are present.
    no_axis_prior = 1.0 if (img.has_data and not img.has_chart_axes) else 0.0
    if not img.has_data: no_axis_prior = 0.5
    score = (0.25 * boxes + 0.35 * cap + 0.20 * edges_ok
               + 0.20 * no_axis_prior)
    return SpecialistScore(
        kind=FigureKind.FLOW_DIAGRAM, score=round(score, 3),
        reason=f"rect_ccs={img.n_rect_components}, axes={img.has_chart_axes}",
        components={"boxes": boxes, "caption_kw": cap,
                     "edges_ok": edges_ok, "no_axis_prior": no_axis_prior})


def specialist_schematic(caption: str, ocr_text: str,
                          img: ImageFeatures) -> SpecialistScore:
    cap = _caption_keyword_score(caption, ocr_text, FigureKind.SCHEMATIC)
    boxes = _scale(img.n_rect_components, 3, 15) if img.has_data else 0.0
    no_axis_prior = 1.0 if (img.has_data and not img.has_chart_axes) else 0.0
    if not img.has_data: no_axis_prior = 0.5
    score = 0.30 * boxes + 0.50 * cap + 0.20 * no_axis_prior
    return SpecialistScore(
        kind=FigureKind.SCHEMATIC, score=round(score, 3),
        reason=f"boxes={img.n_rect_components}, axes={img.has_chart_axes}",
        components={"boxes": boxes, "caption_kw": cap,
                     "no_axis_prior": no_axis_prior})


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
    wide = 1.0 if (img.has_data and img.aspect > 2.5) else 0.0
    # OCR-text-side detection (passed in by caller)
    has_eq_glyph_ocr = 0.0
    if ocr_text:
        if re.search(r"[=∑∫√≤≥±·×÷]", ocr_text):
            has_eq_glyph_ocr = 1.0
    # Image-feature-side detection (from feature-time tesseract pass)
    has_eq_glyph_img = 1.0 if (img.has_data and img.has_math_glyphs) else 0.0
    # Either signal is fine
    has_eq_glyph = max(has_eq_glyph_ocr, has_eq_glyph_img)
    # Equations don't have plot axes
    no_axis_prior = 1.0 if (img.has_data and not img.has_chart_axes) else 0.0
    if not img.has_data: no_axis_prior = 0.5
    score = (0.30 * cap + 0.25 * wide + 0.30 * has_eq_glyph
               + 0.15 * no_axis_prior)
    return SpecialistScore(
        kind=FigureKind.EQUATION, score=round(score, 3),
        reason=f"aspect={img.aspect}, has_math={bool(has_eq_glyph)}, axes={img.has_chart_axes}",
        components={"caption_kw": cap, "wide_aspect": wide,
                     "has_math_glyph": has_eq_glyph,
                     "no_axis_prior": no_axis_prior})


def specialist_decorative(caption: str, ocr_text: str,
                           img: ImageFeatures) -> SpecialistScore:
    """Detect images that aren't really figures: page banners, logos,
    section dividers, decorative borders. These figures shouldn't
    waste extractor time."""
    if not img.has_data:
        # Without features we can't decide -- low confidence
        return SpecialistScore(
            kind=FigureKind.DECORATIVE, score=0.0,
            reason="no image data")
    feature_score = 1.0 if img.is_decorative else 0.0
    cap = _caption_keyword_score(caption, ocr_text,
                                   FigureKind.DECORATIVE)
    # An informative caption should SUPPRESS the decorative score
    # (real figures have captions).
    caption_present = 1.0 if (caption and len(caption.split()) >= 4) else 0.0
    # If caption present, halve the score: decorative figures
    # rarely have proper captions.
    base = 0.70 * feature_score + 0.30 * cap
    if caption_present:
        base *= 0.4
    return SpecialistScore(
        kind=FigureKind.DECORATIVE, score=round(base, 3),
        reason=img.decorative_reason or "no decorative signal",
        components={"feature_signal": feature_score,
                     "caption_kw": cap,
                     "caption_suppresses": caption_present})


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
    specialist_decorative,
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
