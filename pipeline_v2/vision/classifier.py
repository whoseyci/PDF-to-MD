"""
Caption-text-based figure classifier.

Given a figure's caption (and optionally its OCR text), guess what kind
of figure it is. The classification drives prompt selection in the
runner: we ask the vision model very different questions of a flow
diagram, a bar chart, or a map.

This classifier is **deterministic and offline** — no LLM. It uses a
keyword scoring table over the caption text. When the caption is
empty or ambiguous, it falls back to ``FigureKind.UNKNOWN`` and the
runner uses a generic "describe in one sentence" prompt.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Tuple

from .base import FigureKind


# Per-kind keyword sets. The classifier scores each kind by counting
# how many of its keywords appear in the caption/OCR text, then picks
# the highest-scoring kind. Ties resolve in declaration order.

_KEYWORDS = {
    FigureKind.FLOW_DIAGRAM: [
        "workflow", "flowchart", "flow chart", "flow diagram", "pipeline",
        "process diagram", "process flow", "step", "scheme", "schema",
        "framework", "decision tree",
    ],
    FigureKind.SCHEMATIC: [
        "schematic", "diagram of", "conceptual model", "concept map",
        "structure of", "architecture", "model structure", "components",
        "overview of the", "illustration of",
    ],
    FigureKind.MAP: [
        "map of", "map showing", "geographic", "study area", "study site",
        "study location", "location of", "site map", "aerial", "satellite",
        "spatial distribution", "region", "country", "watershed", "basin",
    ],
    FigureKind.BAR_CHART: [
        "bar chart", "bar plot", "bar graph",
        "histogram", "frequency", "proportion of", "proportions of",
        "percentage of", "mean of", "average", "colonies per",
        "kg per", "mg per", "ha-1", "yield",
        "log-transformed", "log10",
    ],
    FigureKind.STACKED_BAR_CHART: [
        "stacked bar", "stacked-bar", "stacked column",
        "land use", "land cover", "cover type", "composition",
        "proportion of", "share of", "breakdown of",
    ],
    FigureKind.BOX_PLOT: [
        "box plot", "boxplot", "box-and-whisker", "whisker",
        "quartile", "median", "interquartile",
    ],
    FigureKind.LINE_PLOT: [
        "line plot", "line graph", "time series", "trend",
        "over time", "over the years", "annual", "monthly",
        "evolution", "trajectory",
    ],
    FigureKind.SCATTER_PLOT: [
        "scatter", "scatterplot", "regression", "correlation",
        "ordination", "rda", "pca", "nmds", "biplot",
        "non-metric multidimensional", "principal component",
    ],
    FigureKind.PIE_CHART: [
        "pie chart", "pie", "donut", "doughnut",
        "share of", "composition",
    ],
    FigureKind.PHOTO: [
        "photograph", "photo of", "image of", "view of",
        "picture of", "field photograph", "aerial photo",
        "ground photograph",
    ],
    FigureKind.EQUATION: [
        "equation", "formula", "mathematical expression",
    ],
    FigureKind.TABLE_AS_IMAGE: [
        "table", "matrix",
    ],
    FigureKind.DATA_PLOT: [
        "effect estimate", "estimate of", "ci", "confidence interval",
        "error bar",
    ],
}


# A small set of phrases that, if found, ALMOST GUARANTEE a particular
# kind (used to break ties). Each tuple is (regex, kind, reason).
_STRONG_HINTS = [
    (re.compile(r"\b(workflow|flowchart|pipeline)\b", re.IGNORECASE),
     FigureKind.FLOW_DIAGRAM, "strong-hint:workflow/flowchart/pipeline"),
    (re.compile(r"\bbox\s*plots?\b|\bbox-and-whisker\b", re.IGNORECASE),
     FigureKind.BOX_PLOT, "strong-hint:boxplot"),
    (re.compile(r"\bnmds\b|\brda\d?\b|\bpca\b|\bordination\b", re.IGNORECASE),
     FigureKind.SCATTER_PLOT, "strong-hint:ordination"),
    (re.compile(r"\bmap\s+(of|showing)\b|\bstudy\s+area\b|\baerial\s+photo", re.IGNORECASE),
     FigureKind.MAP, "strong-hint:map/study area"),
    (re.compile(r"\bphotograph\b|\bphoto\s+of\b", re.IGNORECASE),
     FigureKind.PHOTO, "strong-hint:photograph"),
    (re.compile(r"\b(stacked|grouped)\s+bar\b", re.IGNORECASE),
     FigureKind.STACKED_BAR_CHART, "strong-hint:stacked/grouped bar"),
    (re.compile(r"\bland\s+(use|cover)\s+composition\b|\bcomposition\s+by\b",
                re.IGNORECASE),
     FigureKind.STACKED_BAR_CHART, "strong-hint:composition"),
]


def classify_figure(
    caption: Optional[str],
    ocr_text: Optional[str] = None,
) -> Tuple[FigureKind, str]:
    """
    Classify a figure based on its caption text (and OCR fallback).

    Returns
    -------
    (kind, reason)
        ``kind`` is the chosen `FigureKind`.
        ``reason`` is a short human-readable string explaining WHY this
        kind was chosen — useful for debugging the classifier and for
        logging in the sidecar JSON.
    """
    haystack_parts = []
    if caption:
        haystack_parts.append(caption)
    if ocr_text:
        # OCR is noisy — only use it for classification when there's no caption
        if not caption:
            haystack_parts.append(ocr_text)
    haystack = " ".join(haystack_parts).lower().strip()
    if not haystack:
        return FigureKind.UNKNOWN, "no-caption-no-ocr"

    # 1) Strong hints win immediately
    for pattern, kind, reason in _STRONG_HINTS:
        if pattern.search(haystack):
            return kind, reason

    # 2) Keyword scoring
    scores: Counter = Counter()
    matched_keywords: dict[FigureKind, list[str]] = {}
    for kind, kws in _KEYWORDS.items():
        hits = [kw for kw in kws if kw in haystack]
        if hits:
            scores[kind] = len(hits)
            matched_keywords[kind] = hits

    if not scores:
        return FigureKind.UNKNOWN, "no-keyword-match"

    best_kind, best_score = scores.most_common(1)[0]
    return best_kind, f"keyword-match:{','.join(matched_keywords[best_kind][:3])}"
