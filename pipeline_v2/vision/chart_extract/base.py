"""Base types for chart extractors."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExtractionStatus(str, Enum):
    OK = "ok"; PARTIAL = "partial"; UNSUPPORTED = "unsupported"
    NO_AXIS = "no_axis"; NO_BARS = "no_bars"; OCR_FAILED = "ocr_failed"
    ERROR = "error"


@dataclass
class ChartExtractionResult:
    extractor: str = ""
    status: ExtractionStatus = ExtractionStatus.ERROR
    reason: str = ""
    orientation: Optional[str] = None
    value_axis: Optional[str] = None
    value_label: Optional[str] = None
    category_label: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    series: List[str] = field(default_factory=list)
    matrix: List[List[float]] = field(default_factory=list)
    box_stats: Optional[List[Dict[str, Any]]] = None
    pie_slices: Optional[List[Dict[str, Any]]] = None
    scatter_summary: Optional[List[Dict[str, Any]]] = None
    line_series: Optional[List[Dict[str, Any]]] = None
    extracted_data: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    plot_area: Optional[List[int]] = None
    bar_boxes: List[List[int]] = field(default_factory=list)
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self); d["status"] = self.status.value; return d

    def to_markdown_table(self) -> Optional[str]:
        if self.status in (ExtractionStatus.NO_AXIS, ExtractionStatus.NO_BARS,
                            ExtractionStatus.UNSUPPORTED, ExtractionStatus.ERROR,
                            ExtractionStatus.OCR_FAILED):
            return None
        if self.extracted_data and isinstance(self.extracted_data, dict) \
                and self.extracted_data.get("per_panel_markdown"):
            return self.extracted_data["per_panel_markdown"]
        if self.box_stats and self.categories and \
                len(self.box_stats) == len(self.categories):
            vh = self.value_label or "Value"
            ch = self.category_label or "Category"
            lines = [f"| {ch} | min | Q1 | median | Q3 | max |",
                     "|---|---|---|---|---|---|"]
            for cat, st in zip(self.categories, self.box_stats):
                lines.append(f"| {cat} | {_fmt(st.get('min'))} | {_fmt(st.get('q1'))} "
                              f"| {_fmt(st.get('median'))} | {_fmt(st.get('q3'))} "
                              f"| {_fmt(st.get('max'))} |")
            return "\n".join(lines)
        if self.pie_slices:
            lines = ["| Slice | Fraction | Percent |", "|---|---|---|"]
            for s in self.pie_slices:
                pct = s.get('fraction', 0) * 100 if s.get('fraction') is not None else None
                lines.append(f"| {s.get('label','?')} | {_fmt(s.get('fraction'))} "
                              f"| {_fmt(pct)}% |")
            return "\n".join(lines)
        if self.scatter_summary:
            lines = ["| Series | N points | X min..max (mean) | Y min..max (mean) |",
                     "|---|---|---|---|"]
            for s in self.scatter_summary:
                lines.append(f"| {s.get('series','?')} | {s.get('n_points',0)} "
                              f"| {_fmt(s.get('x_min'))}..{_fmt(s.get('x_max'))} "
                              f"({_fmt(s.get('x_mean'))}) "
                              f"| {_fmt(s.get('y_min'))}..{_fmt(s.get('y_max'))} "
                              f"({_fmt(s.get('y_mean'))}) |")
            return "\n".join(lines)
        if self.line_series:
            lines = []
            for s in self.line_series:
                pts = s.get("points", [])
                if not pts: continue
                lines.append(f"**{s.get('series','?')}** ({len(pts)} sampled points)")
                lines.append("| X | Y |"); lines.append("|---|---|")
                for x, y in pts:
                    lines.append(f"| {_fmt(x)} | {_fmt(y)} |")
                lines.append("")
            return "\n".join(lines).rstrip() or None
        if self.matrix and self.series:
            header = [self.category_label or "Category", *self.series]
            lines = ["| " + " | ".join(header) + " |",
                     "|" + "|".join(["---"] * len(header)) + "|"]
            for i, cat in enumerate(self.categories):
                row = [cat] + [_fmt(self.matrix[i][j]) for j in range(len(self.series))]
                lines.append("| " + " | ".join(row) + " |")
            return "\n".join(lines)
        if self.categories and self.values and \
                len(self.categories) == len(self.values):
            vh = self.value_label or "Value"
            ch = self.category_label or "Category"
            lines = [f"| {ch} | {vh} |", "|---|---|"]
            for c, v in zip(self.categories, self.values):
                lines.append(f"| {c} | {_fmt(v)} |")
            return "\n".join(lines)
        return None


def _fmt(v) -> str:
    if v is None: return ""
    try: v = float(v)
    except (TypeError, ValueError): return str(v)
    if abs(v - round(v)) < 1e-9: return str(int(round(v)))
    return f"{v:.3g}"


class ChartExtractor(ABC):
    name: str = "abstract"

    @abstractmethod
    def extract(self, image_path: Path, *,
                caption: Optional[str] = None,
                ocr_text: Optional[str] = None) -> ChartExtractionResult: ...
