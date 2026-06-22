"""Placeholder extractors that return UNSUPPORTED.

The full geometric implementations (stacked_bars, box_plot, pie_chart,
scatter, line_plot) lived here in the previous session. They're not
re-emitted in this slimmer build to keep the package under the
workspace snapshot cap. The registry routes those kinds to these
stubs, which signal UNSUPPORTED so the runner cleanly falls back to
the Gemma 4 VLM path.

To restore the full implementations, see OPTIMIZATION_NOTES.md
section (k) which documents what each extractor did and how to
rebuild them.
"""
from __future__ import annotations
from pathlib import Path

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus


class _UnsupportedExtractor(ChartExtractor):
    why = "no classical extractor implemented yet"

    def extract(self, image_path: Path, *, caption=None, ocr_text=None):
        return ChartExtractionResult(
            extractor=self.name, status=ExtractionStatus.UNSUPPORTED,
            reason=self.why, confidence=0.0,
        )


class StackedBarsExtractor(_UnsupportedExtractor):
    name = "stacked_bars/stub"
    why = "stacked bars stub (full impl in OPTIMIZATION_NOTES history)"


class BoxPlotExtractor(_UnsupportedExtractor):
    name = "box_plot/stub"
    why = "box plot stub (full impl in OPTIMIZATION_NOTES history)"


class PieChartExtractor(_UnsupportedExtractor):
    name = "pie_chart/stub"
    why = "pie chart stub (full impl in OPTIMIZATION_NOTES history)"


class ScatterExtractor(_UnsupportedExtractor):
    name = "scatter/stub"
    why = "scatter stub (full impl in OPTIMIZATION_NOTES history)"


class LinePlotExtractor(_UnsupportedExtractor):
    name = "line_plot/stub"
    why = "line plot stub (full impl in OPTIMIZATION_NOTES history)"
