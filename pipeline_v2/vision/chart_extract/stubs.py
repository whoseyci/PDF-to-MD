"""Stubs / compat shims.

E8 (Jun 2026): full geometric implementations exist now in
``stacked_bars.py``, ``box_plot.py``, ``pie_chart.py``,
``scatter_plot.py`` and ``line_plot.py``. We re-export them from
this module so any older imports (e.g. ``from .stubs import
PieChartExtractor``) still work, and we keep an
``_UnsupportedExtractor`` available for kinds that genuinely have no
geometric extractor (e.g. heatmaps).
"""
from __future__ import annotations
from pathlib import Path

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus
from .stacked_bars import StackedBarsExtractor  # noqa: F401
from .box_plot import BoxPlotExtractor  # noqa: F401
from .pie_chart import PieChartExtractor  # noqa: F401
from .scatter_plot import ScatterExtractor  # noqa: F401
from .line_plot import LinePlotExtractor  # noqa: F401


class _UnsupportedExtractor(ChartExtractor):
    why = "no classical extractor implemented yet"

    def extract(self, image_path: Path, *, caption=None, ocr_text=None):
        return ChartExtractionResult(
            extractor=self.name, status=ExtractionStatus.UNSUPPORTED,
            reason=self.why, confidence=0.0,
        )


class HeatmapExtractor(_UnsupportedExtractor):
    name = "heatmap/stub"
    why = "heatmap geometric extractor not implemented"
