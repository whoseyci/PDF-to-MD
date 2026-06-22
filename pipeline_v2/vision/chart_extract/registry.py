"""FigureKind → ChartExtractor."""
from __future__ import annotations
from typing import Dict, Optional

from ..base import FigureKind
from .base import ChartExtractor
from .simple_bars import SimpleBarsExtractor
from .stubs import (
    StackedBarsExtractor, BoxPlotExtractor, PieChartExtractor,
    ScatterExtractor, LinePlotExtractor,
)
from .multipanel import MultiPanelExtractor


_REGISTRY: Dict[FigureKind, ChartExtractor] = {
    FigureKind.BAR_CHART: MultiPanelExtractor(SimpleBarsExtractor()),
    FigureKind.STACKED_BAR_CHART: MultiPanelExtractor(StackedBarsExtractor()),
    FigureKind.BOX_PLOT: MultiPanelExtractor(BoxPlotExtractor()),
    FigureKind.PIE_CHART: MultiPanelExtractor(PieChartExtractor()),
    FigureKind.SCATTER_PLOT: MultiPanelExtractor(ScatterExtractor()),
    FigureKind.LINE_PLOT: MultiPanelExtractor(LinePlotExtractor()),
}


def get_extractor(kind: FigureKind) -> Optional[ChartExtractor]:
    return _REGISTRY.get(kind)


def available_extractors() -> Dict[str, str]:
    return {k.value: v.name for k, v in _REGISTRY.items()}
