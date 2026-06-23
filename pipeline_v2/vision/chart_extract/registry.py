"""FigureKind -> ChartExtractor.

By default chart kinds dispatch to the geometric extractors wrapped
in `MultiPanelExtractor`. If you want DePlot as a backup / vote
partner, build a cascade explicitly:

    from pipeline_v2.vision.chart_extract import (
        build_chart_extractor, DeplotExtractor)

    bar = build_chart_extractor(FigureKind.BAR_CHART,
                                  with_deplot=True,
                                  deplot_shared=DeplotExtractor())
"""
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
from .multi_extractor import CascadingExtractor

try:
    from .deplot import DeplotExtractor
    _DEPLOT_AVAILABLE = True
except ImportError:
    _DEPLOT_AVAILABLE = False


def build_chart_extractor(kind: FigureKind, *,
                            with_deplot: bool = False,
                            deplot_shared=None) -> Optional[ChartExtractor]:
    """Build the ChartExtractor for a given FigureKind.

    If `with_deplot` is True and `deplot_shared` is provided (a single
    DeplotExtractor instance to avoid reloading the model), the geometric
    extractor is wrapped in a cascade that falls through to DePlot when
    the geometric pass returns low confidence.
    """
    base = {
        FigureKind.BAR_CHART:         SimpleBarsExtractor(),
        FigureKind.STACKED_BAR_CHART: StackedBarsExtractor(),
        FigureKind.BOX_PLOT:          BoxPlotExtractor(),
        FigureKind.PIE_CHART:         PieChartExtractor(),
        FigureKind.SCATTER_PLOT:      ScatterExtractor(),
        FigureKind.LINE_PLOT:         LinePlotExtractor(),
    }.get(kind)
    if base is None:
        return None
    if with_deplot and _DEPLOT_AVAILABLE:
        deplot = deplot_shared if deplot_shared is not None \
            else DeplotExtractor()
        base = CascadingExtractor([base, deplot])
    return MultiPanelExtractor(base)


_REGISTRY: Dict[FigureKind, ChartExtractor] = {
    k: build_chart_extractor(k)
    for k in (FigureKind.BAR_CHART, FigureKind.STACKED_BAR_CHART,
                FigureKind.BOX_PLOT, FigureKind.PIE_CHART,
                FigureKind.SCATTER_PLOT, FigureKind.LINE_PLOT)
}


def get_extractor(kind: FigureKind) -> Optional[ChartExtractor]:
    return _REGISTRY.get(kind)


def available_extractors() -> Dict[str, str]:
    out = {k.value: v.name for k, v in _REGISTRY.items()}
    out["_deplot_available"] = str(_DEPLOT_AVAILABLE)
    return out
