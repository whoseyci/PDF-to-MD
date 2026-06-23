"""FigureKind -> ChartExtractor.

The default registry cascades the fast geometric extractors with
**DePlot as a fallback** (when transformers is available and the
google/deplot weights are downloaded). This means:

  * `bar_chart` → SimpleBars (fast, ~0.5s) → DePlot fallback if PARTIAL
  * `stacked_bar_chart` / `box_plot` / `pie_chart` / `scatter_plot` /
    `line_plot` → stub (returns UNSUPPORTED) → DePlot fallback
    (40-110s per figure)

DePlot is opt-out via the `PDF2MD_DISABLE_DEPLOT=1` environment
variable in case you really want geometric-only (e.g. on a host
that can't afford the 1.5GB RAM during inference). Without the
fallback, all the non-bar chart kinds return UNSUPPORTED and the
runner falls back to Gemma 4 (16+ minutes per figure -- much
slower than DePlot).

DePlot is also lazy-loaded: the model only loads when first
called, so importing this module is cheap even with the cascade
enabled.
"""
from __future__ import annotations
import os
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

# DePlot is optional; we lazy-load and gracefully degrade if
# unavailable. Set PDF2MD_DISABLE_DEPLOT=1 to skip even checking.
_DISABLE_DEPLOT = os.environ.get("PDF2MD_DISABLE_DEPLOT", "0") == "1"
try:
    if not _DISABLE_DEPLOT:
        from .deplot_subprocess import DeplotSubprocessExtractor
        _DEPLOT_AVAILABLE = True
    else:
        _DEPLOT_AVAILABLE = False
except ImportError:
    _DEPLOT_AVAILABLE = False


def build_chart_extractor(kind: FigureKind, *,
                            with_deplot: Optional[bool] = None,
                            ) -> Optional[ChartExtractor]:
    """Build the ChartExtractor for a given FigureKind.

    ``with_deplot``:
      * ``None`` (default) → use DePlot if available
      * ``True``           → require DePlot (raises if unavailable)
      * ``False``          → never use DePlot
    """
    geometric = {
        FigureKind.BAR_CHART:         SimpleBarsExtractor(),
        FigureKind.STACKED_BAR_CHART: StackedBarsExtractor(),
        FigureKind.BOX_PLOT:          BoxPlotExtractor(),
        FigureKind.PIE_CHART:         PieChartExtractor(),
        FigureKind.SCATTER_PLOT:      ScatterExtractor(),
        FigureKind.LINE_PLOT:         LinePlotExtractor(),
    }.get(kind)
    if geometric is None:
        return None

    use_deplot = with_deplot
    if use_deplot is None:
        use_deplot = _DEPLOT_AVAILABLE
    if use_deplot and not _DEPLOT_AVAILABLE:
        if with_deplot is True:
            raise RuntimeError(
                "DePlot requested but not importable. Install transformers "
                "and download google/deplot, OR pass with_deplot=False.")
        use_deplot = False
    if use_deplot:
        # Subprocess-isolated DePlot for memory safety on 2GB hosts.
        # The cascade only invokes DePlot when the geometric extractor
        # returns < OK confidence, so plain bar charts cost 0 model loads.
        deplot = DeplotSubprocessExtractor(
            per_image_timeout=300, max_image_dim=320, max_new_tokens=200)
        geometric = CascadingExtractor([geometric, deplot])
    return MultiPanelExtractor(geometric)


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
