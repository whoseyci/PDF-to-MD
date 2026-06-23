"""Classical chart-extraction tools (geometric, no LLM) + optional DePlot."""
from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus
from .registry import (
    get_extractor, available_extractors, build_chart_extractor,
)
from .multipanel import MultiPanelExtractor
from .multi_extractor import CascadingExtractor
from .simple_bars import SimpleBarsExtractor

try:
    from .deplot import DeplotExtractor  # noqa: F401
    _has_deplot = True
except ImportError:
    _has_deplot = False

__all__ = ["ChartExtractor", "ChartExtractionResult", "ExtractionStatus",
           "get_extractor", "available_extractors", "build_chart_extractor",
           "MultiPanelExtractor", "CascadingExtractor", "SimpleBarsExtractor"]
if _has_deplot:
    __all__.append("DeplotExtractor")
