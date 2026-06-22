"""
Core abstractions for the vision-model harness.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class FigureKind(str, Enum):
    """High-level taxonomy used to route a figure to the right prompt."""

    BAR_CHART = "bar_chart"
    STACKED_BAR_CHART = "stacked_bar_chart"
    BOX_PLOT = "box_plot"
    LINE_PLOT = "line_plot"
    SCATTER_PLOT = "scatter_plot"
    PIE_CHART = "pie_chart"
    MAP = "map"
    PHOTO = "photo"
    FLOW_DIAGRAM = "flow_diagram"     # convertible to mermaid
    SCHEMATIC = "schematic"           # boxes, arrows, labels — also mermaid candidate
    EQUATION = "equation"             # math rendered as image
    TABLE_AS_IMAGE = "table_as_image" # figure that is actually a table
    DATA_PLOT = "data_plot"           # generic informative plot — alt text
    DECORATIVE = "decorative"         # logo / banner / icon — skip
    UNKNOWN = "unknown"


@dataclass
class FigureVisionResult:
    """
    Output of a vision-model pass over one figure.

    The result is intentionally model-agnostic: every backend produces
    the same shape so downstream code (sidecar JSON writer, paper.md
    injector) is unchanged regardless of which model ran.
    """

    figure_id: str                      # "fig-001"
    kind: FigureKind = FigureKind.UNKNOWN
    classifier_reason: str = ""         # why classify_figure chose this kind
    model_name: str = ""                # backend identifier
    prompt: str = ""                    # exact prompt sent to the model
    raw_output: str = ""                # exact text the model returned
    alt_text: Optional[str] = None      # short single-sentence description (always present on success)
    mermaid: Optional[str] = None       # filled when kind in (flow_diagram, schematic) and validation passes
    markdown_table: Optional[str] = None  # filled for table-like or bar/box/scatter charts whose data values are printed
    extracted_data: Optional[Dict[str, Any]] = None  # any structured data the model produced
    # Result of the classical (non-VLM) chart extractor, if one ran.
    # Populated for bar/box/scatter/line/pie kinds whether the
    # extractor succeeded or not — its `status` tells you what happened.
    chart_extraction: Optional[Dict[str, Any]] = None
    # Result of the VLM cross-checker (verdict ∈ {"ok","flag","skipped","error"}),
    # only present when chart_extraction succeeded AND a VLM was supplied.
    validator: Optional[Dict[str, Any]] = None
    error: Optional[str] = None         # populated on any failure
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()}


class VisionModel(ABC):
    """
    Minimal interface every vision backend must implement.

    A backend is responsible for: loading its own weights/clients on
    demand, accepting one image + one text prompt, and returning a raw
    string. All higher-level logic (classification, prompt selection,
    output validation, caching) lives in the harness.
    """

    name: str = "unknown"

    @abstractmethod
    def describe(self, image_path: Path, prompt: str, *, max_new_tokens: int = 200) -> str:
        """Run a single image+prompt → text pass. Should raise on hard errors;
        the caller is responsible for try/except + fail-safe handling."""

    # Optional: backends may override to expose preferred image size / etc.
    def preferred_max_image_dim(self) -> int:
        return 512
