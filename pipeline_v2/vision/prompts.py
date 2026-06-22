"""
Per-figure-kind prompt templates.

Each prompt is short (small VLMs lose focus on long prompts) and
constrains the output format tightly so the validators downstream
can check whether the model actually produced what we asked for.

The prompts take a caption (and optionally a list of "hint" strings —
e.g. OCR text we already pulled, axis labels we know about) so the
model has the textual context to ground its description.
"""
from __future__ import annotations

from typing import List, Optional

from .base import FigureKind


# Short shared prefix that pins voice and length.
_SYSTEM_PREFIX = (
    "You are a careful scientific illustrator describing a figure from a "
    "research paper. Be concise, factual, and grounded in the image. "
    "Do not invent numeric values you cannot read."
)


def _caption_hint(caption: Optional[str]) -> str:
    if not caption:
        return ""
    return f"\nThe figure caption is: \"{caption.strip()}\""


def _ocr_hint(ocr_text: Optional[str], cap: int = 400) -> str:
    if not ocr_text:
        return ""
    snip = ocr_text.strip()
    if len(snip) > cap:
        snip = snip[:cap] + "…"
    return f"\nOCR-extracted text from the image:\n```\n{snip}\n```"


# ---------------------------------------------------------------------------
# Prompt builders, one per FigureKind
# ---------------------------------------------------------------------------

def _prompt_flow_diagram(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a workflow / process diagram or schematic with "
        f"labelled boxes connected by arrows."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "Produce ONLY a Mermaid diagram (no prose). Use `flowchart TD`. "
        "Each labelled box becomes a node; each arrow becomes an edge. "
        "Use short node IDs (A, B, C, …) and put the label in quotes. "
        "If you cannot produce a faithful diagram, output exactly the "
        "single line `MERMAID_UNAVAILABLE`."
    )


def _prompt_schematic(caption, ocr):
    # Same prompt as flow diagram; the validator decides what to keep.
    return _prompt_flow_diagram(caption, ocr)


def _prompt_bar_chart(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a bar chart."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "Step 1: in ONE short sentence, describe what the chart shows.\n"
        "Step 2: if the chart prints numeric values on each bar, "
        "reproduce them as a markdown table with columns "
        "`Category | Value (units)`. Otherwise, write the single line "
        "`TABLE_UNAVAILABLE` for the table."
    )


def _prompt_box_plot(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a box plot."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "In ONE short sentence, describe what the box plot compares and "
        "what the y-axis measures. Do NOT invent quartile values."
    )


def _prompt_line_plot(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a line plot / time series."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "In ONE short sentence, describe what is plotted on each axis "
        "and the overall trend."
    )


def _prompt_scatter_plot(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a scatter plot or ordination biplot."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "In ONE short sentence, describe the axes and what the points "
        "represent."
    )


def _prompt_pie_chart(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a pie / donut chart."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "Step 1: in ONE short sentence, describe what the chart shows.\n"
        "Step 2: if the chart prints percentages on each slice, "
        "reproduce them as a markdown table `Category | Percent`. "
        "Otherwise write `TABLE_UNAVAILABLE` for the table."
    )


def _prompt_map(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a map."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "In ONE short sentence, describe what region is shown and what "
        "the map is highlighting."
    )


def _prompt_photo(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a photograph."
        f"{_caption_hint(caption)}\n\n"
        "In ONE short sentence, describe what is visible in the photo."
    )


def _prompt_table(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is actually a table rendered as an image."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "Reproduce the table as a GitHub-flavored markdown table. "
        "Use the OCR text above as a hint for column and row labels. "
        "If you cannot reconstruct the table, output the single line "
        "`TABLE_UNAVAILABLE`."
    )


def _prompt_equation(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is a mathematical equation rendered as an image."
        f"{_caption_hint(caption)}\n\n"
        "Transcribe the equation in LaTeX (inside `$$…$$`). If you cannot, "
        "output the single line `LATEX_UNAVAILABLE`."
    )


def _prompt_data_plot(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"This figure is an effect-estimate / forest plot or similar "
        f"informative chart."
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "In ONE short sentence, describe what is being compared and "
        "what kind of estimate is plotted."
    )


def _prompt_unknown(caption, ocr):
    return (
        f"{_SYSTEM_PREFIX}\n"
        f"{_caption_hint(caption)}{_ocr_hint(ocr)}\n\n"
        "In ONE short sentence (≤ 25 words), describe what the figure "
        "shows. Do NOT speculate beyond what is visible."
    )


_DISPATCH = {
    FigureKind.FLOW_DIAGRAM: _prompt_flow_diagram,
    FigureKind.SCHEMATIC: _prompt_schematic,
    FigureKind.BAR_CHART: _prompt_bar_chart,
    FigureKind.BOX_PLOT: _prompt_box_plot,
    FigureKind.LINE_PLOT: _prompt_line_plot,
    FigureKind.SCATTER_PLOT: _prompt_scatter_plot,
    FigureKind.PIE_CHART: _prompt_pie_chart,
    FigureKind.MAP: _prompt_map,
    FigureKind.PHOTO: _prompt_photo,
    FigureKind.TABLE_AS_IMAGE: _prompt_table,
    FigureKind.EQUATION: _prompt_equation,
    FigureKind.DATA_PLOT: _prompt_data_plot,
    FigureKind.UNKNOWN: _prompt_unknown,
    # Decorative figures don't get a prompt — they skip the vision step.
}


def build_prompt(kind: FigureKind, caption: Optional[str], ocr_text: Optional[str]) -> str:
    """Build the prompt text for the given figure kind."""
    builder = _DISPATCH.get(kind, _prompt_unknown)
    return builder(caption, ocr_text)
