"""
Per-figure-subtype VLM prompts, inspired by VikParuchuri/marker's
`marker/processors/llm/llm_image_description.py`.

The original `prompts.py` in this package already handles charts /
diagrams / maps via the classifier. This module adds richer prompts
for the long-tail subtypes that the classifier returns `UNKNOWN`
for, but which a VLM can still produce something useful for:

  * `algorithm`        --> Markdown pseudocode in a fenced block
  * `code_listing`     --> language-tagged fenced code block
  * `equation`         --> LaTeX in $$...$$
  * `screenshot`       --> alt-text plus optional Mermaid for a UI flow
  * `microscopy`       --> caption-style description (no measurement)
  * `gel_blot`         --> caption-style with band positions
  * `decision_tree`    --> Mermaid
  * `sankey`           --> Mermaid (sankey or flowchart)

The picker `prompt_for_caption()` uses keyword detection on the caption
to choose between these. If the caption gives no clue we fall back to
the existing generic alt-text prompt.

These prompts are designed for Gemma 4 E2B (our local VLM) -- they're
short and front-load the directive so the model can answer before
running out of thinking tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FigurePrompt:
    """A picked prompt plus the kind we matched on."""
    subkind: str
    prompt: str
    expected_format: str  # "markdown_table" | "mermaid" | "latex" |
                            # "code" | "pseudocode" | "alt_text"


# --------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------

_GENERIC_ALT = """\
In ONE short sentence, describe what this scientific figure shows.
If it is a chart, mention the chart type and what each axis shows.
"""

_ALGORITHM = """\
This figure is an algorithm / pseudocode block from a research paper.
Transcribe it as Markdown pseudocode inside a fenced code block
labeled ``` text. Preserve indentation. Do not add explanations.
"""

_CODE = """\
This figure is a code listing. Transcribe it as a fenced code block.
Add the language tag after the opening backticks (e.g. ```python).
Preserve indentation exactly. Do not add explanations.
"""

_EQUATION = """\
This figure is a mathematical equation. Transcribe it as LaTeX
inside `$$ ... $$` delimiters. Use standard LaTeX commands
(\\frac, \\sum, \\int, \\alpha, etc.). Do not add explanations.
"""

_SCREENSHOT = """\
This figure is a screenshot of a user interface or web page. In ONE
short paragraph (≤ 3 sentences), describe what the user sees and the
key UI elements visible. Do not invent text you cannot read.
"""

_MICROSCOPY = """\
This figure is a microscopy / imaging image (electron, fluorescence,
histology, or similar). In ONE short caption-style sentence, describe
what the image shows: subject, technique if visible (scale bar,
staining), and any obvious structures. Do NOT estimate measurements.
"""

_GEL_BLOT = """\
This figure is a gel image, western blot, or similar electrophoretic
analysis. In ONE short caption-style sentence, list the visible lanes
(left to right) and any clearly labeled band sizes. Do NOT invent
band positions you cannot see.
"""

_DECISION_TREE = """\
Transcribe this decision tree as Mermaid. Reply with ONLY a fenced
mermaid code block — no prose. Use:

```mermaid
flowchart TD
    A{Question?} -->|Yes| B[Outcome 1]
    A -->|No| C[Outcome 2]
```

If you cannot read the tree, reply with exactly: UNREADABLE
"""

_SANKEY = """\
This figure is a Sankey diagram showing flow between categories.
Transcribe it as Mermaid:

```mermaid
sankey-beta
SourceA,TargetA,quantity
SourceB,TargetB,quantity
```

Reply with ONLY the fenced block — no prose. If you cannot read the
flows, reply with exactly: UNREADABLE
"""


# --------------------------------------------------------------------
# Caption-based picker
# --------------------------------------------------------------------

# Each rule: (compiled regex, FigurePrompt). First-match wins.
_RULES = [
    (re.compile(r"\b(algorithm|pseudo[- ]?code)\b", re.IGNORECASE),
     lambda: FigurePrompt("algorithm", _ALGORITHM, "pseudocode")),
    # NOTE: order matters -- gel/blot must come BEFORE equation so
    # "Western blot showing protein expression" doesn't match
    # "expression" first.
    (re.compile(r"\b(gel|blot|western|northern|southern|"
                 r"electrophore\w+|PAGE)\b", re.IGNORECASE),
     lambda: FigurePrompt("gel_blot", _GEL_BLOT, "alt_text")),
    (re.compile(r"\b(code\s+(listing|snippet|sample)|source\s+code|"
                 r"function\s+definition|class\s+definition|"
                 r"listing\s+\d+|^listing\b)\b",
                 re.IGNORECASE),
     lambda: FigurePrompt("code_listing", _CODE, "code")),
    (re.compile(r"\b(equation|formula|mathematical\s+expression)\b",
                 re.IGNORECASE),
     lambda: FigurePrompt("equation", _EQUATION, "latex")),
    (re.compile(r"\b(decision\s+tree|classification\s+tree)\b",
                 re.IGNORECASE),
     lambda: FigurePrompt("decision_tree", _DECISION_TREE, "mermaid")),
    (re.compile(r"\bsankey\b", re.IGNORECASE),
     lambda: FigurePrompt("sankey", _SANKEY, "mermaid")),
    (re.compile(r"\b(screenshot|user\s+interface|UI\b|web\s+page)\b",
                 re.IGNORECASE),
     lambda: FigurePrompt("screenshot", _SCREENSHOT, "alt_text")),
    (re.compile(r"\b(micrograph|microscopy|histolog\w+|fluoresce\w+|"
                 r"SEM|TEM|H&E)\b", re.IGNORECASE),
     lambda: FigurePrompt("microscopy", _MICROSCOPY, "alt_text")),
]


def prompt_for_caption(caption: Optional[str]) -> FigurePrompt:
    """Pick the best prompt for a figure based on its caption text.

    Returns the generic alt-text prompt if no rule matches.
    """
    if caption:
        for regex, factory in _RULES:
            if regex.search(caption):
                return factory()
    return FigurePrompt("generic", _GENERIC_ALT, "alt_text")


# --------------------------------------------------------------------
# Output post-processing for each expected_format
# --------------------------------------------------------------------

def postprocess_response(raw: str, expected_format: str) -> str:
    """Strip preamble / explanation from the model's response.

    Light handling: trims any text BEFORE the first fenced block when
    we expected code / mermaid / latex, since the model often prefaces
    with ``Here is the code:``.
    """
    if not raw:
        return raw
    raw = raw.strip()
    if expected_format in ("code", "pseudocode", "mermaid"):
        # Capture the FIRST complete fenced block: opening ``` (with
        # optional language tag) ... closing ```.
        m = re.search(r"```[A-Za-z0-9_]*\s*\n.*?\n```",
                       raw, re.DOTALL)
        if m:
            raw = m.group(0)
    elif expected_format == "latex":
        # Wrap stray latex in $$ if model omitted them
        if "$$" not in raw and not raw.startswith("$"):
            raw = "$$ " + raw.strip().rstrip("$") + " $$"
    return raw.strip()
