"""
Diagram-to-Mermaid extraction via Gemma 4 E2B.

For figures that are conceptual diagrams (theory-of-planned-behaviour
models, causal loop diagrams, flow charts, schematic models, etc.),
we ask Gemma 4 to read the boxes + arrows + labels and emit a
syntactically-valid Mermaid graph. Mermaid renders natively in GitHub
Markdown, MkDocs, Obsidian, Notion, etc., so the resulting paper.md
is much more useful than alt-text alone.

Usage:

    from pipeline_v2.vision.factory import make_model
    from pipeline_v2.vision.mermaid_extract import MermaidExtractor

    m = MermaidExtractor(make_model("gemma4-e2b"))
    result = m.extract(image_path, caption="Theory of Planned Behaviour")
    print(result.mermaid)        # ```mermaid ... ``` block
    print(result.confidence)     # 0..1
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .base import VisionModel


_MERMAID_PROMPT = """\
Transcribe this diagram as Mermaid. Reply with ONLY a fenced mermaid
code block — no thinking, no prose, no preamble.

```mermaid
flowchart LR
    A[label] --> B[label]
```
"""


@dataclass
class MermaidExtractionResult:
    mermaid: Optional[str] = None        # the fenced ```mermaid ... ``` block
    raw_output: str = ""
    nodes: List[str] = None
    edges: List[str] = None
    confidence: float = 0.0
    reason: str = ""
    elapsed_seconds: float = 0.0

    def __post_init__(self):
        if self.nodes is None: self.nodes = []
        if self.edges is None: self.edges = []


class MermaidExtractor:
    """Pulls a Mermaid diagram out of a figure using a vision model."""

    def __init__(self, model: VisionModel, *,
                 max_new_tokens: int = 400):
        self.model = model
        self.max_new_tokens = int(max_new_tokens)

    def extract(self, image_path: Path, *,
                caption: Optional[str] = None) -> MermaidExtractionResult:
        t0 = time.time()
        prompt = _MERMAID_PROMPT
        if caption:
            prompt += f"\nFigure caption (for context only): {caption.strip()!r}\n"
        try:
            raw = self.model.describe(image_path, prompt,
                                        max_new_tokens=self.max_new_tokens)
        except Exception as e:
            return MermaidExtractionResult(
                raw_output="", reason=f"{type(e).__name__}: {e}",
                elapsed_seconds=round(time.time() - t0, 2))

        result = MermaidExtractionResult(
            raw_output=raw, elapsed_seconds=round(time.time() - t0, 2))

        if not raw or "UNREADABLE" in raw.upper()[:200]:
            result.reason = "model returned UNREADABLE or empty"
            return result

        mermaid = _extract_mermaid_block(raw)
        if not mermaid:
            result.reason = "no ```mermaid ... ``` block found in output"
            return result

        # Validate the Mermaid syntactically.
        ok, why, nodes, edges = _validate_mermaid(mermaid)
        result.mermaid = f"```mermaid\n{mermaid}\n```"
        result.nodes = nodes
        result.edges = edges
        result.confidence = 0.7 if ok else 0.35
        result.reason = "ok" if ok else f"weak validation: {why}"
        return result


# --------------------------------------------------------------------
# Mermaid parser / validator (no `mmdc` dependency)
# --------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```\s*mermaid\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_NODE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\[([^\]]+)\]")
_EDGE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*-->\s*"
    r"(?:\|([^|]*)\|)?\s*([A-Za-z][A-Za-z0-9_]*)")
_GRAPH_HEADER_RE = re.compile(
    r"^\s*(graph|flowchart)\s+(TD|TB|BT|LR|RL)\s*$",
    re.IGNORECASE | re.MULTILINE)


def _extract_mermaid_block(text: str) -> Optional[str]:
    """Find the first ```mermaid ... ``` block; return its inner text."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Sometimes the model omits the fence and starts directly with
    # `flowchart LR ...`; accept that too.
    if re.search(r"^\s*(flowchart|graph)\s+(TD|TB|BT|LR|RL)\b",
                  text, re.IGNORECASE | re.MULTILINE):
        return text.strip()
    return None


def _validate_mermaid(src: str):
    """
    Light syntactic check: must have a graph/flowchart header, must
    have ≥ 1 node-bracket and ≥ 1 arrow. Returns (ok, reason, nodes, edges).
    """
    if not _GRAPH_HEADER_RE.search(src):
        return False, "no `graph TD/LR` or `flowchart TD/LR` header", [], []
    nodes = list(dict.fromkeys(  # de-dupe but keep order
        f"{m.group(1)}[{m.group(2).strip()}]"
        for m in _NODE_RE.finditer(src)))
    edges_raw = [(m.group(1), m.group(2), m.group(3))
                  for m in _EDGE_RE.finditer(src)]
    edges = [
        (f"{a}-->|{lbl.strip()}|{b}" if lbl else f"{a}-->{b}")
        for (a, lbl, b) in edges_raw
    ]
    if not nodes:
        return False, "no `Identifier[label]` node definitions found", nodes, edges
    if not edges:
        return False, "no `-->` arrows found", nodes, edges
    # Sanity: every edge endpoint should appear as a defined node OR be
    # implicit (Mermaid allows undefined-then-implicit). We allow both
    # but warn if more than half of edge endpoints are undefined.
    defined_ids = {n.split("[")[0] for n in nodes}
    endpoints = set()
    for a, _l, b in edges_raw:
        endpoints.add(a); endpoints.add(b)
    undef = endpoints - defined_ids
    if undef and len(undef) > len(endpoints) / 2:
        return (False,
                f"{len(undef)} of {len(endpoints)} edge endpoints are "
                f"undefined nodes: {sorted(undef)[:5]}",
                nodes, edges)
    return True, "ok", nodes, edges
