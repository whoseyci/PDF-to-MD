"""VLM-as-validator: asks the VLM whether an extracted table matches the image."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..base import VisionModel
from .base import ChartExtractionResult, ExtractionStatus


@dataclass
class ValidatorVerdict:
    verdict: str
    note: str = ""
    raw: str = ""
    elapsed_seconds: float = 0.0


_VALIDATOR_PROMPT_TEMPLATE = """\
You are checking whether a transcribed data table matches a chart image.

The chart was automatically read by a non-AI tool, which produced this
table:

{table}

Look at the image. Answer with EXACTLY ONE LINE:

  OK         -- if the table is roughly consistent with what the chart shows
  FLAG <why> -- if the table is obviously wrong (wrong number of bars,
                wildly wrong values, wrong categories, etc.)

Do not repeat the table. Do not add commentary. One line only.
"""


def validate_with_vlm(image_path, extraction, model):
    if model is None:
        return ValidatorVerdict(verdict="skipped", note="no model")
    if extraction.status not in (ExtractionStatus.OK, ExtractionStatus.PARTIAL):
        return ValidatorVerdict(verdict="skipped",
                                 note=f"status={extraction.status.value}")
    table = extraction.to_markdown_table()
    if not table:
        return ValidatorVerdict(verdict="skipped", note="no table")
    prompt = _VALIDATOR_PROMPT_TEMPLATE.format(table=table)
    import time
    t0 = time.time()
    try:
        out = model.describe(image_path, prompt=prompt, max_new_tokens=40)
        raw = (out or "").strip()
        first_line = raw.splitlines()[0].strip() if raw else ""
        fl = first_line.lower()
        if fl.startswith("ok"):
            return ValidatorVerdict(verdict="ok", raw=raw,
                                     elapsed_seconds=round(time.time() - t0, 3))
        if fl.startswith("flag"):
            note = first_line[4:].lstrip(":-– ").strip()
            return ValidatorVerdict(verdict="flag", note=note, raw=raw,
                                     elapsed_seconds=round(time.time() - t0, 3))
        return ValidatorVerdict(verdict="skipped",
                                 note="non-conforming output", raw=raw,
                                 elapsed_seconds=round(time.time() - t0, 3))
    except Exception as e:
        return ValidatorVerdict(verdict="error",
                                 note=f"{type(e).__name__}: {e}",
                                 elapsed_seconds=round(time.time() - t0, 3))
