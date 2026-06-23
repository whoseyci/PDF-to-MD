"""
DePlot chart-to-table extractor (Google research, 2023).

`google/deplot` is a 282M Pix2Struct model fine-tuned on the
chart-derendering task: given a chart image, emit the underlying
data table. It's small enough to run on CPU in our 2 GB sandbox
(via the subprocess wrapper) and complements our two existing
chart paths:

  1. Geometric (`SimpleBarsExtractor` etc.) -- exact pixel measurements
     but only works for chart kinds we've implemented (bar, box,
     stacked, etc.).
  2. Gemma 4 (`Gemma4SubprocessModel`) -- reads everything but is
     slow and qualitative.
  3. DePlot (this module) -- specialist VLM, fills the gap between
     1 and 2.

DePlot's output is its own custom format ("TITLE | x | y\\nLabel | 1 | 2"
joined by ` <0x0A> ` separators). We parse it back into a real
markdown table.

Memory note: DePlot is 282M params ~ 1.1 GB at fp32. On a 2 GB
sandbox you should EITHER use the subprocess-isolated variant
`DeplotSubprocessExtractor` OR resize images to <= 640px on the
long edge to keep activation memory bounded.

Cost on 2 vCPU CPU-only: ~25-65 s per figure depending on image size.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus


_DEFAULT_REPO = "google/deplot"
_DEPLOT_QUERY = "Generate underlying data table of the figure below:"


class DeplotExtractor(ChartExtractor):
    """Pix2Struct-based chart -> table extractor.

    Lazy-loads the model on first call so that importing this module
    is cheap when DePlot isn't actually used.
    """

    name = "deplot/v1"

    def __init__(self,
                 *,
                 repo_id: str = _DEFAULT_REPO,
                 max_new_tokens: int = 512,
                 dtype: str = "float32",
                 device: Optional[str] = None,
                 max_image_dim: int = 640,
                 **_):
        self.repo_id = repo_id
        self.max_new_tokens = int(max_new_tokens)
        self.dtype = dtype
        self.device = device
        self.max_image_dim = int(max_image_dim)
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import (Pix2StructProcessor,
                                    Pix2StructForConditionalGeneration)
        import torch
        self._processor = Pix2StructProcessor.from_pretrained(self.repo_id)
        torch_dtype = getattr(torch, self.dtype, torch.float32)
        self._model = Pix2StructForConditionalGeneration.from_pretrained(
            self.repo_id, torch_dtype=torch_dtype)
        if self.device:
            self._model = self._model.to(self.device)
        self._model.eval()

    def extract(self, image_path: Path, *,
                caption: Optional[str] = None,
                ocr_text: Optional[str] = None) -> ChartExtractionResult:
        t0 = time.time()
        r = ChartExtractionResult(extractor=self.name,
                                   status=ExtractionStatus.ERROR)
        try:
            from PIL import Image
            self._ensure_loaded()
            import torch
            img = Image.open(image_path).convert("RGB")
            if self.max_image_dim and max(img.size) > self.max_image_dim:
                scale = self.max_image_dim / max(img.size)
                new_size = (int(img.size[0] * scale),
                              int(img.size[1] * scale))
                img = img.resize(new_size, Image.LANCZOS)
            inputs = self._processor(images=img, text=_DEPLOT_QUERY,
                                       return_tensors="pt")
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                pred = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens)
            raw = self._processor.decode(pred[0], skip_special_tokens=True)
            parsed = _parse_deplot_output(raw)
            if parsed is None:
                r.status = ExtractionStatus.NO_BARS
                r.reason = ("deplot output did not parse as a table: "
                             + raw[:120])
                r.elapsed_seconds = round(time.time() - t0, 3)
                return r
            title, headers, rows = parsed
            r.value_label = title or None
            if headers:
                r.category_label = headers[0]
            if len(headers) == 2 and rows:
                r.categories = [row[0] for row in rows]
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row[1].replace(",", "")))
                    except (ValueError, IndexError):
                        vals.append(None)
                r.values = [v for v in vals if v is not None]
                if len(r.values) != len(rows):
                    r.warnings.append(
                        f"{len(rows) - len(r.values)} of {len(rows)} "
                        "values were not numeric")
            elif len(headers) > 2 and rows:
                r.series = headers[1:]
                r.categories = [row[0] for row in rows]
                matrix = []
                for row in rows:
                    parsed_row = []
                    for v in row[1:]:
                        try:
                            parsed_row.append(float(v.replace(",", "")))
                        except (ValueError, AttributeError):
                            parsed_row.append(0.0)
                    matrix.append(parsed_row)
                r.matrix = matrix
            r.confidence = 0.75
            r.status = ExtractionStatus.OK
            r.reason = (f"deplot parsed {len(r.categories)} rows "
                         + (f"x {len(r.series)} series" if r.series else ""))
            r.extracted_data = {"deplot_raw_output": raw,
                                  "deplot_title": title}
        except Exception as e:
            r.status = ExtractionStatus.ERROR
            r.reason = f"{type(e).__name__}: {e}"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r


_ROW_SEP_RE = re.compile(r"\s*<0x0A>\s*|\s*\n\s*")


def _parse_deplot_output(raw: str
                          ) -> Optional[Tuple[Optional[str],
                                                List[str],
                                                List[List[str]]]]:
    """Return (title, headers, rows) or None if unparseable."""
    raw = raw.strip()
    if not raw:
        return None
    rows = [r.strip() for r in _ROW_SEP_RE.split(raw) if r.strip()]
    if not rows:
        return None
    title = None
    if rows[0].upper().startswith("TITLE"):
        m = re.match(r"^TITLE\s*[|:]\s*(.+)$", rows[0], re.IGNORECASE)
        title = m.group(1).strip() if m else None
        rows = rows[1:]
    if not rows:
        return None
    headers = [c.strip() for c in rows[0].split("|")]
    data_rows = []
    for row in rows[1:]:
        cells = [c.strip() for c in row.split("|")]
        if not cells or all(not c for c in cells):
            continue
        if len(cells) < len(headers):
            cells = cells + [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        data_rows.append(cells)
    if not headers or not data_rows:
        return None
    return title, headers, data_rows
