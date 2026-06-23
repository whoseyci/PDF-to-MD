"""Multi-extractor voting / cascade for chart-data extraction.

Runs multiple chart extractors against the same image and picks the
most trustworthy result by their reported confidence and the amount
of overlap between them.

The cascade is short-circuited at the top so a fast-path figure
costs no more than the first extractor alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus


class CascadingExtractor(ChartExtractor):
    """Cascade of `ChartExtractor`s, short-circuited on a confident hit."""

    name = "cascade"

    def __init__(self,
                 extractors: List[ChartExtractor],
                 short_circuit_confidence: float = 0.8,
                 **_):
        if not extractors:
            raise ValueError("cascade needs at least one extractor")
        self.extractors = extractors
        self.short_circuit_confidence = float(short_circuit_confidence)
        self.name = "cascade(" + "+".join(e.name for e in extractors) + ")"

    def extract(self, image_path: Path, *,
                caption: Optional[str] = None,
                ocr_text: Optional[str] = None) -> ChartExtractionResult:
        results: List[ChartExtractionResult] = []
        for ext in self.extractors:
            try:
                r = ext.extract(image_path, caption=caption,
                                  ocr_text=ocr_text)
            except Exception as e:
                r = ChartExtractionResult(
                    extractor=ext.name, status=ExtractionStatus.ERROR,
                    reason=f"{type(e).__name__}: {e}")
            results.append(r)
            if (r.status == ExtractionStatus.OK and
                r.confidence >= self.short_circuit_confidence):
                if len(results) > 1:
                    r.warnings.append(
                        f"cascade short-circuited after {ext.name}")
                r.extracted_data = {**(r.extracted_data or {}),
                                      "cascade_results": [
                                          {"extractor": x.extractor,
                                            "status": x.status.value,
                                            "confidence": x.confidence,
                                            "reason": x.reason}
                                          for x in results]}
                r.extractor = self.name
                return r
        priority = {ExtractionStatus.OK: 3, ExtractionStatus.PARTIAL: 2}
        results.sort(
            key=lambda r: (priority.get(r.status, 0), r.confidence),
            reverse=True)
        best = results[0]
        best.extractor = self.name
        best.extracted_data = {**(best.extracted_data or {}),
                                 "cascade_results": [
                                     {"extractor": x.extractor,
                                       "status": x.status.value,
                                       "confidence": x.confidence,
                                       "reason": x.reason}
                                     for x in results]}
        return best
