"""Wrap a single-panel extractor with multi-panel splitting + axis sharing."""
from __future__ import annotations
import tempfile, time
from pathlib import Path

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus
from .panel_split import detect_panels, crop_panel


def _retry_with_padding(inner, image_path, panel, original_image_path):
    from PIL import Image
    try:
        with Image.open(original_image_path) as im:
            full_w, full_h = im.size
            ext_x0 = 0
            ext_y0 = max(0, panel.y0 - 10)
            ext_x1 = panel.x1
            ext_y1 = full_h
            crop = im.crop((ext_x0, ext_y0, ext_x1, ext_y1))
        with tempfile.NamedTemporaryFile(prefix="ext_panel_", suffix=".png",
                                            delete=False) as tf:
            crop.save(tf.name)
            ext_path = Path(tf.name)
        sub = inner.extract(ext_path)
        try: ext_path.unlink()
        except Exception: pass
        return sub
    except Exception:
        return None


class MultiPanelExtractor(ChartExtractor):
    def __init__(self, inner):
        self.inner = inner
        self.name = f"multipanel({inner.name})"

    def extract(self, image_path, *, caption=None, ocr_text=None):
        t0 = time.time()
        panels = detect_panels(image_path)
        if len(panels) <= 1:
            r = self.inner.extract(image_path, caption=caption, ocr_text=ocr_text)
            r.extractor = f"{self.name}#1panel"
            r.elapsed_seconds = round(time.time() - t0, 3)
            return r

        panel_results = []
        per_panel_tables = []
        all_warnings = []
        with tempfile.TemporaryDirectory(prefix="panel_") as tdir:
            tdir = Path(tdir)
            for i, p in enumerate(panels):
                crop_path = tdir / f"panel_{i:02d}.png"
                ok = crop_panel(image_path, p, crop_path)
                if not ok:
                    panel_results.append({
                        "label": p.label or f"panel-{i+1}",
                        "row": p.row, "col": p.col,
                        "bbox": [p.x0, p.y0, p.x1, p.y1],
                        "result": {"status": "error", "reason": "crop failed"},
                    })
                    continue
                sub = self.inner.extract(crop_path, caption=caption,
                                          ocr_text=ocr_text)
                if sub.status == ExtractionStatus.NO_AXIS:
                    retry = _retry_with_padding(self.inner, crop_path, p,
                                                  image_path)
                    if retry is not None and retry.status != ExtractionStatus.NO_AXIS:
                        sub = retry
                        sub.warnings.append("axis labels inherited from sibling panel")
                panel_results.append({
                    "label": p.label or f"panel-{i+1}",
                    "row": p.row, "col": p.col,
                    "bbox": [p.x0, p.y0, p.x1, p.y1],
                    "result": sub.to_dict(),
                })
                tbl = sub.to_markdown_table()
                if tbl:
                    label = p.label or f"Panel {i + 1}"
                    per_panel_tables.append(f"**Panel {label}**\n\n{tbl}")
                if sub.warnings:
                    all_warnings.extend(sub.warnings)
        n_ok = sum(1 for pr in panel_results
                    if pr["result"].get("status") == "ok")
        n_partial = sum(1 for pr in panel_results
                         if pr["result"].get("status") == "partial")
        n_total = len(panel_results)
        out = ChartExtractionResult(
            extractor=self.name,
            status=(ExtractionStatus.OK if n_ok == n_total
                    else (ExtractionStatus.PARTIAL if n_ok + n_partial > 0
                          else ExtractionStatus.NO_BARS)),
            reason=(f"{n_ok}/{n_total} panels ok, {n_partial} partial"),
            confidence=(0.95 * n_ok + 0.5 * n_partial) / max(1, n_total),
            warnings=all_warnings,
            elapsed_seconds=round(time.time() - t0, 3),
        )
        out.extracted_data = {
            "panels": panel_results,
            "panel_count": n_total,
            "panel_grid": _grid_shape(panel_results),
            "per_panel_markdown": ("\n\n".join(per_panel_tables)
                                     if per_panel_tables else None),
        }
        return out


def _grid_shape(panel_results):
    rows = {pr.get("row", 0) for pr in panel_results}
    cols = {pr.get("col", 0) for pr in panel_results}
    return f"{len(rows)}x{len(cols)}"
