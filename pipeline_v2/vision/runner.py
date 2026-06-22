"""
Top-level runner: classify → prompt → vision-model → validate → cache.

Designed to be CALLED FROM `convert.py` after `reorganize_and_describe_figures`
has finished. Each call processes a single figure and writes a sidecar
JSON file (`fig-NNN.vision.json`) next to the image. The sidecar is
cached: if it already exists for the same model and prompt, we skip
re-running the model.

All vision-model errors are CAUGHT and recorded in the result. A
single failing figure must NEVER abort the batch.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from .base import FigureKind, FigureVisionResult, VisionModel
from .classifier import classify_figure
from .prompts import build_prompt
from .validators import (
    validate_short_sentence,
    validate_mermaid,
    validate_markdown_table,
    validate_latex,
)
from .chart_extract import get_extractor
from .chart_extract.base import ExtractionStatus
from .chart_extract.validator import validate_with_vlm


# Token budgets per kind — Mermaid + tables need more room than alt sentences
_MAX_NEW_TOKENS = {
    FigureKind.FLOW_DIAGRAM: 400,
    FigureKind.SCHEMATIC: 400,
    FigureKind.TABLE_AS_IMAGE: 500,
    FigureKind.BAR_CHART: 300,
    FigureKind.PIE_CHART: 300,
    FigureKind.EQUATION: 150,
    # default for everything else: short alt sentence
}


def _load_ocr_text(figure_meta: Dict[str, Any], paper_dir: Path) -> Optional[str]:
    """Read the OCR sidecar created in stage 1, if any."""
    rel = figure_meta.get("ocr_text_file")
    if not rel:
        return None
    p = paper_dir / rel
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _sidecar_path(paper_dir: Path, fig_id: str) -> Path:
    return paper_dir / "figures" / f"{fig_id}.vision.json"


def process_figure(
    figure_meta: Dict[str, Any],
    paper_dir: Path,
    model: VisionModel,
    *,
    force: bool = False,
    log_prefix: str = "",
) -> FigureVisionResult:
    """
    Run the vision pipeline for one figure.

    Parameters
    ----------
    figure_meta : dict
        One entry from ``paper.json``'s ``figures`` list.
        Must contain: ``id``, ``file`` (relative to paper_dir),
        and may contain ``caption_text``, ``ocr_text_file``.
    paper_dir : Path
        Root output dir for this paper (e.g. ``output/baden-bohm-2023/``).
    model : VisionModel
        A concrete `VisionModel` instance.
    force : bool
        Re-run even if a sidecar already exists.
    log_prefix : str
        Optional prefix for stdout logs.

    Returns
    -------
    FigureVisionResult
        Always returns a result — failures go in `.error`, never raised.
    """
    fig_id = figure_meta["id"]
    img_rel = figure_meta["file"]
    img_path = paper_dir / img_rel

    sidecar = _sidecar_path(paper_dir, fig_id)

    # Cache hit?
    if sidecar.exists() and not force:
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            res = FigureVisionResult(figure_id=fig_id)
            for k, v in cached.items():
                if hasattr(res, k):
                    setattr(res, k, v)
            if isinstance(res.kind, str):
                try:
                    res.kind = FigureKind(res.kind)
                except ValueError:
                    res.kind = FigureKind.UNKNOWN
            return res
        except Exception:
            pass  # fall through and re-generate

    caption = figure_meta.get("caption_text") or None
    ocr_text = _load_ocr_text(figure_meta, paper_dir)

    # 1) Classify
    kind, reason = classify_figure(caption, ocr_text)
    result = FigureVisionResult(
        figure_id=fig_id,
        kind=kind,
        classifier_reason=reason,
        model_name=model.name,
    )

    # 1a) For chart-shaped figures, try CLASSICAL extraction first.
    # This produces a real markdown table from pixel measurements,
    # rather than asking the VLM to read numbers it cannot reliably
    # read. The result -- including the failure status if any -- is
    # always recorded in result.chart_extraction.
    chart_table: Optional[str] = None
    extractor = get_extractor(kind) if img_path.exists() else None
    if extractor is not None:
        try:
            chart_res = extractor.extract(img_path, caption=caption,
                                            ocr_text=ocr_text)
            result.chart_extraction = chart_res.to_dict()
            if chart_res.status in (ExtractionStatus.OK,
                                       ExtractionStatus.PARTIAL):
                chart_table = chart_res.to_markdown_table()
                if log_prefix:
                    print(f"{log_prefix}chart-extract {fig_id}: "
                            f"{chart_res.status.value} "
                            f"({len(chart_res.values)} bars, "
                            f"conf={chart_res.confidence:.2f})")
        except Exception as e:
            result.chart_extraction = {
                "extractor": extractor.name,
                "status": "error",
                "reason": f"{type(e).__name__}: {e}",
            }

    # 1b) For diagram-shaped figures, try MERMAID extraction via the
    # vision model. Mermaid renders natively in GitHub markdown and
    # gives the reader a clickable, queryable version of the diagram
    # in place of an opaque image.
    if kind in (FigureKind.FLOW_DIAGRAM, FigureKind.SCHEMATIC) \
            and img_path.exists():
        try:
            from .mermaid_extract import MermaidExtractor
            mext = MermaidExtractor(model)
            mres = mext.extract(img_path, caption=caption)
            if mres.mermaid:
                result.mermaid = mres.mermaid
                result.extracted_data = {
                    **(result.extracted_data or {}),
                    "mermaid_nodes": mres.nodes,
                    "mermaid_edges": mres.edges,
                    "mermaid_confidence": mres.confidence,
                    "mermaid_reason": mres.reason,
                    "mermaid_elapsed_seconds": mres.elapsed_seconds,
                }
                # Build a generic alt-text from node count
                n = len(mres.nodes)
                result.alt_text = (
                    f"Conceptual diagram with {n} labeled node"
                    f"{'s' if n != 1 else ''}, rendered below as Mermaid."
                )
                if log_prefix:
                    print(f"{log_prefix}mermaid-extract {fig_id}: "
                            f"{len(mres.nodes)} nodes, {len(mres.edges)} edges, "
                            f"conf={mres.confidence:.2f}")
                _save(sidecar, result)
                return result
            elif log_prefix:
                print(f"{log_prefix}mermaid-extract {fig_id}: "
                        f"FAIL ({mres.reason}), falling back to alt-text")
        except Exception as e:
            if log_prefix:
                print(f"{log_prefix}mermaid-extract {fig_id}: "
                        f"EXC {type(e).__name__}: {e}")

    # 2) Build prompt + 3) Run model (fully fail-safe)
    if kind == FigureKind.DECORATIVE:
        result.error = "skip-decorative"
        _save(sidecar, result)
        return result

    if not img_path.exists():
        result.error = f"image-missing:{img_rel}"
        _save(sidecar, result)
        return result

    prompt = build_prompt(kind, caption, ocr_text)
    result.prompt = prompt
    max_new_tokens = _MAX_NEW_TOKENS.get(kind, 120)

    t0 = time.time()
    try:
        raw = model.describe(img_path, prompt, max_new_tokens=max_new_tokens)
        result.raw_output = raw
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # Fail-safe: capture any error WITHOUT aborting the batch
        result.error = f"{type(e).__name__}: {e}"
        result.elapsed_seconds = round(time.time() - t0, 2)
        if log_prefix:
            print(f"{log_prefix}vision-error {fig_id}: {result.error}")
        _save(sidecar, result)
        return result
    result.elapsed_seconds = round(time.time() - t0, 2)

    # 4) Validate per-kind
    _populate_validated(result)

    # 4a) If classical extraction produced a table, prefer it over
    # whatever the VLM said for chart kinds. Then optionally ask the
    # VLM to cross-check the table; if the VLM flags it as obviously
    # wrong, demote to the VLM alt-text + record the flag.
    if chart_table is not None:
        try:
            from .chart_extract.base import ChartExtractionResult
            ce_dict = result.chart_extraction or {}
            verdict = validate_with_vlm(img_path,
                                          _reconstitute(ce_dict),
                                          model)
            result.validator = {
                "verdict": verdict.verdict,
                "note": verdict.note,
                "elapsed_seconds": verdict.elapsed_seconds,
            }
            if verdict.verdict == "flag":
                # Keep both: the table is still recorded in
                # chart_extraction, but we don't substitute it into
                # the alt-text path.
                if log_prefix:
                    print(f"{log_prefix}validator flagged {fig_id}: "
                            f"{verdict.note!r}")
            else:
                # Promote the extracted table.
                result.markdown_table = chart_table
                if not result.alt_text:
                    result.alt_text = _chart_alt_summary(ce_dict)
                # If the VLM path failed validation but classical
                # extraction succeeded, clear the error -- we have a
                # real result to show.
                if result.error == "output-failed-validation":
                    result.error = None
        except Exception as e:
            result.validator = {"verdict": "error", "note": str(e)}
            # Still keep the table even if the validator failed.
            result.markdown_table = chart_table
            if not result.alt_text:
                result.alt_text = _chart_alt_summary(result.chart_extraction or {})

    # 5) Cache
    _save(sidecar, result)
    return result


def _reconstitute(ce_dict: Dict[str, Any]):
    """Build a `ChartExtractionResult` from a serialised dict so we
    can call `to_markdown_table()` / pass it to the validator without
    keeping the live object around."""
    from .chart_extract.base import ChartExtractionResult, ExtractionStatus
    obj = ChartExtractionResult()
    for k, v in ce_dict.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    if isinstance(obj.status, str):
        try:
            obj.status = ExtractionStatus(obj.status)
        except ValueError:
            obj.status = ExtractionStatus.ERROR
    return obj


def _chart_alt_summary(ce_dict: Dict[str, Any]) -> str:
    """A safe one-sentence fallback alt-text built from the
    classical extraction (no VLM, so no hallucination). The kind of
    chart is inferred from which fields the extractor populated."""
    cats = ce_dict.get("categories") or []
    vlabel = ce_dict.get("value_label") or "values"
    clabel = ce_dict.get("category_label") or "categories"
    box_stats = ce_dict.get("box_stats")
    series = ce_dict.get("series") or []
    matrix = ce_dict.get("matrix") or []
    if box_stats and cats:
        return (f"Box plot showing the distribution of {vlabel} across "
                f"{len(cats)} {clabel}. See table for min, Q1, median, "
                f"Q3, max per group.")
    if matrix and series and cats:
        return (f"Stacked bar chart of {vlabel} across {len(cats)} "
                f"{clabel}, split into {len(series)} series. See table "
                f"for per-segment values.")
    vals = ce_dict.get("values") or []
    if cats and vals and len(cats) == len(vals):
        orient = ce_dict.get("orientation") or "vertical"
        return (f"Bar chart of {vlabel} across {len(cats)} {clabel} "
                f"({orient}). See table for the extracted values.")
    return "Chart; see extracted table for values."


def _populate_validated(result: FigureVisionResult) -> None:
    """Run the right validators for the figure kind and populate
    `alt_text`, `mermaid`, `markdown_table`, `extracted_data`."""
    kind = result.kind
    raw = result.raw_output or ""

    # Mermaid is only valid for diagram-shaped figures
    if kind in (FigureKind.FLOW_DIAGRAM, FigureKind.SCHEMATIC):
        mermaid = validate_mermaid(raw)
        if mermaid:
            result.mermaid = mermaid
            # Also derive a one-line alt
            result.alt_text = (
                result.classifier_reason
                and f"Diagram extracted via vision model ({result.model_name})."
            )
            return
        # Mermaid failed → fall back to short-sentence alt text

    # Tables are valid for table-as-image, bar charts, pie charts
    if kind in (FigureKind.TABLE_AS_IMAGE, FigureKind.BAR_CHART, FigureKind.PIE_CHART):
        tbl = validate_markdown_table(raw)
        if tbl:
            result.markdown_table = tbl
            # also try a short sentence (chart prose) before the table
            alt = validate_short_sentence(raw, max_words=40)
            if alt:
                result.alt_text = alt
            return

    if kind == FigureKind.EQUATION:
        latex = validate_latex(raw)
        if latex:
            result.extracted_data = {"latex": latex}
            result.alt_text = "Equation."
            return

    # Default path: short alt sentence
    alt = validate_short_sentence(raw, max_words=60)
    if alt:
        result.alt_text = alt
    else:
        result.error = "output-failed-validation"


def _save(sidecar: Path, result: FigureVisionResult) -> None:
    """Persist the result as JSON next to the image. Defensive: never raises."""
    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Sidecar writes never block the pipeline
        pass
