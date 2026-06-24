"""Compare parallel extractor against reflective runner across the
stress-bench cases."""
from __future__ import annotations
import json
import sys
import time
import collections
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from pipeline_v2.vision.base import FigureKind
from pipeline_v2.vision.mixture_classifier import classify_figure_hybrid
from pipeline_v2.vision.chart_extract.reflective_runner import (
    run_reflective_extraction)
from pipeline_v2.vision.chart_extract.parallel_extractor import (
    run_parallel_extraction, run_smart_extraction)

FIGS_DIR = ROOT / "_stress_figs"

EXPECTED_WINNERS = {
    "bar_chart":         {"bar_chart"},
    "stacked_bar_chart": {"stacked_bar_chart", "bar_chart"},
    "pie_chart":         {"pie_chart"},
    "line_plot":         {"line_plot"},
    "scatter_plot":      {"scatter_plot"},
    "box_plot":          {"box_plot"},
    "flow_diagram":      {"flow_diagram", "schematic"},
    "equation":          {"equation"},
    "decorative":        {"decorative"},
}


_OCR_CACHE: dict = {}

def _read_ocr(p: Path) -> str:
    key = str(p)
    if key in _OCR_CACHE:
        return _OCR_CACHE[key]
    try:
        import pytesseract
        from PIL import Image
        out = pytesseract.image_to_string(Image.open(p))
    except Exception:
        out = ""
    _OCR_CACHE[key] = out
    return out


# Collect cases from FIGS_DIR. Limit to 2 variants per truth kind to
# keep runtime under ~6 minutes.
_by_truth = collections.defaultdict(list)
for img in sorted(FIGS_DIR.glob("*.png")):
    name = img.stem
    if "__" not in name:
        continue
    truth, variant = name.split("__", 1)
    _by_truth[truth].append((truth, variant, img))
cases = []
for truth, items in _by_truth.items():
    cases.extend(items[:2])

CAPTIONS = {
    "rich": {
        "bar_chart": "Figure 1. Bar chart of yield by treatment group.",
        "stacked_bar_chart": "Figure 2. Stacked bar chart of land cover.",
        "pie_chart": "Figure 3. Pie chart of land use share (%).",
        "line_plot": "Figure 4. Line plot of treatment response over time.",
        "scatter_plot": "Figure 5. Scatter plot of pH versus crop yield.",
        "box_plot": "Figure 6. Box plot of median values by treatment.",
        "flow_diagram": "Figure 7. Workflow diagram showing the analysis pipeline.",
        "decorative": "Chapter divider.",
        "equation": "Equation 1. Linear regression model.",
    },
    "empty": {k: "" for k in [
        "bar_chart","stacked_bar_chart","pie_chart","line_plot",
        "scatter_plot","box_plot","flow_diagram","decorative","equation"]},
}


rows = []
for cap_label in ["rich", "empty"]:
    for truth, variant, img in cases:
        cap = CAPTIONS[cap_label].get(truth, "")
        ocr = _read_ocr(img)
        _ok_status = ("ok", "partial")
        # decorative SHOULD be unsupported; equation may be (no pix2tex)
        if truth in ("decorative", "equation"):
            _ok_status = ("ok", "partial", "unsupported")
        # Reflective
        t0 = time.time()
        refl = run_reflective_extraction(
            image_path=img, caption=cap, ocr_text=ocr)
        refl_t = time.time() - t0
        refl_kind = refl.final_kind or ""
        refl_status = refl.result.status.value if refl.result else "none"
        refl_correct = (refl_kind in EXPECTED_WINNERS.get(truth, set())
                         and refl_status in _ok_status)
        # Parallel (with classifier hint)
        mix = classify_figure_hybrid(caption=cap, image_path=img,
                                        ocr_text=ocr)
        t0 = time.time()
        par = run_parallel_extraction(
            image_path=img, caption=cap, ocr_text=ocr,
            classifier_hint=mix.top_kind)
        par_t = time.time() - t0
        par_kind = par.winner_kind or ""
        par_status = par.winner.status.value if par.winner else "none"
        par_correct = (par_kind in EXPECTED_WINNERS.get(truth, set())
                        and par_status in _ok_status)
        # Smart
        t0 = time.time()
        smart = run_smart_extraction(
            image_path=img, caption=cap, ocr_text=ocr)
        sm_t = time.time() - t0
        sm_kind = smart.winner_kind or ""
        sm_status = smart.winner.status.value if smart.winner else "none"
        sm_correct = (sm_kind in EXPECTED_WINNERS.get(truth, set())
                       and sm_status in _ok_status)
        rows.append({
            "truth": truth, "variant": variant, "cap": cap_label,
            "refl_kind": refl_kind, "refl_status": refl_status,
            "refl_correct": refl_correct, "refl_s": round(refl_t, 2),
            "par_kind": par_kind, "par_status": par_status,
            "par_correct": par_correct, "par_s": round(par_t, 2),
            "par_n_run": par.n_extractors_run,
            "sm_kind": sm_kind, "sm_status": sm_status,
            "sm_correct": sm_correct, "sm_s": round(sm_t, 2),
            "sm_n_run": smart.n_extractors_run,
        })


n = len(rows)
agg = {
    "n": n,
    "reflective": {
        "correct": sum(1 for r in rows if r["refl_correct"]),
        "avg_s": round(sum(r["refl_s"] for r in rows) / n, 2),
    },
    "parallel_hinted": {
        "correct": sum(1 for r in rows if r["par_correct"]),
        "avg_s": round(sum(r["par_s"] for r in rows) / n, 2),
        "avg_extractors_run": round(sum(r["par_n_run"] for r in rows) / n, 1),
    },
    "smart": {
        "correct": sum(1 for r in rows if r["sm_correct"]),
        "avg_s": round(sum(r["sm_s"] for r in rows) / n, 2),
        "avg_extractors_run": round(sum(r["sm_n_run"] for r in rows) / n, 1),
    },
}

by_cap = defaultdict(lambda: {"n": 0, "refl": 0, "par": 0, "sm": 0})
for r in rows:
    c = r["cap"]
    by_cap[c]["n"] += 1
    if r["refl_correct"]: by_cap[c]["refl"] += 1
    if r["par_correct"]: by_cap[c]["par"] += 1
    if r["sm_correct"]: by_cap[c]["sm"] += 1

by_kind = defaultdict(lambda: {"n": 0, "refl": 0, "par": 0, "sm": 0})
for r in rows:
    k = r["truth"]
    by_kind[k]["n"] += 1
    if r["refl_correct"]: by_kind[k]["refl"] += 1
    if r["par_correct"]: by_kind[k]["par"] += 1
    if r["sm_correct"]: by_kind[k]["sm"] += 1

md = ["# Parallel-extractor bench (vs reflective)", "",
        f"Cases: **{n}** = {len(cases)} figures × 2 caption conditions.",
        "Truth = expected winning extractor kind.",
        "",
        "## Aggregate", "",
        "| Strategy | correct/total | avg s | avg extractors run |",
        "|---|---|---|---|",
        f"| Reflective (kind-ladder) | "
        f"{agg['reflective']['correct']}/{n} = "
        f"{round(100*agg['reflective']['correct']/n,1)}% | "
        f"{agg['reflective']['avg_s']}s | (varies) |",
        f"| Parallel + classifier hint | "
        f"{agg['parallel_hinted']['correct']}/{n} = "
        f"{round(100*agg['parallel_hinted']['correct']/n,1)}% | "
        f"{agg['parallel_hinted']['avg_s']}s | "
        f"{agg['parallel_hinted']['avg_extractors_run']} |",
        f"| **Smart (caption-decisive→reflective, else→parallel)** | "
        f"**{agg['smart']['correct']}/{n} = "
        f"{round(100*agg['smart']['correct']/n,1)}%** | "
        f"{agg['smart']['avg_s']}s | "
        f"{agg['smart']['avg_extractors_run']} |",
        "",
        "## By caption condition", "",
        "| Condition | Reflective | Parallel+hint | Smart |",
        "|---|---|---|---|"]
for cap, s in by_cap.items():
    if not s['n']: continue
    md.append(
        f"| {cap} | {s['refl']}/{s['n']} ({round(100*s['refl']/s['n'],1)}%) | "
        f"{s['par']}/{s['n']} ({round(100*s['par']/s['n'],1)}%) | "
        f"{s['sm']}/{s['n']} ({round(100*s['sm']/s['n'],1)}%) |"
    )
md.append("")
md.append("## By truth kind")
md.append("")
md.append("| Kind | n | Reflective | Parallel+hint | Smart |")
md.append("|---|---|---|---|---|")
for k in sorted(by_kind.keys()):
    s = by_kind[k]
    md.append(
        f"| {k} | {s['n']} | "
        f"{s['refl']}/{s['n']} ({round(100*s['refl']/s['n'],1)}%) | "
        f"{s['par']}/{s['n']} ({round(100*s['par']/s['n'],1)}%) | "
        f"{s['sm']}/{s['n']} ({round(100*s['sm']/s['n'],1)}%) |"
    )
md.append("")
md.append("## Smart failures")
md.append("")
md.append("| truth | variant | cap | smart picked | status |")
md.append("|---|---|---|---|---|")
for r in rows:
    if not r["sm_correct"]:
        md.append(f"| {r['truth']} | {r['variant']} | {r['cap']} | "
                   f"{r['sm_kind']} | {r['sm_status']} |")

out_md = ROOT / "PARALLEL_REPORT.md"
out_md.write_text("\n".join(md), encoding="utf-8")
out_json = ROOT / "PARALLEL_REPORT.json"
out_json.write_text(json.dumps({
    "aggregate": agg,
    "by_caption_condition": {k: dict(v) for k, v in by_cap.items()},
    "by_kind": {k: dict(v) for k, v in by_kind.items()},
    "rows": rows,
}, indent=2), encoding="utf-8")
print(json.dumps(agg, indent=2))
print(f"\nwrote {out_md}")
