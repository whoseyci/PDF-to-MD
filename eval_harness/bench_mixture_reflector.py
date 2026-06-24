"""Bench the new Mixture classifier (E15) + Reflective runner (E17).

Generates synthetic figures of each kind, then measures:

  * Classifier hit-rate: old keyword classifier vs Mixture classifier
  * Extraction success: vanilla cascade vs reflective runner

Writes eval_harness/MIXTURE_REPORT.md + .json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

FIGS_DIR = ROOT / "_mixture_figs"


# ----------------------------------------------------------------------
# Synthetic figure generators (one per FigureKind we have a specialist for)
# ----------------------------------------------------------------------

def gen_bar(path: Path, seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    cats = list("ABCDE")
    vals = rng.integers(5, 30, size=5)
    ax.bar(cats, vals, color="#3b75af")
    ax.set_xlabel("Group"); ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)
    return "Figure 1. Yield by treatment group."


def gen_stacked(path: Path, seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    cats = list("ABCD")
    a = rng.integers(5, 20, 4); b = rng.integers(5, 15, 4); c = rng.integers(2, 12, 4)
    ax.bar(cats, a, color="#3b75af")
    ax.bar(cats, b, bottom=a, color="#ef8636")
    ax.bar(cats, c, bottom=a + b, color="#519e3e")
    ax.set_xlabel("Group"); ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)
    return "Figure 2. Stacked bar chart of land cover composition."


def gen_pie(path: Path, seed=0):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    ax.pie([40, 30, 20, 10], colors=["#3b75af", "#ef8636", "#519e3e", "#c33e3e"])
    ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)
    return "Figure 3. Pie chart of land use share (%)."


def gen_line(path: Path, seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    x = np.linspace(0, 10, 50)
    ax.plot(x, np.sin(x) * 3 + 5, color="#3b75af", linewidth=2)
    ax.plot(x, np.cos(x) * 2 + 5, color="#ef8636", linewidth=2)
    ax.set_xlabel("Time"); ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)
    return "Figure 4. Time series of treatment response."


def gen_scatter(path: Path, seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    x1 = rng.normal(2, 0.5, 40); y1 = rng.normal(2, 0.5, 40)
    x2 = rng.normal(7, 0.6, 40); y2 = rng.normal(8, 0.6, 40)
    ax.scatter(x1, y1, color="#3b75af", s=25)
    ax.scatter(x2, y2, color="#ef8636", s=25)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)
    return "Figure 5. Scatter plot of measurements."


def gen_box(path: Path, seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    data = [rng.normal(5, 1, 40), rng.normal(8, 1, 40),
            rng.normal(3, 1, 40), rng.normal(10, 1, 40)]
    ax.boxplot(data, tick_labels=list("ABCD"), patch_artist=True,
                boxprops=dict(facecolor="#a3c4dd"))
    ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)
    return "Figure 6. Box plot of treatment medians."


def gen_diagram(path: Path, seed=0):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    from matplotlib.patches import Rectangle
    for x, label in zip([1.5, 5.0, 8.5], ["Start", "Process", "End"]):
        ax.add_patch(Rectangle((x - 0.8, 2.5), 1.6, 0.8, fill=False,
                                  edgecolor="black", linewidth=2))
        ax.text(x, 2.9, label, ha="center", va="center", fontsize=13)
    for x1, x2 in [(2.3, 4.2), (5.8, 7.7)]:
        ax.annotate("", xy=(x2, 2.9), xytext=(x1, 2.9),
                     arrowprops=dict(arrowstyle="->", lw=2))
    plt.savefig(path, bbox_inches="tight"); plt.close(fig)
    return "Figure 7. Workflow diagram showing the analysis pipeline."


GEN_BY_KIND = {
    "bar_chart": gen_bar,
    "stacked_bar_chart": gen_stacked,
    "pie_chart": gen_pie,
    "line_plot": gen_line,
    "scatter_plot": gen_scatter,
    "box_plot": gen_box,
    "flow_diagram": gen_diagram,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _read_ocr(image_path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception:
        return ""


# ----------------------------------------------------------------------
# Bench
# ----------------------------------------------------------------------

def run():
    if not HAS_MPL:
        print("matplotlib missing"); return
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    from pipeline_v2.vision.classifier import classify_figure
    from pipeline_v2.vision.mixture_classifier import classify_with_mixture
    from pipeline_v2.vision.chart_extract.reflective_runner import (
        run_reflective_extraction)
    from pipeline_v2.vision.chart_extract.registry import (
        build_chart_extractor)
    from pipeline_v2.vision.base import FigureKind

    rows: List[Dict[str, Any]] = []
    for kind_name, gen in GEN_BY_KIND.items():
        for seed in (0, 1):
            img = FIGS_DIR / f"{kind_name}_s{seed}.png"
            caption = gen(img, seed=seed)
            ocr = _read_ocr(img)

            # Old keyword classifier
            t0 = time.time()
            old_kind, _old_reason = classify_figure(caption, ocr_text=ocr)
            old_t = time.time() - t0
            # New mixture
            t0 = time.time()
            mix = classify_with_mixture(caption=caption,
                                          image_path=img, ocr_text=ocr)
            mix_t = time.time() - t0

            # Vanilla extraction (top kind only)
            t0 = time.time()
            kind_for_ext = FigureKind(kind_name) if kind_name in [k.value for k in FigureKind] else None
            if kind_for_ext:
                ex = build_chart_extractor(kind_for_ext)
                vanilla = ex.extract(img, caption=caption, ocr_text=ocr) if ex else None
            else:
                vanilla = None
            vanilla_t = time.time() - t0
            vanilla_status = vanilla.status.value if vanilla else "no_extractor"
            vanilla_conf = vanilla.confidence if vanilla else 0.0

            # Reflective
            t0 = time.time()
            trace = run_reflective_extraction(
                image_path=img, caption=caption, ocr_text=ocr)
            refl_t = time.time() - t0
            refl_status = trace.result.status.value if trace.result else "no_result"
            refl_conf = trace.result.confidence if trace.result else 0.0
            refl_kind = trace.final_kind or "?"
            n_steps = len(trace.steps)

            rows.append({
                "truth_kind": kind_name,
                "seed": seed,
                # classifiers
                "old_kind": old_kind.value,
                "old_correct": old_kind.value == kind_name,
                "old_s": round(old_t, 3),
                "mix_top_kind": mix.top_kind.value,
                "mix_top_conf": mix.top_confidence,
                "mix_correct": mix.top_kind.value == kind_name,
                "mix_margin": mix.margin,
                "mix_s": round(mix_t, 3),
                # extraction
                "vanilla_status": vanilla_status,
                "vanilla_conf": round(vanilla_conf, 3),
                "vanilla_s": round(vanilla_t, 3),
                "refl_status": refl_status,
                "refl_conf": round(refl_conf, 3),
                "refl_kind": refl_kind,
                "refl_steps": n_steps,
                "refl_s": round(refl_t, 3),
            })

    # --- Hard case: no caption (image-only) ---
    rows_nocap: List[Dict[str, Any]] = []
    for kind_name, gen in GEN_BY_KIND.items():
        for seed in (0, 1):
            img = FIGS_DIR / f"{kind_name}_s{seed}.png"
            ocr = _read_ocr(img)
            kind_for_ext = (FigureKind(kind_name)
                            if kind_name in [k.value for k in FigureKind]
                            else None)
            # Old keyword classifier with no caption
            old_kind, _ = classify_figure("", ocr_text=ocr)
            # New mixture with no caption
            mix = classify_with_mixture(caption="",
                                          image_path=img, ocr_text=ocr)
            # Vanilla: use truth kind so vanilla cant cheat with classifier signal
            if kind_for_ext:
                ex = build_chart_extractor(kind_for_ext)
                vanilla = (ex.extract(img, caption="", ocr_text=ocr)
                           if ex else None)
            else:
                vanilla = None
            vanilla_ok = (vanilla is not None and
                          vanilla.status.value == "ok")
            # Reflective uses Mixture choice
            trace = run_reflective_extraction(
                image_path=img, caption="", ocr_text=ocr)
            refl_ok = (trace.result is not None and
                       trace.result.status.value == "ok")
            rows_nocap.append({
                "truth_kind": kind_name,
                "old_kind": old_kind.value,
                "old_correct": old_kind.value == kind_name,
                "mix_top_kind": mix.top_kind.value,
                "mix_correct": mix.top_kind.value == kind_name,
                "mix_top_conf": mix.top_confidence,
                "vanilla_ok": vanilla_ok,
                "refl_ok": refl_ok,
                "refl_kind": trace.final_kind,
                "refl_steps": len(trace.steps),
            })

    n_nc = len(rows_nocap)
    old_hits_nc = sum(1 for r in rows_nocap if r["old_correct"])
    mix_hits_nc = sum(1 for r in rows_nocap if r["mix_correct"])
    vanilla_ok_nc = sum(1 for r in rows_nocap if r["vanilla_ok"])
    refl_ok_nc = sum(1 for r in rows_nocap if r["refl_ok"])

    # Aggregate
    n = len(rows)
    old_hits = sum(1 for r in rows if r["old_correct"])
    mix_hits = sum(1 for r in rows if r["mix_correct"])
    vanilla_ok = sum(1 for r in rows if r["vanilla_status"] == "ok")
    refl_ok = sum(1 for r in rows if r["refl_status"] == "ok")
    avg_old_s = round(sum(r["old_s"] for r in rows) / n, 3)
    avg_mix_s = round(sum(r["mix_s"] for r in rows) / n, 3)
    avg_van_s = round(sum(r["vanilla_s"] for r in rows) / n, 3)
    avg_refl_s = round(sum(r["refl_s"] for r in rows) / n, 3)

    summary = {
        "n_cases": n,
        "classifier_with_caption": {
            "old_keyword": {"hits": old_hits, "acc": round(old_hits/n, 3),
                              "avg_s": avg_old_s},
            "mixture":     {"hits": mix_hits, "acc": round(mix_hits/n, 3),
                              "avg_s": avg_mix_s},
        },
        "classifier_no_caption": {
            "old_keyword": {"hits": old_hits_nc,
                              "acc": round(old_hits_nc/n_nc, 3)},
            "mixture":     {"hits": mix_hits_nc,
                              "acc": round(mix_hits_nc/n_nc, 3)},
        },
        "extractor_with_caption": {
            "vanilla":    {"ok": vanilla_ok, "rate": round(vanilla_ok/n, 3),
                             "avg_s": avg_van_s},
            "reflective": {"ok": refl_ok, "rate": round(refl_ok/n, 3),
                             "avg_s": avg_refl_s},
        },
        "extractor_no_caption": {
            "vanilla":    {"ok": vanilla_ok_nc,
                             "rate": round(vanilla_ok_nc/n_nc, 3)},
            "reflective": {"ok": refl_ok_nc,
                             "rate": round(refl_ok_nc/n_nc, 3)},
        },
    }

    # Render Markdown
    lines = ["# Mixture (E15) + Reflective (E17) bench", "",
              f"Cases: {n} synthetic figures across "
              f"{len(GEN_BY_KIND)} kinds (×2 seeds each).",
              "",
              "## Classifier — WITH explicit captions",
              "",
              "| Classifier | hit-rate | avg s |",
              "|---|---|---|",
              f"| keyword (old) | {old_hits}/{n} = {round(old_hits/n*100,1)}% | {avg_old_s}s |",
              f"| **mixture (E15)** | **{mix_hits}/{n} = {round(mix_hits/n*100,1)}%** | {avg_mix_s}s |",
              "",
              "## Classifier — WITHOUT captions (image-only stress test)",
              "",
              "| Classifier | hit-rate |",
              "|---|---|",
              f"| keyword (old) | {old_hits_nc}/{n_nc} = {round(old_hits_nc/n_nc*100,1)}% |",
              f"| **mixture (E15)** | **{mix_hits_nc}/{n_nc} = {round(mix_hits_nc/n_nc*100,1)}%** |",
              "",
              "## Extractor success — WITH captions",
              "",
              "| Strategy | OK rate | avg s |",
              "|---|---|---|",
              f"| vanilla (truth kind) | {vanilla_ok}/{n} = {round(vanilla_ok/n*100,1)}% | {avg_van_s}s |",
              f"| **reflective (E17)** | **{refl_ok}/{n} = {round(refl_ok/n*100,1)}%** | {avg_refl_s}s |",
              "",
              "## Extractor success — WITHOUT captions",
              "",
              "| Strategy | OK rate |",
              "|---|---|",
              f"| vanilla (truth kind, cheating) | {vanilla_ok_nc}/{n_nc} = {round(vanilla_ok_nc/n_nc*100,1)}% |",
              f"| **reflective (E17)** | **{refl_ok_nc}/{n_nc} = {round(refl_ok_nc/n_nc*100,1)}%** |",
              "",
              "## Per-case detail", "",
              "| truth | old? | mix? | mix.top (conf) | vanilla | refl | refl_kind | steps |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['truth_kind']}_s{r['seed']} | "
            f"{'✅' if r['old_correct'] else '❌ '+r['old_kind']} | "
            f"{'✅' if r['mix_correct'] else '❌ '+r['mix_top_kind']} | "
            f"{r['mix_top_kind']} ({r['mix_top_conf']:.2f}) | "
            f"{r['vanilla_status']} ({r['vanilla_conf']:.2f}) | "
            f"{r['refl_status']} ({r['refl_conf']:.2f}) | "
            f"{r['refl_kind']} | {r['refl_steps']} |"
        )
    lines.append("")

    (ROOT / "MIXTURE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "MIXTURE_REPORT.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8")

    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote eval_harness/MIXTURE_REPORT.md + .json")


if __name__ == "__main__":
    run()
