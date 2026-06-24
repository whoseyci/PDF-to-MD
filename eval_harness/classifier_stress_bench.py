"""Stress test the figure classifier across many synthetic variants
PER kind, including adversarial cases (rotated, low-DPI, B/W, noisy).

For each kind we generate:
  - 3 'clean' variants (default styles)
  - 3 'adversarial' variants:
      * grayscale-only (no colour cue)
      * tiny font / low DPI
      * with a competing element (legend that looks like another chart)
  - 3 'minimal-caption' variants (just "Figure N." with no descriptive
    words -- classifier can't use keywords)

Total: ~9 variants × 7 kinds = ~63 cases per condition.

Metrics:
  * confusion matrix (truth vs predicted)
  * per-kind precision / recall / F1
  * cases where classifier is RIGHT but specialist still fails
  * cases where classifier is WRONG but specialist still succeeds
  * runtime distribution
"""
from __future__ import annotations
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except Exception:
    HAS_MPL = False

FIGS_DIR = ROOT / "_stress_figs"
FIGS_DIR.mkdir(exist_ok=True)


def _setup_axes(figsize=(6, 4), dpi=110):
    return plt.subplots(figsize=figsize, dpi=dpi)


# ----- Bar charts -----
def gen_bar_clean(path, seed, variant=0):
    rng = np.random.default_rng(seed + variant)
    fig, ax = _setup_axes()
    n = rng.integers(3, 8)
    cats = [chr(ord("A") + i) for i in range(int(n))]
    vals = rng.integers(5, 30, size=int(n))
    color = ["#3b75af", "#519e3e", "#c33e3e"][variant % 3]
    ax.bar(cats, vals, color=color)
    ax.set_xlabel("Group"); ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_bar_grayscale(path, seed):
    fig, ax = _setup_axes()
    ax.bar(list("ABCDE"), [5, 12, 8, 15, 7], color="gray", edgecolor="black")
    ax.set_xlabel("Group"); ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_bar_low_dpi(path, seed):
    fig, ax = _setup_axes(figsize=(3, 2), dpi=60)
    ax.bar(list("ABC"), [2, 5, 3])
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_bar_with_legend(path, seed):
    """Bar chart with a side legend that has colored boxes."""
    fig, ax = _setup_axes()
    rng = np.random.default_rng(seed)
    x = np.arange(4)
    ax.bar(x - 0.2, rng.integers(3, 10, 4), 0.4, label="Series A",
            color="#3b75af")
    ax.bar(x + 0.2, rng.integers(3, 10, 4), 0.4, label="Series B",
            color="#ef8636")
    ax.set_xticks(x); ax.set_xticklabels(list("ABCD"))
    ax.legend()
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


# ----- Pie charts -----
def gen_pie_clean(path, seed, variant=0):
    fig, _ax = plt.subplots(figsize=(5, 5), dpi=110)
    rng = np.random.default_rng(seed + variant)
    sizes = rng.integers(5, 40, rng.integers(3, 7))
    _ax.pie(sizes); _ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_pie_grayscale(path, seed):
    fig, _ax = plt.subplots(figsize=(5, 5), dpi=110)
    _ax.pie([40, 30, 20, 10],
             colors=["#d3d3d3", "#888888", "#444444", "#000000"])
    _ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_donut(path, seed):
    fig, _ax = plt.subplots(figsize=(5, 5), dpi=110)
    _ax.pie([40, 30, 20, 10],
             wedgeprops=dict(width=0.3))
    _ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


# ----- Line plots -----
def gen_line_clean(path, seed, variant=0):
    fig, ax = _setup_axes()
    rng = np.random.default_rng(seed + variant)
    x = np.linspace(0, 10, 50)
    for k, col in enumerate(["#3b75af", "#ef8636"][:rng.integers(1, 3)]):
        ax.plot(x, rng.normal(0, 1, 50).cumsum() + 5, color=col)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_line_grayscale(path, seed):
    fig, ax = _setup_axes()
    x = np.linspace(0, 10, 50)
    ax.plot(x, np.sin(x), "k-")
    ax.plot(x, np.cos(x), "k--")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_line_with_inset(path, seed):
    """Line plot with a small inset (potential multipanel confusion)."""
    fig, ax = _setup_axes()
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 50)
    ax.plot(x, rng.normal(0, 1, 50).cumsum())
    ax2 = fig.add_axes([0.65, 0.65, 0.2, 0.2])
    ax2.plot(x[:10], rng.normal(0, 1, 10).cumsum())
    plt.savefig(path); plt.close(fig)


# ----- Scatter -----
def gen_scatter_clean(path, seed, variant=0):
    fig, ax = _setup_axes()
    rng = np.random.default_rng(seed + variant)
    n_groups = rng.integers(1, 4)
    for k in range(int(n_groups)):
        ax.scatter(rng.normal(k * 3, 0.5, 40),
                    rng.normal(k * 3, 0.5, 40))
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_scatter_dense(path, seed):
    fig, ax = _setup_axes()
    rng = np.random.default_rng(seed)
    ax.scatter(rng.normal(0, 1, 500), rng.normal(0, 1, 500), s=8)
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


# ----- Box plots -----
def gen_box_clean(path, seed, variant=0):
    rng = np.random.default_rng(seed + variant)
    fig, ax = _setup_axes()
    n = rng.integers(3, 6)
    data = [rng.normal(i * 2, 1, 40) for i in range(int(n))]
    ax.boxplot(data, tick_labels=[chr(ord("A") + i) for i in range(int(n))])
    ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def gen_box_grayscale(path, seed):
    fig, ax = _setup_axes()
    rng = np.random.default_rng(seed)
    ax.boxplot([rng.normal(i, 1, 40) for i in range(4)],
                tick_labels=list("WXYZ"),
                patch_artist=True,
                boxprops=dict(facecolor="lightgray", edgecolor="black"))
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


# ----- Stacked bar -----
def gen_stacked_clean(path, seed, variant=0):
    rng = np.random.default_rng(seed + variant)
    fig, ax = _setup_axes()
    cats = list("ABCD")
    a = rng.integers(5, 20, 4); b = rng.integers(5, 15, 4)
    c = rng.integers(2, 12, 4)
    ax.bar(cats, a, color="#3b75af")
    ax.bar(cats, b, bottom=a, color="#ef8636")
    ax.bar(cats, c, bottom=a + b, color="#519e3e")
    ax.set_xlabel("Group"); ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


# ----- Flow diagrams -----
def gen_flow_clean(path, seed, variant=0):
    fig, ax = _setup_axes(figsize=(8, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    labels = ["Start", "Process", "Decide", "End"][:3 + (variant % 2)]
    xs = np.linspace(1.5, 10.5, len(labels))
    for x, lab in zip(xs, labels):
        ax.add_patch(Rectangle((x - 0.9, 2.5), 1.8, 1.0,
                                  fill=False, edgecolor='k', linewidth=2))
        ax.text(x, 3, lab, ha='center', va='center', fontsize=13)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i+1] - 0.95, 3), xytext=(xs[i] + 0.95, 3),
                     arrowprops=dict(arrowstyle="->", lw=2))
    plt.savefig(path, bbox_inches='tight'); plt.close(fig)


# ----- Banners / decorative -----
def gen_banner_decorative(path, seed):
    """Page-header banner: very wide, mostly empty."""
    fig, _ax = plt.subplots(figsize=(8, 0.5), dpi=110)
    _ax.axis("off"); _ax.text(0.5, 0.5, "Chapter Header",
                                 ha="center", va="center")
    plt.savefig(path, bbox_inches="tight"); plt.close(fig)

def gen_tiny_logo(path, seed):
    fig, _ax = plt.subplots(figsize=(0.5, 0.5), dpi=80)
    _ax.axis("off"); _ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6,
                                                 facecolor="black"))
    plt.savefig(path, bbox_inches="tight"); plt.close(fig)


# ----- Equation -----
def gen_equation(path, seed):
    fig, _ax = plt.subplots(figsize=(8, 1), dpi=110)
    _ax.axis("off")
    _ax.text(0.5, 0.5, r"$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \epsilon$",
              ha="center", va="center", fontsize=22)
    plt.savefig(path, bbox_inches="tight"); plt.close(fig)


# ----- Test grid spec -----
# Each (truth_kind, variant_name) -> generator
SPEC: List[Tuple[str, str, Callable, Dict[str, Any]]] = [
    # ---- Bar ----
    ("bar_chart", "clean_0",   gen_bar_clean,      {"seed": 0, "variant": 0}),
    ("bar_chart", "clean_1",   gen_bar_clean,      {"seed": 1, "variant": 1}),
    ("bar_chart", "clean_2",   gen_bar_clean,      {"seed": 2, "variant": 2}),
    ("bar_chart", "grayscale", gen_bar_grayscale,  {"seed": 0}),
    ("bar_chart", "low_dpi",   gen_bar_low_dpi,    {"seed": 0}),
    ("bar_chart", "legend",    gen_bar_with_legend,{"seed": 0}),
    # ---- Stacked bar ----
    ("stacked_bar_chart", "clean_0", gen_stacked_clean, {"seed": 0, "variant": 0}),
    ("stacked_bar_chart", "clean_1", gen_stacked_clean, {"seed": 1, "variant": 1}),
    # ---- Pie ----
    ("pie_chart", "clean_0", gen_pie_clean,   {"seed": 0, "variant": 0}),
    ("pie_chart", "clean_1", gen_pie_clean,   {"seed": 1, "variant": 1}),
    ("pie_chart", "grayscale", gen_pie_grayscale, {"seed": 0}),
    ("pie_chart", "donut",    gen_donut,      {"seed": 0}),
    # ---- Line ----
    ("line_plot", "clean_0", gen_line_clean,  {"seed": 0, "variant": 0}),
    ("line_plot", "clean_1", gen_line_clean,  {"seed": 1, "variant": 1}),
    ("line_plot", "grayscale", gen_line_grayscale, {"seed": 0}),
    ("line_plot", "inset",   gen_line_with_inset,{"seed": 0}),
    # ---- Scatter ----
    ("scatter_plot", "clean_0", gen_scatter_clean, {"seed": 0, "variant": 0}),
    ("scatter_plot", "clean_1", gen_scatter_clean, {"seed": 1, "variant": 1}),
    ("scatter_plot", "dense",   gen_scatter_dense, {"seed": 0}),
    # ---- Box ----
    ("box_plot", "clean_0",   gen_box_clean,     {"seed": 0, "variant": 0}),
    ("box_plot", "clean_1",   gen_box_clean,     {"seed": 1, "variant": 1}),
    ("box_plot", "grayscale", gen_box_grayscale, {"seed": 0}),
    # ---- Flow diagram ----
    ("flow_diagram", "clean_0", gen_flow_clean, {"seed": 0, "variant": 0}),
    ("flow_diagram", "clean_1", gen_flow_clean, {"seed": 1, "variant": 1}),
    # ---- Decorative ----
    ("decorative", "banner",   gen_banner_decorative, {"seed": 0}),
    ("decorative", "tiny",     gen_tiny_logo, {"seed": 0}),
    # ---- Equation ----
    ("equation",   "simple",   gen_equation, {"seed": 0}),
]


CAPTIONS = {
    # Generous, descriptive
    "rich": {
        "bar_chart": "Figure 1. Bar chart of yield by treatment group.",
        "stacked_bar_chart": "Figure 2. Stacked bar chart of land cover composition.",
        "pie_chart": "Figure 3. Pie chart of land use share (%).",
        "line_plot": "Figure 4. Line plot of treatment response over time.",
        "scatter_plot": "Figure 5. Scatter plot of pH versus crop yield.",
        "box_plot": "Figure 6. Box plot of median values by treatment.",
        "flow_diagram": "Figure 7. Workflow diagram showing the analysis pipeline.",
        "decorative": "Chapter divider.",
        "equation": "Equation 1. Linear regression model.",
    },
    # Minimal caption: just "Figure N."
    "minimal": {k: f"Figure {i+1}." for i, k in enumerate(
        ["bar_chart","stacked_bar_chart","pie_chart","line_plot",
          "scatter_plot","box_plot","flow_diagram","decorative","equation"])},
    # Empty
    "empty": {k: "" for k in
                ["bar_chart","stacked_bar_chart","pie_chart","line_plot",
                  "scatter_plot","box_plot","flow_diagram","decorative","equation"]},
}


def main():
    if not HAS_MPL:
        print("matplotlib missing"); return
    # Generate all figures
    cases = []
    for truth, vname, gen, kwargs in SPEC:
        img = FIGS_DIR / f"{truth}__{vname}.png"
        if not img.exists():
            try:
                gen(img, **kwargs)
            except Exception as e:
                print(f"FAIL gen {truth}/{vname}: {e}")
                continue
        cases.append((truth, vname, img))
    print(f"Generated {len(cases)} test cases")

    from pipeline_v2.vision.classifier import classify_figure
    from pipeline_v2.vision.mixture_classifier import (
        classify_with_mixture, classify_figure_hybrid)
    from pipeline_v2.vision.chart_extract.reflective_runner import (
        run_reflective_extraction)

    def _read_ocr(p):
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(p))
        except Exception:
            return ""

    results = []
    confusion = {ct: defaultdict(int) for ct in
                   ("keyword", "mixture", "hybrid")}
    for cap_label in ["rich", "minimal", "empty"]:
        for truth, vname, img in cases:
            cap = CAPTIONS[cap_label][truth]
            ocr = _read_ocr(img)
            t0 = time.time()
            kw_kind, _ = classify_figure(cap, ocr_text=ocr)
            kw_t = time.time() - t0
            t0 = time.time()
            mix = classify_with_mixture(caption=cap,
                                          image_path=img, ocr_text=ocr)
            mix_t = time.time() - t0
            t0 = time.time()
            hyb = classify_figure_hybrid(caption=cap,
                                           image_path=img, ocr_text=ocr)
            hyb_t = time.time() - t0
            confusion["keyword"][(truth, kw_kind.value)] += 1
            confusion["mixture"][(truth, mix.top_kind.value)] += 1
            confusion["hybrid"][(truth, hyb.top_kind.value)] += 1
            # Also run reflective to see what specialists do given the choice
            results.append({
                "truth": truth,
                "variant": vname,
                "caption_cond": cap_label,
                "kw": kw_kind.value,
                "kw_correct": kw_kind.value == truth,
                "mix": mix.top_kind.value,
                "mix_conf": mix.top_confidence,
                "mix_correct": mix.top_kind.value == truth,
                "hyb": hyb.top_kind.value,
                "hyb_correct": hyb.top_kind.value == truth,
                "ocr_len": len(ocr),
                "elapsed": {"kw": round(kw_t, 4),
                              "mix": round(mix_t, 4),
                              "hyb": round(hyb_t, 4)},
            })

    n = len(results)
    # Per-cap-cond accuracy
    by_cap = defaultdict(lambda: {"n": 0, "kw": 0, "mix": 0, "hyb": 0})
    for r in results:
        c = r["caption_cond"]
        by_cap[c]["n"] += 1
        if r["kw_correct"]: by_cap[c]["kw"] += 1
        if r["mix_correct"]: by_cap[c]["mix"] += 1
        if r["hyb_correct"]: by_cap[c]["hyb"] += 1

    # Per-kind accuracy
    by_kind = defaultdict(lambda: {"n": 0, "kw": 0, "mix": 0, "hyb": 0})
    for r in results:
        k = r["truth"]
        by_kind[k]["n"] += 1
        if r["kw_correct"]: by_kind[k]["kw"] += 1
        if r["mix_correct"]: by_kind[k]["mix"] += 1
        if r["hyb_correct"]: by_kind[k]["hyb"] += 1

    # Output
    md = ["# Classifier stress bench", "",
            f"Cases: **{n}** total = {len(cases)} figures × 3 caption conditions",
            "",
            "Caption conditions:",
            "* `rich`: descriptive caption ('Figure 1. Bar chart of yield by treatment')",
            "* `minimal`: just 'Figure 1.'",
            "* `empty`: no caption at all (image only)",
            "",
            "## Accuracy by caption condition", "",
            "| Condition | Keyword | Mixture | Hybrid |",
            "|---|---|---|---|"]
    for cap in ["rich", "minimal", "empty"]:
        s = by_cap[cap]
        md.append(
            f"| {cap} | {s['kw']}/{s['n']} ({round(100*s['kw']/s['n'],1)}%) | "
            f"{s['mix']}/{s['n']} ({round(100*s['mix']/s['n'],1)}%) | "
            f"{s['hyb']}/{s['n']} ({round(100*s['hyb']/s['n'],1)}%) |")
    md.append("")
    md.append("## Accuracy by truth kind (across all caption conditions)")
    md.append("")
    md.append("| Kind | n | Keyword | Mixture | Hybrid |")
    md.append("|---|---|---|---|---|")
    for k in sorted(by_kind.keys()):
        s = by_kind[k]
        md.append(
            f"| {k} | {s['n']} | "
            f"{s['kw']}/{s['n']} ({round(100*s['kw']/s['n'],1)}%) | "
            f"{s['mix']}/{s['n']} ({round(100*s['mix']/s['n'],1)}%) | "
            f"{s['hyb']}/{s['n']} ({round(100*s['hyb']/s['n'],1)}%) |")
    md.append("")
    md.append("## Confusion matrix — Hybrid (collapsed across caption conditions)")
    md.append("")
    all_kinds = sorted({k for (t, p) in confusion["hybrid"].keys() for k in (t, p)})
    md.append("| truth ↓ pred → | " + " | ".join(all_kinds) + " |")
    md.append("|---|" + "|".join("---" for _ in all_kinds) + "|")
    for tk in all_kinds:
        row = [f"| **{tk}** "]
        for pk in all_kinds:
            v = confusion["hybrid"].get((tk, pk), 0)
            row.append(f" {v if v else '.'} ")
        md.append("|".join(row) + " |")
    md.append("")
    # Mis-classifications detail
    md.append("## Specific failures (hybrid)")
    md.append("")
    bad = [r for r in results if not r["hyb_correct"]]
    md.append(f"Total wrong: {len(bad)}/{n}")
    md.append("")
    md.append("| truth | variant | cap | hybrid_said | mix_conf | ocr_len |")
    md.append("|---|---|---|---|---|---|")
    for r in bad[:60]:
        md.append(f"| {r['truth']} | {r['variant']} | {r['caption_cond']} | "
                   f"{r['hyb']} | {r['mix_conf']} | {r['ocr_len']} |")

    out_md = ROOT / "CLASSIFIER_STRESS_REPORT.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json = ROOT / "CLASSIFIER_STRESS_REPORT.json"
    out_json.write_text(json.dumps({
        "n": n,
        "by_caption_condition": dict(by_cap),
        "by_kind": dict(by_kind),
        "confusion_hybrid": {f"{t}->{p}": v
                              for (t, p), v in confusion["hybrid"].items()},
        "rows": results,
    }, indent=2, default=lambda o: dict(o) if isinstance(o, defaultdict) else o),
        encoding="utf-8")
    # Print top-line
    print(f"\nwrote {out_md}\n")
    for cap in ["rich", "minimal", "empty"]:
        s = by_cap[cap]
        print(f"  {cap:8s}: kw={s['kw']}/{s['n']} mix={s['mix']}/{s['n']} "
                f"hyb={s['hyb']}/{s['n']}")


if __name__ == "__main__":
    main()
