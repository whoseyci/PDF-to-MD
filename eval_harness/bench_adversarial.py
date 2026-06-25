"""Adversarial figures designed to FOOL extractors.

For each truth kind, build figures that look like a different kind:

  * A_bar_styled_as_box -- bar chart with thin horizontal cap at top
    of each bar (looks like a box median line)
  * A_scatter_with_regression -- scatter + best-fit line (looks like
    line plot)
  * A_pie_with_bordered_legend -- pie with a rectangular legend in
    corner (looks like a chart with axes)
  * A_box_with_outliers -- box plot with many outlier dots (looks
    like scatter)
  * A_horizontal_bars -- horizontal bars (often misread as box plot
    rows)
  * A_dense_legend_charts -- chart with a legend that has colored
    boxes (could be misread as stacked bars)
  * A_pictograph_bar -- 'bars' made of stacked icons (not pixel bars)
  * A_3d_bar_chart -- 3D-perspective bars
  * A_grouped_bar -- multiple groups, multiple bars per group

We measure: did the smart extractor stay correct under adversarial
conditions, and which adversarial styles still trip it up.
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

FIGS_DIR = ROOT / "_adversarial_figs"
FIGS_DIR.mkdir(exist_ok=True)


def gen_bar_with_box_styling(path):
    """Bar chart but each bar has a thin horizontal "cap" near the
    top -- looks like a box plot median line."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    cats = list("ABCDE")
    vals = [12, 18, 8, 22, 15]
    bars = ax.bar(cats, vals, color="#a3c4dd", edgecolor="black")
    # Add a "median line" inside each bar
    for b, v in zip(bars, vals):
        ax.hlines(v * 0.7, b.get_x() + 0.05, b.get_x() + 0.95,
                  color="black", linewidth=2)
    ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def gen_scatter_with_regression(path):
    """Scatter + linear-fit line. Naive extractor might call it line."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    x = rng.uniform(0, 10, 80); y = 1.5 * x + rng.normal(0, 1.5, 80)
    ax.scatter(x, y, s=15, color="#3b75af")
    # Best-fit line
    m, b = np.polyfit(x, y, 1)
    ax.plot(sorted(x), m * np.array(sorted(x)) + b,
            color="#c33e3e", linewidth=1.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def gen_pie_with_legend_box(path):
    """Pie chart with a bordered legend in the corner -- adds a
    rectangle that might look like an axis."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sizes = [40, 30, 20, 10]
    wedges, _ = ax.pie(sizes,
                        colors=["#3b75af", "#ef8636",
                                 "#519e3e", "#c33e3e"])
    ax.set_aspect("equal")
    ax.legend(["A", "B", "C", "D"], loc="upper right",
              frameon=True, edgecolor="black")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def gen_box_with_outliers(path):
    """Box plot with many outlier dots (could look like scatter)."""
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    data = []
    for i in range(4):
        # Mix of normal + heavy outliers
        d = np.concatenate([rng.normal(i * 2 + 3, 1, 30),
                             rng.uniform(0, 20, 12)])  # outliers
        data.append(d)
    ax.boxplot(data, tick_labels=list("ABCD"))
    ax.set_ylabel("Value")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def gen_horizontal_bars(path):
    """Horizontal bar chart -- often misread as box plot rows."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    cats = ["First Q", "Second Q", "Third Q", "Fourth Q"]
    vals = [25, 18, 12, 8]
    ax.barh(cats, vals, color="#3b75af")
    ax.set_xlabel("Count")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def gen_chart_with_colorful_legend(path):
    """Chart with a colorful legend (could be misread as stacked)."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    x = np.linspace(0, 10, 30)
    ax.plot(x, np.sin(x), label="alpha", color="#3b75af", linewidth=2)
    ax.plot(x, np.cos(x), label="beta", color="#ef8636", linewidth=2)
    ax.plot(x, np.sin(x + 1), label="gamma", color="#519e3e", linewidth=2)
    ax.plot(x, np.cos(x + 1), label="delta", color="#c33e3e", linewidth=2)
    ax.legend()
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def gen_grouped_bars(path):
    """Multi-group bars -- could be misread as scatter or box."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    x = np.arange(4); width = 0.35
    ax.bar(x - width/2, [3, 5, 4, 7], width,
            label="2022", color="#3b75af")
    ax.bar(x + width/2, [4, 6, 5, 8], width,
            label="2023", color="#ef8636")
    ax.set_xticks(x); ax.set_xticklabels(list("ABCD"))
    ax.legend()
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def gen_negative_value_bars(path):
    """Bars with negative values -- some extending below baseline."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    cats = list("ABCDE")
    vals = [5, -3, 8, -7, 2]
    ax.bar(cats, vals, color="#3b75af")
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def gen_log_scale_axis(path):
    """Bar chart with log-scale y axis -- tick labels are non-linear."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.bar(list("ABCD"), [1, 10, 100, 1000], color="#3b75af")
    ax.set_yscale("log")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


SPEC = [
    ("bar_chart", "bar_styled_as_box", gen_bar_with_box_styling,
     "bar with internal horizontal cap (looks like box median line)"),
    ("scatter_plot", "scatter_with_regression", gen_scatter_with_regression,
     "scatter + best-fit line (looks like line plot)"),
    ("pie_chart", "pie_with_legend_box", gen_pie_with_legend_box,
     "pie + bordered legend (extra rect could look like axis)"),
    ("box_plot", "box_with_outliers", gen_box_with_outliers,
     "box plot with many outlier dots (looks like scatter)"),
    ("bar_chart", "horizontal_bars", gen_horizontal_bars,
     "horizontal bar chart"),
    ("line_plot", "chart_with_colorful_legend", gen_chart_with_colorful_legend,
     "line chart with multi-color legend (could look stacked)"),
    ("bar_chart", "grouped_bars", gen_grouped_bars,
     "multi-group bars (could look like scatter)"),
    ("bar_chart", "negative_values", gen_negative_value_bars,
     "bars with negative values"),
    ("bar_chart", "log_scale", gen_log_scale_axis,
     "bar chart with log-scale y axis"),
]

# Captions are intentionally vague (matches real adversarial conditions
# where caption might not name the figure type)
CAPTIONS = ["Figure 1. Results of the experiment.", ""]


def main():
    if not HAS_MPL:
        print("matplotlib missing"); return
    from pipeline_v2.vision.chart_extract.parallel_extractor import (
        run_smart_extraction)

    # Generate
    cases = []
    for truth, vname, gen, desc in SPEC:
        img = FIGS_DIR / f"{truth}__{vname}.png"
        if not img.exists():
            try: gen(img)
            except Exception as e:
                print(f"FAIL gen {vname}: {e}"); continue
        cases.append((truth, vname, img, desc))
    print(f"Generated {len(cases)} adversarial cases")

    rows = []
    for truth, vname, img, desc in cases:
        for cap in CAPTIONS:
            t0 = time.time()
            tr = run_smart_extraction(image_path=img, caption=cap)
            elapsed = round(time.time() - t0, 2)
            winner_kind = tr.winner_kind or ""
            winner_status = tr.winner.status.value if tr.winner else "none"
            # "Correct" = picked truth_kind OR a closely-related kind
            # (stacked_bar accepted for bar, schematic for flow_diagram).
            accepted = {truth}
            if truth == "bar_chart":
                accepted.add("stacked_bar_chart")
            if truth == "flow_diagram":
                accepted.add("schematic")
            correct = (winner_kind in accepted
                        and winner_status in ("ok", "partial"))
            rows.append({
                "truth": truth, "variant": vname, "desc": desc,
                "cap": "rich" if cap else "empty",
                "winner_kind": winner_kind,
                "winner_status": winner_status,
                "elapsed_s": elapsed,
                "correct": correct,
                "arb": tr.arbitration_reason[:60],
            })

    # Aggregate
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    by_variant: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "rows": []})
    for r in rows:
        v = r["variant"]
        by_variant[v]["n"] += 1
        by_variant[v]["rows"].append(r)
        if r["correct"]: by_variant[v]["correct"] += 1

    md = ["# Adversarial bench", "",
            f"Cases: **{n}** = {len(cases)} adversarial figures × "
            f"{len(CAPTIONS)} caption conditions",
            "",
            f"## Overall: **{n_correct}/{n} = "
            f"{round(100*n_correct/n,1)}% correct**",
            "",
            "## By variant",
            "",
            "| variant | description | correct/n |",
            "|---|---|---|"]
    for v, s in by_variant.items():
        rate = round(100 * s['correct'] / s['n'], 1)
        desc = s['rows'][0]['desc']
        md.append(f"| {v} | {desc} | {s['correct']}/{s['n']} ({rate}%) |")
    md.append("")
    md.append("## Failures")
    md.append("")
    md.append("| truth | variant | cap | picked | status | arb |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        if not r["correct"]:
            md.append(f"| {r['truth']} | {r['variant']} | {r['cap']} | "
                       f"{r['winner_kind']} | {r['winner_status']} | "
                       f"{r['arb']} |")

    out_md = ROOT / "ADVERSARIAL_REPORT.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json = ROOT / "ADVERSARIAL_REPORT.json"
    out_json.write_text(json.dumps({
        "n": n, "n_correct": n_correct,
        "by_variant": {k: dict(v) for k, v in by_variant.items()},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n{n_correct}/{n} = {round(100*n_correct/n,1)}% correct")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
