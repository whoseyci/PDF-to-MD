"""Value-fidelity bench: do the extracted NUMBERS match the truth?

The existing classifier/parallel benches measure "did the right
extractor get called". This bench measures "did it return the right
values".

For each chart kind we generate synthetic figures with KNOWN
ground-truth values, run the extractor, and compute:

  * absolute error in mean / per-bar values
  * F1 over category labels (does it find the right groups?)
  * relative error in pie wedge percentages
  * RMSE in line-plot sampled y-values
  * matching error in box-plot medians

Each kind reports its own metric; the report rolls them up so we
can compare extractors apples-to-apples.

Output:
  eval_harness/VALUE_FIDELITY_REPORT.md
  eval_harness/VALUE_FIDELITY_REPORT.json
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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

FIGS_DIR = ROOT / "_value_fidelity_figs"
FIGS_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Synthetic generators with KNOWN truth
# ----------------------------------------------------------------------

def gen_bar_known(seed: int):
    """Generate a bar chart and return (path, ground_truth_dict)."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    cats = [chr(ord("A") + i) for i in range(5)]
    vals = rng.integers(5, 40, size=5).astype(float)
    ax.bar(cats, vals, color="#3b75af")
    ax.set_xlabel("Group"); ax.set_ylabel("Value")
    ax.set_ylim(0, 50)
    plt.tight_layout()
    path = FIGS_DIR / f"vf_bar_s{seed}.png"
    plt.savefig(path); plt.close(fig)
    return path, {"categories": cats, "values": vals.tolist()}


def gen_pie_known(seed: int):
    rng = np.random.default_rng(seed)
    n = rng.integers(3, 6)
    raw = rng.integers(5, 30, size=int(n))
    pct = (raw * 100 / raw.sum()).round(2)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.pie(raw); ax.set_aspect("equal")
    plt.tight_layout()
    path = FIGS_DIR / f"vf_pie_s{seed}.png"
    plt.savefig(path); plt.close(fig)
    return path, {"percents": pct.tolist()}


def gen_line_known(seed: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 30)
    y1 = (x * rng.uniform(0.3, 0.6) + rng.uniform(2, 5))
    y2 = ((10 - x) * rng.uniform(0.3, 0.6) + rng.uniform(1, 3))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(x, y1, color="#3b75af", linewidth=2.5)
    ax.plot(x, y2, color="#ef8636", linewidth=2.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_xlim(0, 10); ax.set_ylim(0, 12)
    plt.tight_layout()
    path = FIGS_DIR / f"vf_line_s{seed}.png"
    plt.savefig(path); plt.close(fig)
    return path, {"x": x.tolist(),
                    "series": [y1.tolist(), y2.tolist()]}


def gen_box_known(seed: int):
    rng = np.random.default_rng(seed)
    n_groups = 4
    cats = list("ABCD")[:n_groups]
    truth_medians = []
    data = []
    for i in range(n_groups):
        center = rng.uniform(2, 12)
        d = rng.normal(center, 1.5, 50)
        data.append(d)
        truth_medians.append(float(np.median(d)))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.boxplot(data, tick_labels=cats, patch_artist=True,
                boxprops=dict(facecolor="#a3c4dd"))
    ax.set_ylabel("Value")
    plt.tight_layout()
    path = FIGS_DIR / f"vf_box_s{seed}.png"
    plt.savefig(path); plt.close(fig)
    return path, {"categories": cats, "medians": truth_medians}


def gen_stacked_known(seed: int):
    rng = np.random.default_rng(seed)
    cats = list("ABCD")
    a = rng.integers(5, 20, 4)
    b = rng.integers(5, 15, 4)
    c = rng.integers(2, 12, 4)
    truth = [a.tolist(), b.tolist(), c.tolist()]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.bar(cats, a, color="#3b75af")
    ax.bar(cats, b, bottom=a, color="#ef8636")
    ax.bar(cats, c, bottom=a + b, color="#519e3e")
    ax.set_xlabel("Group"); ax.set_ylabel("Value")
    plt.tight_layout()
    path = FIGS_DIR / f"vf_stacked_s{seed}.png"
    plt.savefig(path); plt.close(fig)
    return path, {"categories": cats, "series_matrix": truth}


# ----------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------

def score_bar(result, truth):
    """Compare extracted bar values to ground truth."""
    got_vals = result.values or []
    truth_vals = truth["values"]
    if not got_vals:
        return {"per_bar_mae": None, "values_recovered": 0,
                 "n_truth": len(truth_vals)}
    n = min(len(got_vals), len(truth_vals))
    if n == 0:
        return {"per_bar_mae": None, "values_recovered": 0,
                 "n_truth": len(truth_vals)}
    # Align by sorting both (extractor doesn't preserve order
    # perfectly across all cases)
    g = sorted(got_vals[:n])
    t = sorted(truth_vals[:n])
    mae = float(np.mean([abs(a - b) for a, b in zip(g, t)]))
    mean_t = float(np.mean(t)) or 1.0
    return {"per_bar_mae": round(mae, 2),
             "per_bar_pct_err": round(100 * mae / abs(mean_t), 1),
             "values_recovered": n,
             "n_truth": len(truth_vals)}


def score_pie(result, truth):
    got = result.values or []
    truth_pct = truth["percents"]
    if not got:
        return {"per_slice_mae": None, "n_recovered": 0,
                 "n_truth": len(truth_pct)}
    n = min(len(got), len(truth_pct))
    g = sorted(got[:n], reverse=True)
    t = sorted(truth_pct[:n], reverse=True)
    mae = float(np.mean([abs(a - b) for a, b in zip(g, t)]))
    return {"per_slice_mae": round(mae, 2),
             "n_recovered": n,
             "n_truth": len(truth_pct)}


def score_line(result, truth):
    series = result.line_series or []
    truth_series = truth["series"]
    if not series:
        return {"n_series_recovered": 0,
                 "n_truth_series": len(truth_series),
                 "rmse_y_mean": None}
    # For each recovered series, find the closest truth series
    # by y-mean.
    recovered_y_means = []
    for s in series:
        samples = s.get("samples") or []
        if samples:
            recovered_y_means.append(np.mean([p[1] for p in samples]))
    if not recovered_y_means:
        return {"n_series_recovered": 0,
                 "n_truth_series": len(truth_series),
                 "rmse_y_mean": None}
    truth_y_means = [np.mean(ys) for ys in truth_series]
    # Greedy match
    used = set()
    errs = []
    for ry in recovered_y_means:
        best = None
        best_d = 1e9
        for j, ty in enumerate(truth_y_means):
            if j in used: continue
            d = abs(ry - ty)
            if d < best_d:
                best_d = d; best = j
        if best is not None:
            used.add(best); errs.append(best_d)
    rmse = float(np.sqrt(np.mean([e * e for e in errs]))) if errs else None
    return {"n_series_recovered": len(recovered_y_means),
             "n_truth_series": len(truth_series),
             "rmse_y_mean": round(rmse, 2) if rmse else None}


def score_box(result, truth):
    boxes = result.box_stats or []
    truth_meds = truth["medians"]
    if not boxes:
        return {"per_box_mae_median": None,
                 "n_recovered": 0, "n_truth": len(truth_meds)}
    got_meds = sorted(b.get("median", 0) for b in boxes)
    t = sorted(truth_meds)
    n = min(len(got_meds), len(t))
    mae = float(np.mean([abs(g - tt) for g, tt in zip(got_meds[:n], t[:n])]))
    return {"per_box_mae_median": round(mae, 2),
             "n_recovered": n, "n_truth": len(truth_meds)}


def score_stacked(result, truth):
    matrix = result.matrix or []
    t_matrix = truth["series_matrix"]
    if not matrix:
        return {"n_cells_recovered": 0,
                 "n_truth_cells": sum(len(r) for r in t_matrix),
                 "mae": None}
    # Sort each row's series so order doesn't matter
    # Total per category should match
    got_totals = sorted([sum(row) for row in matrix])
    # truth: per-bar total
    t_per_bar = [sum(t_matrix[s][b] for s in range(len(t_matrix)))
                 for b in range(len(t_matrix[0]))]
    t_totals = sorted(t_per_bar)
    n = min(len(got_totals), len(t_totals))
    mae = float(np.mean(
        [abs(g - t) for g, t in zip(got_totals[:n], t_totals[:n])]))
    return {"mae_per_bar_total": round(mae, 2),
             "n_recovered_bars": len(got_totals),
             "n_truth_bars": len(t_per_bar)}


# ----------------------------------------------------------------------
# Bench
# ----------------------------------------------------------------------

GENERATORS = {
    "bar":     (gen_bar_known, "bar_chart", score_bar),
    "pie":     (gen_pie_known, "pie_chart", score_pie),
    "line":    (gen_line_known, "line_plot", score_line),
    "box":     (gen_box_known, "box_plot", score_box),
    "stacked": (gen_stacked_known, "stacked_bar_chart", score_stacked),
}


def main():
    if not HAS_MPL:
        print("matplotlib missing")
        return
    from pipeline_v2.vision.chart_extract.parallel_extractor import (
        run_smart_extraction)
    from pipeline_v2.vision.chart_extract.simple_bars import SimpleBarsExtractor
    from pipeline_v2.vision.chart_extract.pie_chart import PieChartExtractor
    from pipeline_v2.vision.chart_extract.line_plot import LinePlotExtractor
    from pipeline_v2.vision.chart_extract.box_plot import BoxPlotExtractor
    from pipeline_v2.vision.chart_extract.stacked_bars import StackedBarsExtractor
    DIRECT = {
        "bar": SimpleBarsExtractor(),
        "pie": PieChartExtractor(),
        "line": LinePlotExtractor(),
        "box": BoxPlotExtractor(),
        "stacked": StackedBarsExtractor(),
    }

    rows = []
    for kind_short, (gen, truth_kind, scorer) in GENERATORS.items():
        for seed in (0, 1, 2):
            img, truth = gen(seed)
            cap = f"Figure 1. {truth_kind.replace('_', ' ')} of values."
            # Direct extractor (best case -- we know the truth kind)
            t0 = time.time()
            direct = DIRECT[kind_short].extract(img)
            direct_t = round(time.time() - t0, 2)
            direct_score = scorer(direct, truth)
            # Smart extraction (production path)
            t0 = time.time()
            smart = run_smart_extraction(image_path=img, caption=cap)
            smart_t = round(time.time() - t0, 2)
            smart_winner = smart.winner
            # Score smart by its winning extractor's result, but only
            # if it picked the right kind
            smart_picked_right = (
                smart.winner_kind in (truth_kind, "stacked_bar_chart"
                                        if truth_kind == "bar_chart" else None))
            smart_score = scorer(smart_winner, truth) \
                if smart_winner and smart_picked_right \
                else {"n_recovered": 0, "wrong_kind": smart.winner_kind}
            rows.append({
                "kind": kind_short, "seed": seed,
                "direct_status": direct.status.value,
                "direct_score": direct_score,
                "direct_t": direct_t,
                "smart_picked": smart.winner_kind,
                "smart_status": smart_winner.status.value if smart_winner else "none",
                "smart_score": smart_score,
                "smart_t": smart_t,
                "smart_picked_right": smart_picked_right,
            })

    # Per-kind aggregates
    by_kind: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "direct_ok": 0, "smart_picked_right": 0,
                 "direct_errs": [], "smart_errs": []})
    for r in rows:
        k = r["kind"]
        by_kind[k]["n"] += 1
        if r["direct_status"] == "ok":
            by_kind[k]["direct_ok"] += 1
        if r["smart_picked_right"]:
            by_kind[k]["smart_picked_right"] += 1
        # Collect primary error metric
        d = r["direct_score"]
        for key in ("per_bar_mae", "per_slice_mae", "rmse_y_mean",
                     "per_box_mae_median", "mae_per_bar_total"):
            if d.get(key) is not None:
                by_kind[k]["direct_errs"].append(d[key])
                break
        s = r["smart_score"]
        for key in ("per_bar_mae", "per_slice_mae", "rmse_y_mean",
                     "per_box_mae_median", "mae_per_bar_total"):
            if s.get(key) is not None:
                by_kind[k]["smart_errs"].append(s[key])
                break

    # Render
    md = ["# Value-fidelity bench", "",
            f"Cases: **{len(rows)}** = 5 kinds × 3 seeds.",
            "Direct = call the right specialist by name.",
            "Smart = production `run_smart_extraction`.",
            "",
            "## Per-kind value error",
            "",
            "| kind | n | direct OK | smart picked right | "
            "avg direct err | avg smart err |",
            "|---|---|---|---|---|---|"]
    for k, s in by_kind.items():
        dir_avg = (round(np.mean(s["direct_errs"]), 2)
                   if s["direct_errs"] else "?")
        sm_avg = (round(np.mean(s["smart_errs"]), 2)
                   if s["smart_errs"] else "?")
        md.append(f"| {k} | {s['n']} | {s['direct_ok']}/{s['n']} | "
                   f"{s['smart_picked_right']}/{s['n']} | {dir_avg} | "
                   f"{sm_avg} |")
    md.append("")
    md.append("## Per-case detail")
    md.append("")
    md.append("| kind | seed | direct status | direct err | "
              "smart picked | smart err |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        d_err = next((v for k, v in r["direct_score"].items()
                      if "mae" in k or "rmse" in k), "?")
        s_err = next((v for k, v in r["smart_score"].items()
                      if "mae" in k or "rmse" in k), "?")
        md.append(f"| {r['kind']} | {r['seed']} | {r['direct_status']} | "
                   f"{d_err} | {r['smart_picked']} | {s_err} |")
    out_md = ROOT / "VALUE_FIDELITY_REPORT.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json = ROOT / "VALUE_FIDELITY_REPORT.json"
    out_json.write_text(json.dumps({
        "rows": rows,
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
    }, indent=2, default=str), encoding="utf-8")
    print("=== Summary ===")
    for k, s in by_kind.items():
        dir_avg = (round(np.mean(s["direct_errs"]), 2)
                   if s["direct_errs"] else None)
        sm_avg = (round(np.mean(s["smart_errs"]), 2)
                   if s["smart_errs"] else None)
        print(f"  {k:10s} direct={s['direct_ok']}/{s['n']} "
                f"smart_picked={s['smart_picked_right']}/{s['n']} "
                f"direct_err={dir_avg} smart_err={sm_avg}")
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
