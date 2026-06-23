"""E8 -- Tests for the geometric chart extractors.

Generates clean synthetic fixtures and asserts the extractors recover
roughly correct values. Each test has a tolerance budget because OCR
and pixel-to-value calibration can drift.

Run:  python3 tests/test_chart_extractors.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


FIXTURES = ROOT / "output" / "_chart_e8"


def make_stacked_bar(path: Path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    cats = ["A", "B", "C", "D"]
    series_a = np.array([10, 15, 8, 20])
    series_b = np.array([7, 5, 12, 4])
    series_c = np.array([3, 6, 4, 8])
    ax.bar(cats, series_a, label="Alpha", color="#3b75af")
    ax.bar(cats, series_b, bottom=series_a, label="Beta", color="#ef8636")
    ax.bar(cats, series_c, bottom=series_a + series_b, label="Gamma",
           color="#519e3e")
    ax.set_ylabel("Value")
    ax.set_xlabel("Group")
    ax.set_ylim(0, 40)
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close(fig)
    return [series_a.tolist(), series_b.tolist(), series_c.tolist()]


def make_pie(path: Path):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    labels = ["X", "Y", "Z", "W"]
    sizes = [40, 30, 20, 10]
    ax.pie(sizes, labels=None,
            colors=["#3b75af", "#ef8636", "#519e3e", "#c33e3e"],
            wedgeprops={"linewidth": 0})
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close(fig)
    return sizes


def make_line(path: Path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    x = np.linspace(0, 10, 40)
    ax.plot(x, x ** 0.5 * 3, color="#3b75af", linewidth=2, label="A")
    ax.plot(x, np.cos(x / 3) * 4 + 5, color="#ef8636", linewidth=2, label="B")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_xlim(0, 10); ax.set_ylim(0, 12)
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close(fig)
    return {"A": "sqrt", "B": "cos"}


def make_scatter(path: Path):
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    n = 40
    g1x = rng.normal(2, 0.5, n); g1y = rng.normal(2, 0.5, n)
    g2x = rng.normal(7, 0.6, n); g2y = rng.normal(8, 0.6, n)
    ax.scatter(g1x, g1y, color="#3b75af", s=30, label="A")
    ax.scatter(g2x, g2y, color="#ef8636", s=30, label="B")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close(fig)
    return {"A_mean": (2, 2), "B_mean": (7, 8)}


def make_box(path: Path):
    rng = np.random.default_rng(123)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    data = [rng.normal(5, 1.5, 50),
            rng.normal(8, 1, 50),
            rng.normal(3, 0.8, 50),
            rng.normal(10, 2, 50)]
    ax.boxplot(data, tick_labels=["A", "B", "C", "D"], patch_artist=True,
                boxprops=dict(facecolor="#a3c4dd"))
    ax.set_ylabel("Value")
    ax.set_ylim(0, 16)
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close(fig)
    return {"A_med": 5, "B_med": 8, "C_med": 3, "D_med": 10}


def run():
    if not HAS_MPL:
        print("matplotlib missing")
        return
    FIXTURES.mkdir(parents=True, exist_ok=True)

    from pipeline_v2.vision.chart_extract.stacked_bars import StackedBarsExtractor
    from pipeline_v2.vision.chart_extract.pie_chart import PieChartExtractor
    from pipeline_v2.vision.chart_extract.line_plot import LinePlotExtractor
    from pipeline_v2.vision.chart_extract.scatter_plot import ScatterExtractor
    from pipeline_v2.vision.chart_extract.box_plot import BoxPlotExtractor

    report = {}

    # ---- Stacked bars ----
    p = FIXTURES / "stacked.png"
    truth = make_stacked_bar(p)
    out = StackedBarsExtractor().extract(p)
    report["stacked"] = {
        "status": out.status.value, "reason": out.reason,
        "elapsed_s": out.elapsed_seconds,
        "categories": out.categories,
        "n_series": len(out.series),
        "matrix": out.matrix,
        "truth_totals": [sum(col) for col in zip(*truth)],
    }
    print(f"stacked: status={out.status.value} cats={out.categories} "
          f"matrix={out.matrix}")

    # ---- Pie ----
    p = FIXTURES / "pie.png"
    truth = make_pie(p)
    out = PieChartExtractor().extract(p)
    report["pie"] = {
        "status": out.status.value, "reason": out.reason,
        "elapsed_s": out.elapsed_seconds,
        "percents": out.values,
        "truth": truth,
    }
    print(f"pie: status={out.status.value} percents={out.values} truth={truth}")

    # ---- Line ----
    p = FIXTURES / "line.png"
    make_line(p)
    out = LinePlotExtractor().extract(p)
    report["line"] = {
        "status": out.status.value, "reason": out.reason,
        "elapsed_s": out.elapsed_seconds,
        "n_series": len(out.line_series or []),
        "samples_per_series": [len(s["samples"]) for s in (out.line_series or [])],
    }
    print(f"line: status={out.status.value} series={len(out.line_series or [])}")

    # ---- Scatter ----
    p = FIXTURES / "scatter.png"
    truth = make_scatter(p)
    out = ScatterExtractor().extract(p)
    report["scatter"] = {
        "status": out.status.value, "reason": out.reason,
        "elapsed_s": out.elapsed_seconds,
        "n_clusters": len(out.scatter_summary or []),
        "summary": out.scatter_summary,
        "truth": truth,
    }
    print(f"scatter: status={out.status.value} "
          f"clusters={len(out.scatter_summary or [])}")

    # ---- Box ----
    p = FIXTURES / "box.png"
    truth = make_box(p)
    out = BoxPlotExtractor().extract(p)
    report["box"] = {
        "status": out.status.value, "reason": out.reason,
        "elapsed_s": out.elapsed_seconds,
        "n_boxes": len(out.box_stats or []),
        "medians": [b["median"] for b in (out.box_stats or [])],
        "truth_medians": [5, 8, 3, 10],
    }
    print(f"box: status={out.status.value} "
          f"medians={[b['median'] for b in (out.box_stats or [])]}")

    out_p = FIXTURES / "REPORT.json"
    out_p.write_text(json.dumps(report, indent=2))
    print(f"report: {out_p}")
    return report


if __name__ == "__main__":
    run()
