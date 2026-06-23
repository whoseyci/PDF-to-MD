"""E4 -- Arrow direction accuracy test.

Generates 10 synthetic flowcharts with known arrow directions, runs
the diagram_extract pipeline, and reports direction-correctness rate.
"""
from __future__ import annotations
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, FancyArrowPatch, Rectangle
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def gen_arrow_diagram(out_path: Path, n_nodes: int = 3,
                       layout: str = "lr", seed: int = 0):
    """Generate a synthetic diagram with N nodes connected by arrows
    in a known direction. Returns the ground-truth edge list
    [(src_label, dst_label), ...]."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)

    fig.set_size_inches(8, 4)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    if layout == "lr":
        xs = np.linspace(1.6, 10.4, n_nodes)
        ys = np.full(n_nodes, 3.0)
    elif layout == "tb":
        xs = np.full(n_nodes, 6.0)
        ys = np.linspace(5.5, 0.5, n_nodes)
    else:  # mixed
        xs = rng.uniform(1.5, 10.5, n_nodes)
        ys = rng.uniform(1.0, 5.0, n_nodes)
    label_words = ["Start", "Step1", "Step2", "Step3", "Step4", "End"]
    labels = label_words[:n_nodes]
    centres = []
    for x, y, lab in zip(xs, ys, labels):
        w, h = 1.6, 0.9
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                                  fill=False, edgecolor="black", linewidth=2.5))
        ax.text(x, y, lab, ha="center", va="center", fontsize=13,
                fontweight="bold")
        centres.append((x, y))

    truth = []
    for i in range(n_nodes - 1):
        x1, y1 = centres[i]; x2, y2 = centres[i + 1]
        # Shrink to bbox edge
        dx = x2 - x1; dy = y2 - y1
        L = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / L, dy / L
        sx, sy = x1 + ux * 0.9, y1 + uy * 0.55
        ex, ey = x2 - ux * 0.9, y2 - uy * 0.55
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", lw=2,
                                       color="black", mutation_scale=20))
        truth.append((labels[i], labels[i + 1]))

    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return truth


def run():
    if not HAS_MPL:
        print("matplotlib missing, skipping")
        return
    from pipeline_v2.vision.diagram_extract import extract_diagram

    outdir = ROOT / "output" / "_arrow_eval"
    outdir.mkdir(parents=True, exist_ok=True)

    cases = []
    for n in (2, 3, 4):
        for layout in ("lr", "tb"):
            for seed in (0, 1):
                cases.append((n, layout, seed))
    cases = cases[:10]  # cap at 10

    total_edges = 0
    correct_dirs = 0
    per_case = []
    for n, layout, seed in cases:
        name = f"arrow_n{n}_{layout}_s{seed}"
        img = outdir / f"{name}.png"
        truth = gen_arrow_diagram(img, n_nodes=n, layout=layout, seed=seed)
        result = extract_diagram(img)
        edges = result.edges
        nodes_by_id = {n.id: (n.label or n.id).strip()
                       for n in result.nodes}
        got = []
        for e in edges:
            if not e.directed:
                continue
            sl = nodes_by_id.get(e.src, "")
            dl = nodes_by_id.get(e.dst, "")
            if sl and dl:
                got.append((sl, dl))
        matched = sum(1 for t in truth if t in got)
        total_edges += len(truth)
        correct_dirs += matched
        per_case.append({
            "case": name,
            "truth": truth,
            "got_directed": got,
            "n_edges_total": len(edges),
            "matched": matched,
            "n_nodes_detected": len(result.nodes),
        })

    pct = round(100 * correct_dirs / total_edges, 1) if total_edges else 0
    report = {
        "total_truth_edges": total_edges,
        "correct_directions": correct_dirs,
        "pct_correct": pct,
        "per_case": per_case,
    }
    out = outdir / "REPORT.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"E4 arrow direction: {correct_dirs}/{total_edges} = {pct}% correct")
    print(f"report: {out}")
    return report


if __name__ == "__main__":
    run()
