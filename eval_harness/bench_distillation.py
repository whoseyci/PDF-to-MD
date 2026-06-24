"""E16 — Measure how many corpus captions the rule-based 'student'
can handle without invoking the VLM teacher."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline_v2.vision.caption_distill import (assess_caption,
                                                    distill_alt_text,
                                                    DistillStats)


def main():
    output_dir = ROOT / "output"
    stats = DistillStats()
    per_paper = []
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        pj = d / "paper.json"
        if not pj.exists():
            continue
        paper = json.loads(pj.read_text(encoding="utf-8"))
        figs = paper.get("figures") or []
        if not figs:
            continue
        paper_stats = {"slug": d.name, "n_figs": len(figs),
                         "student_ok": 0, "teacher_needed": 0,
                         "empty_caption": 0}
        for f in figs:
            caption = f.get("caption_text") or ""
            r = distill_alt_text(caption=caption,
                                   fig_ocr_text=None,
                                   image_path=None,
                                   teacher_fn=None)  # measure decision only
            stats.add(r)
            if r.source == "student":
                paper_stats["student_ok"] += 1
            elif r.source == "empty":
                paper_stats["empty_caption"] += 1
            else:
                paper_stats["teacher_needed"] += 1
        per_paper.append(paper_stats)

    summary = stats.summary()
    # Total figures, % saved
    n_total = summary["n_total"]
    pct_student = summary["teacher_calls_saved_pct"]

    # Estimated time saved if teacher = ~60s/call (typical Gemma 4 alt-text)
    teacher_per_call_s = 60
    if n_total:
        time_saved_without = stats.n_student * teacher_per_call_s
    else:
        time_saved_without = 0

    out = {"summary": summary, "per_paper": per_paper,
            "estimated_teacher_seconds_saved": time_saved_without,
            "teacher_seconds_per_call_assumption": teacher_per_call_s}

    md = ["# Distillation bench (E16)", "",
            f"Corpus: {n_total} figures across "
            f"{sum(1 for p in per_paper if p['n_figs'])} papers.",
            "",
            "## Verdict",
            "",
            f"* **Student (rule-based) handles {stats.n_student}/{n_total} "
            f"= {pct_student}% of figures** without needing the VLM teacher.",
            f"* Estimated time saved (assuming {teacher_per_call_s}s/teacher call): "
            f"**{time_saved_without:,}s = ~{time_saved_without // 60} minutes**",
            f"* Captions that are completely empty: {paper_stats_total('empty_caption', per_paper)}",
            "",
            "## Per-paper",
            "",
            "| Paper | n figs | student | teacher needed | empty caption |",
            "|---|---|---|---|---|"]
    for p in per_paper:
        md.append(f"| {p['slug']} | {p['n_figs']} | {p['student_ok']} | "
                   f"{p['teacher_needed']} | {p['empty_caption']} |")
    md_out = ROOT / "eval_harness" / "DISTILLATION_REPORT.md"
    md_out.write_text("\n".join(md), encoding="utf-8")
    json_out = ROOT / "eval_harness" / "DISTILLATION_REPORT.json"
    json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nstudent handles {pct_student}% of figures")
    print(f"estimated teacher time saved: ~{time_saved_without // 60} minutes")


def paper_stats_total(key, per_paper):
    return sum(p[key] for p in per_paper)


if __name__ == "__main__":
    main()
