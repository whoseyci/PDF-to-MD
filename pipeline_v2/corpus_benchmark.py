"""E9 -- Honest corpus-level benchmark.

Aggregates the per-paper outputs in ``output/`` into a single
benchmark report. Unlike a fresh re-run (which would take hours),
this reads what's already in place AND optionally exercises
post-processors (figure_refs, dashboard, caption_pairing) on every
paper to confirm they all work.

Writes:
    output/CORPUS_BENCHMARK.json   -- machine-readable metrics
    output/CORPUS_BENCHMARK.md     -- human-readable summary

Usage:
    python3 -m pipeline_v2.corpus_benchmark
    python3 -m pipeline_v2.corpus_benchmark --link-figures
    python3 -m pipeline_v2.corpus_benchmark --pair-captions
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_v2.dashboard import collect_all, aggregate, render_markdown
from pipeline_v2.figure_refs import link_paper_dir


def _resolve_pdf(slug: str, pdf_dir: Path) -> Optional[Path]:
    """Best-effort match a paper slug back to its source PDF.

    Handles NFD vs NFC unicode mismatches and slug variations.
    """
    def norm(s): return unicodedata.normalize("NFC", s).lower()
    candidates = list(pdf_dir.glob("*.pdf"))
    target = norm(slug.replace("-", " "))
    # Fuzzy: pick the PDF whose normalised name shares the most
    # 4-character ngrams with the slug.
    def grams(s, n=4):
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    target_g = grams(target)
    best = None; best_score = 0
    for p in candidates:
        pg = grams(norm(p.stem))
        score = len(pg & target_g)
        if score > best_score:
            best_score = score; best = p
    if best_score >= 4:
        return best
    return None


def run_benchmark(output_dir: Path,
                  pdf_dir: Path,
                  *,
                  link_figures: bool = False,
                  pair_captions: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    # 1. Optionally re-link figure references
    if link_figures:
        from pipeline_v2.figure_refs import link_paper_dir
        n_linked = 0
        for d in sorted(output_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            try:
                link_paper_dir(d)
                n_linked += 1
            except Exception:
                pass

    # 2. Optionally run caption pairing as a sanity check
    pair_stats = None
    if pair_captions:
        from pipeline_v2.caption_pairing import pair_pdf
        per_paper_counts = []
        for d in sorted(output_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            pdf = _resolve_pdf(d.name, pdf_dir)
            if pdf is None:
                continue
            try:
                pairs = pair_pdf(pdf)
                n_paired = sum(1 for p in pairs if p.region is not None)
                per_paper_counts.append({
                    "slug": d.name,
                    "n_captions": len(pairs),
                    "n_paired": n_paired,
                    "pct_paired": round(100 * n_paired / max(1, len(pairs)), 1),
                })
            except Exception as e:
                per_paper_counts.append({"slug": d.name, "error": str(e)})
        total_caps = sum(p.get("n_captions", 0) for p in per_paper_counts)
        total_paired = sum(p.get("n_paired", 0) for p in per_paper_counts)
        pair_stats = {
            "per_paper": per_paper_counts,
            "total_captions": total_caps,
            "total_paired": total_paired,
            "pct_paired": round(100 * total_paired / max(1, total_caps), 1),
        }

    # 3. Dashboard aggregation
    rows = collect_all(output_dir)
    agg = aggregate(rows)

    elapsed = round(time.time() - t0, 2)

    report = {
        "benchmark_seconds": elapsed,
        "options": {"link_figures": link_figures,
                     "pair_captions": pair_captions},
        "aggregates": agg,
        "caption_pairing": pair_stats,
        "n_papers": len(rows),
    }

    return report, rows


def render(report: Dict[str, Any], rows) -> str:
    md_dash = render_markdown(rows, report["aggregates"])
    lines = [
        "# Corpus benchmark (E9)",
        "",
        f"Run time: **{report['benchmark_seconds']}s** "
        f"over {report['n_papers']} papers.",
        "",
        f"Options: `link_figures={report['options']['link_figures']}` "
        f"`pair_captions={report['options']['pair_captions']}`",
        "",
    ]
    if report.get("caption_pairing"):
        cp = report["caption_pairing"]
        lines.append("## Caption pairing (E3) summary")
        lines.append("")
        lines.append(f"* Total captions detected: {cp['total_captions']}")
        lines.append(f"* Successfully paired to a region: {cp['total_paired']}")
        lines.append(f"* Overall pairing rate: **{cp['pct_paired']}%**")
        lines.append("")
        lines.append("| Paper | captions | paired | % |")
        lines.append("|---|---|---|---|")
        for p in cp["per_paper"]:
            if "error" in p:
                lines.append(f"| {p['slug']} | -- | -- | _{p['error'][:30]}_ |")
            else:
                lines.append(f"| {p['slug']} | {p['n_captions']} | "
                              f"{p['n_paired']} | {p['pct_paired']} |")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(md_dash)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    p.add_argument("--link-figures", action="store_true")
    p.add_argument("--pair-captions", action="store_true")
    args = p.parse_args(argv)
    report, rows = run_benchmark(args.output_dir, args.pdf_dir,
                                    link_figures=args.link_figures,
                                    pair_captions=args.pair_captions)
    md = render(report, rows)
    out_md = args.output_dir / "CORPUS_BENCHMARK.md"
    out_json = args.output_dir / "CORPUS_BENCHMARK.json"
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_md} and {out_json}")
    print(f"benchmark: {report['n_papers']} papers in "
          f"{report['benchmark_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
