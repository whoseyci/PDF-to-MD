"""E7 -- Quality dashboard generator.

Walks the ``output/`` tree and emits an aggregated Markdown report
``output/QUALITY_DASHBOARD.md`` with:

  * one row per paper, columns for the key extraction metrics
  * corpus-wide totals + averages
  * a "worst-N papers" list (sorted by missing-figure-count + low coverage)

Usage:
    python3 -m pipeline_v2.dashboard
    python3 -m pipeline_v2.dashboard --output-dir output --out QUALITY.md
    python3 -m pipeline_v2.dashboard --json    # also dump JSON snapshot

Pure stdlib + ``json`` + ``pathlib``. No new dependencies. Safe to
re-run any time -- it never modifies per-paper outputs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Per-paper record
# ----------------------------------------------------------------------

@dataclass
class PaperStats:
    slug: str
    n_pages: int = 0
    raw_words: int = 0
    md_words: int = 0
    coverage: float = 0.0
    conversion_seconds: float = 0.0
    title: str = ""
    doi: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    n_authors: int = 0
    n_figures: int = 0
    n_figures_with_caption: int = 0
    n_chart_extracted: int = 0
    n_mermaid_extracted: int = 0
    n_alt_text: int = 0
    n_references: int = 0
    n_refs_with_doi: int = 0
    n_refs_verified: int = 0
    n_citations_linked: int = 0
    structure_score: int = 0
    confidence: str = ""
    has_docling: bool = False
    has_paper_md: bool = False
    issues: List[str] = field(default_factory=list)


def _safe_load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_paper_stats(paper_dir: Path) -> PaperStats:
    s = PaperStats(slug=paper_dir.name)
    stats = _safe_load(paper_dir / "stats.json") or {}
    paper = _safe_load(paper_dir / "paper.json") or {}
    refs = _safe_load(paper_dir / "references.json") or []

    s.n_pages = int(stats.get("n_pages") or paper.get("n_pages") or 0)
    s.raw_words = int(stats.get("raw_word_count") or 0)
    s.md_words = int(stats.get("md_word_count") or 0)
    s.coverage = float(stats.get("coverage_ratio") or 0.0)
    s.conversion_seconds = float(stats.get("conversion_seconds") or 0.0)
    s.title = str(stats.get("title") or paper.get("metadata", {}).get("title") or "")
    s.doi = stats.get("doi") or paper.get("metadata", {}).get("doi")
    s.year = stats.get("year") or paper.get("metadata", {}).get("year")
    s.journal = stats.get("journal") or paper.get("metadata", {}).get("journal")
    s.n_authors = int(stats.get("n_authors") or 0)
    s.structure_score = int(stats.get("structure_score") or 0)
    s.confidence = str(stats.get("confidence") or "")
    s.n_citations_linked = int(stats.get("n_citations_linked") or 0)

    figs = paper.get("figures") or []
    s.n_figures = len(figs)
    s.n_figures_with_caption = sum(1 for f in figs if f.get("caption_text"))
    for f in figs:
        ed = f.get("extracted_data") or {}
        if ed.get("status") == "ok" or f.get("chart_extracted"):
            s.n_chart_extracted += 1
        if f.get("mermaid_extracted"):
            s.n_mermaid_extracted += 1
        if f.get("alt_text") and not (f.get("chart_extracted") or f.get("mermaid_extracted")):
            s.n_alt_text += 1

    if isinstance(refs, list):
        s.n_references = len(refs)
        for r in refs:
            if r.get("doi"):
                s.n_refs_with_doi += 1
            v = r.get("verification") or {}
            if v.get("status") in ("verified", "match"):
                s.n_refs_verified += 1

    s.has_docling = (paper_dir / "paper.docling.json").exists()
    s.has_paper_md = (paper_dir / "paper.md").exists()

    # Heuristic issues -- accumulated for the "worst papers" list.
    if not s.has_paper_md:
        s.issues.append("missing paper.md")
    if s.coverage and s.coverage < 0.85:
        s.issues.append(f"low coverage {s.coverage:.2f}")
    if s.n_figures == 0 and s.n_pages > 2:
        s.issues.append("no figures detected")
    if s.n_figures and s.n_figures_with_caption < s.n_figures:
        s.issues.append(f"{s.n_figures - s.n_figures_with_caption} fig(s) w/o caption")
    if s.n_references == 0 and s.n_pages > 2:
        s.issues.append("no references parsed")
    return s


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def collect_all(output_dir: Path) -> List[PaperStats]:
    rows: List[PaperStats] = []
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            continue
        if not (d / "stats.json").exists() and not (d / "paper.json").exists():
            continue
        rows.append(collect_paper_stats(d))
    return rows


def aggregate(rows: List[PaperStats]) -> Dict[str, Any]:
    if not rows:
        return {"n_papers": 0}
    def _sum(attr): return sum(getattr(r, attr) for r in rows)
    def _mean(attr):
        vals = [getattr(r, attr) for r in rows if getattr(r, attr)]
        return statistics.mean(vals) if vals else 0.0
    n_papers = len(rows)
    return {
        "n_papers": n_papers,
        "total_pages": _sum("n_pages"),
        "total_figures": _sum("n_figures"),
        "total_references": _sum("n_references"),
        "total_citations_linked": _sum("n_citations_linked"),
        "mean_coverage": round(_mean("coverage"), 3),
        "mean_conversion_seconds": round(_mean("conversion_seconds"), 2),
        "pct_figures_with_caption": _pct(
            _sum("n_figures_with_caption"), _sum("n_figures")),
        "pct_figures_chart_extracted": _pct(
            _sum("n_chart_extracted"), _sum("n_figures")),
        "pct_figures_mermaid": _pct(
            _sum("n_mermaid_extracted"), _sum("n_figures")),
        "pct_figures_alt_text_only": _pct(
            _sum("n_alt_text"), _sum("n_figures")),
        "pct_refs_with_doi": _pct(
            _sum("n_refs_with_doi"), _sum("n_references")),
        "pct_refs_verified": _pct(
            _sum("n_refs_verified"), _sum("n_references")),
        "n_with_docling": sum(1 for r in rows if r.has_docling),
        "n_high_confidence": sum(1 for r in rows if r.confidence == "high"),
    }


def _pct(num: int, den: int) -> float:
    return round(100 * num / den, 1) if den else 0.0


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------

_COLUMNS = [
    ("slug", "Paper", 24),
    ("n_pages", "Pg", 3),
    ("n_figures", "Fig", 3),
    ("n_chart_extracted", "Ch", 3),
    ("n_mermaid_extracted", "Mm", 3),
    ("n_references", "Ref", 4),
    ("n_refs_with_doi", "DOI", 4),
    ("n_refs_verified", "✓", 3),
    ("n_citations_linked", "Cit", 4),
    ("coverage", "Cov", 5),
    ("conversion_seconds", "s", 6),
    ("structure_score", "Str", 3),
    ("confidence", "Conf", 6),
]


def render_markdown(rows: List[PaperStats],
                    agg: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Pipeline-v2 quality dashboard")
    lines.append("")
    lines.append(
        f"Auto-generated by ``pipeline_v2/dashboard.py`` over "
        f"`{agg.get('n_papers', 0)}` papers."
    )
    lines.append("")
    lines.append("## Corpus aggregates")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Papers processed | {agg.get('n_papers', 0)} |")
    lines.append(f"| Total pages | {agg.get('total_pages', 0):,} |")
    lines.append(f"| Total figures | {agg.get('total_figures', 0):,} |")
    lines.append(f"| Total references | {agg.get('total_references', 0):,} |")
    lines.append(f"| Total citations linked | {agg.get('total_citations_linked', 0):,} |")
    lines.append(f"| Mean coverage | {agg.get('mean_coverage', 0):.3f} |")
    lines.append(f"| Mean conversion seconds | {agg.get('mean_conversion_seconds', 0):.1f} |")
    lines.append(f"| % figures w/ caption | {agg.get('pct_figures_with_caption', 0)} |")
    lines.append(f"| % figures with chart_extract OK | {agg.get('pct_figures_chart_extracted', 0)} |")
    lines.append(f"| % figures with mermaid emitted | {agg.get('pct_figures_mermaid', 0)} |")
    lines.append(f"| % figures alt-text only | {agg.get('pct_figures_alt_text_only', 0)} |")
    lines.append(f"| % refs with DOI | {agg.get('pct_refs_with_doi', 0)} |")
    lines.append(f"| % refs verified (Crossref/OpenAlex) | {agg.get('pct_refs_verified', 0)} |")
    lines.append(f"| Papers with docling.json | {agg.get('n_with_docling', 0)} |")
    lines.append(f"| Papers with high confidence | {agg.get('n_high_confidence', 0)} |")
    lines.append("")

    # Per-paper table
    header = "| " + " | ".join(c[1] for c in _COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in _COLUMNS) + "|"
    lines.append("## Per-paper detail")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = []
        for attr, _, _ in _COLUMNS:
            v = getattr(r, attr, "")
            if isinstance(v, float):
                v = f"{v:.2f}"
            elif v is None:
                v = ""
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Worst-N papers
    ranked = sorted(rows, key=lambda r: (
        len(r.issues),
        -((r.coverage or 0.0)),
        -r.n_pages,
    ), reverse=True)
    worst = [r for r in ranked if r.issues][:10]
    lines.append("## Worst-N papers (most issues)")
    lines.append("")
    if not worst:
        lines.append("_No issues detected._")
    else:
        lines.append("| Paper | Issues |")
        lines.append("|---|---|")
        for r in worst:
            lines.append(f"| {r.slug} | {'; '.join(r.issues)} |")
    lines.append("")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--out", type=Path, default=None,
                    help="Markdown output path (default: <output-dir>/QUALITY_DASHBOARD.md)")
    p.add_argument("--json", action="store_true",
                    help="Also dump a machine-readable QUALITY_DASHBOARD.json")
    args = p.parse_args(argv)

    rows = collect_all(args.output_dir)
    agg = aggregate(rows)
    md = render_markdown(rows, agg)

    out = args.out or (args.output_dir / "QUALITY_DASHBOARD.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out} ({len(rows)} papers)")

    if args.json:
        jpath = out.with_suffix(".json")
        jpath.write_text(json.dumps({
            "aggregates": agg,
            "papers": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {jpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
