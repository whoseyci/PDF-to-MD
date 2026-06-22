"""Build a polished master README + aggregate report."""
import json
from pathlib import Path
from collections import Counter

OUT = Path('/home/user/output')


def main():
    papers = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        stats_p = d / "stats.json"
        if not stats_p.exists():
            continue
        s = json.load(open(stats_p))
        papers.append({"slug": d.name, **s})
    
    # Aggregate
    confs = Counter(p["confidence"] for p in papers)
    totals = {
        "papers": len(papers),
        "pages": sum(p.get("n_pages", 0) for p in papers),
        "references": sum(p.get("n_references", 0) for p in papers),
        "linked_citations": sum(p.get("n_citations_linked", 0) for p in papers),
        "figures": sum(p.get("n_figures", 0) for p in papers),
    }
    
    # README sections
    md = []
    md.append("# Converted Paper Corpus (v2)\n")
    md.append(f"Converted **{totals['papers']}** academic PDFs into clean markdown + structured JSON.\n")
    
    md.append("## Aggregate stats\n")
    md.append(f"- **{totals['papers']}** papers, **{totals['pages']:,}** total pages")
    md.append(f"- **{totals['references']:,}** references parsed across the corpus")
    md.append(f"- **{totals['linked_citations']:,}** in-text citations linked to references")
    md.append(f"- **{totals['figures']:,}** figures extracted")
    md.append(f"- Confidence: {confs.get('high', 0)} high, {confs.get('medium', 0)} medium, {confs.get('low', 0)} low\n")
    
    md.append("## What's in each paper directory\n")
    md.append("- `paper.md` — clean human-readable markdown (math notation preserved, cite-linked)")
    md.append("- `paper.json` — fully structured (per-page text, refs, metadata, figures)")
    md.append("- `references.json` — parsed bibliography (id, text, DOI, year, first-author)")
    md.append("- `provenance.json` — per-page provenance map")
    md.append("- `stats.json` — quality metrics for this paper")
    md.append("- `figures/` — extracted figures as PNG")
    md.append("- `raw_text.txt` — pdftotext baseline (for QA)\n")
    
    md.append("## Papers\n")
    md.append("Sorted alphabetically. Click slug to open.\n")
    md.append("| # | Slug | Title | Year | DOI | Pages | Refs | Cites linked | Figs | Coverage | Conf |")
    md.append("|--:|---|---|--:|---|--:|--:|--:|--:|--:|:--:|")
    
    for i, p in enumerate(sorted(papers, key=lambda x: x["slug"]), 1):
        title = (p.get("title") or "—")[:60]
        title = title.replace("|", "\\|").replace("\n", " ")
        year = p.get("year") or "—"
        doi = p.get("doi") or "—"
        if p.get("doi"):
            doi = f"[{p['doi'][:25]}{'…' if len(p['doi']) > 25 else ''}](https://doi.org/{p['doi']})"
        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(p["confidence"], "⚪")
        md.append(
            f"| {i} | [`{p['slug']}`](./{p['slug']}/paper.md) | {title} | {year} | {doi} | "
            f"{p['n_pages']} | {p['n_references']} | {p['n_citations_linked']} | "
            f"{p['n_figures']} | {p['coverage_ratio']} | {conf_emoji} |"
        )
    
    md.append("\n## Notes on pipeline (v2)\n")
    md.append("Built on `pymupdf4llm` + custom postprocessing. Key fixes vs v1:\n")
    md.append("- ✅ `<br>`-stacked tables reshaped into proper multi-row markdown tables")
    md.append("- ✅ Picture-text blocks (OCR'd from table-images) rendered as fenced code blocks")
    md.append("- ✅ Three citation styles linked: `(Author, Year)`, `Author (Year)`, `[N]`")
    md.append("- ✅ Numeric citation linking only fires when refs ARE numbered (no false positives on author footnotes)")
    md.append("- ✅ Multi-cite syntax: `(Smith, 2024; Jones, 2025)` → each linked separately")
    md.append("- ✅ Better title extraction (skips journal banners; first real heading)")
    md.append("- ✅ Authors detected as a list, footnote markers stripped")
    md.append("- ✅ Page header/footer deduplication (running titles removed)")
    md.append("- ✅ DOI auto-detected and rendered as clickable link")
    md.append("- ✅ Junk tables filtered (title blocks, author lists, single-cell pseudo-tables)")
    md.append("- ✅ Large PDFs (>100 pages) processed in chunked mode (memory-safe)")
    md.append("- ✅ Resumable: re-running skips already-done papers")
    
    md.append("\n## Known limitations\n")
    md.append("- Picture-text blocks (OCR'd tables) are preserved as fixed-width text;")
    md.append("  full structured reconstruction without original column positions is unreliable.")
    md.append("  The corresponding figure PNG is always kept too.")
    md.append("- Multi-language papers (translated abstracts in Spanish/Chinese) may show")
    md.append("  coverage > 1 or < 1 because pymupdf4llm extracts only Latin scripts cleanly.")
    md.append("- Citation linking quality depends on a clean references section; some")
    md.append("  papers use atypical formats and a portion of cites remain unlinked.\n")
    
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "_index.json").write_text(
        json.dumps({"totals": totals, "confidences": dict(confs), "papers": papers},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"Built {OUT/'README.md'}")
    print(f"  {totals['papers']} papers, {totals['pages']} pages")
    print(f"  {totals['references']} refs, {totals['linked_citations']} cites linked, {totals['figures']} figures")
    print(f"  Confidence: {dict(confs)}")


if __name__ == "__main__":
    main()
