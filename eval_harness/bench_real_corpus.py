"""Run smart-extraction on the real 35-paper PDF corpus.

For each PDF:
  1. Extract every raster image to a temp dir
  2. For each image: run run_smart_extraction with the figure's
     existing caption (if any) from paper.json
  3. Record the outcome: extracted_kind, status, confidence,
     n_extractors_run, elapsed_s

Aggregates:
  * % of figures that get an OK extraction
  * % of figures classified as decorative (correctly skipped)
  * runtime distribution
  * by-kind breakdown
  * worst-N papers list

Cost: ~5 seconds per figure × 476 figures = ~40 min. Bounded; can be
chunked with --max-papers / --max-figures-per-paper.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _ngrams(s: str, n: int = 4) -> set:
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def resolve_pdf(slug: str, pdf_dir: Path) -> Optional[Path]:
    target = _norm(slug.replace("-", " "))
    target_g = _ngrams(target)
    best = None
    best_score = 0
    for p in pdf_dir.glob("*.pdf"):
        pg = _ngrams(_norm(p.stem))
        score = len(pg & target_g)
        if score > best_score:
            best_score = score
            best = p
    return best if best_score >= 4 else None


def extract_figures_to_dir(pdf_path: Path, out_dir: Path,
                           max_figs: int = 0,
                           min_size_kb: int = 5,
                           skip_first_page: bool = True
                           ) -> List[Dict[str, Any]]:
    """Pull every raster image from the PDF into out_dir. Returns
    a list of {file, page, xref, bytes}.

    Filters:
      * tiny images (< min_size_kb) -- usually icons/inline glyphs
      * page 1 images (skip_first_page=True default) -- usually
        publisher branding, journal cover art. Real data figures
        live on body pages.
    """
    import fitz
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_xrefs = set()
    try:
        for p in range(doc.page_count):
            if skip_first_page and p == 0:
                continue
            page = doc[p]
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    name = f"p{p+1:03d}_x{xref}.png"
                    fp = out_dir / name
                    pix.save(str(fp))
                    sz = fp.stat().st_size
                    pix = None
                    if sz >= min_size_kb * 1024:
                        out.append({
                            "file": str(fp), "page": p + 1,
                            "xref": xref, "bytes": sz,
                        })
                    else:
                        fp.unlink()
                    if max_figs and len(out) >= max_figs:
                        return out
                except Exception:
                    continue
    finally:
        doc.close()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    p.add_argument("--max-papers", type=int, default=0)
    p.add_argument("--max-figures-per-paper", type=int, default=0)
    p.add_argument("--out", type=Path,
                    default=ROOT / "REAL_CORPUS_REPORT.md")
    p.add_argument("--out-json", type=Path,
                    default=ROOT / "REAL_CORPUS_REPORT.json")
    p.add_argument("--per-figure-timeout", type=float, default=30.0)
    p.add_argument("--use-cached", action="store_true",
                    help="Skip re-extraction if /home/user/.tmp/_corpus_figs "
                         "has cached extracts (useful for re-runs)")
    args = p.parse_args(argv)

    from pipeline_v2.vision.chart_extract.parallel_extractor import (
        run_smart_extraction)

    papers = [d for d in sorted(args.output_dir.iterdir())
              if d.is_dir() and not d.name.startswith("_")]
    if args.max_papers:
        papers = papers[: args.max_papers]

    work_root = Path("/home/user/.tmp/_corpus_figs")
    work_root.mkdir(parents=True, exist_ok=True)

    results = []
    all_rows = []
    t_start = time.time()
    for i, paper_dir in enumerate(papers, 1):
        slug = paper_dir.name
        pj = paper_dir / "paper.json"
        if not pj.exists():
            continue
        paper = json.loads(pj.read_text(encoding="utf-8"))
        pdf = resolve_pdf(slug, args.pdf_dir)
        if pdf is None:
            print(f"[{i}/{len(papers)}] SKIP {slug}: no source PDF")
            continue
        print(f"[{i}/{len(papers)}] {slug}: extracting from {pdf.name}")
        # Build caption lookup: figure id -> caption_text
        cap_by_xref: Dict[int, str] = {}
        cap_by_seq: List[str] = []
        for f in paper.get("figures", []):
            cap = (f.get("caption_text") or "").strip()
            cap_by_seq.append(cap)
        # Extract figures
        fig_dir = work_root / slug
        if args.use_cached and fig_dir.exists() and any(fig_dir.glob("*.png")):
            figs_info = [{"file": str(f), "page": 0, "xref": 0,
                            "bytes": f.stat().st_size}
                         for f in sorted(fig_dir.glob("*.png"))]
        else:
            for old in fig_dir.glob("*.png"):
                old.unlink()
            figs_info = extract_figures_to_dir(
                pdf, fig_dir, max_figs=args.max_figures_per_paper)
        if not figs_info:
            print(f"  no figures extracted")
            continue
        print(f"  got {len(figs_info)} figures, running smart-extraction")
        paper_stats = {
            "slug": slug, "n_figs_extracted": len(figs_info),
            "by_status": Counter(), "by_kind": Counter(),
            "elapsed_s": 0.0, "n_ok": 0, "n_partial": 0,
            "n_unsupported_decorative": 0,
        }
        for j, fi in enumerate(figs_info):
            img = Path(fi["file"])
            # Use sequential caption mapping (fig-001 → cap_by_seq[0])
            cap = cap_by_seq[j] if j < len(cap_by_seq) else ""
            t0 = time.time()
            try:
                tr = run_smart_extraction(image_path=img, caption=cap)
            except Exception as e:
                row = {
                    "slug": slug, "fig_idx": j, "img": img.name,
                    "caption": cap[:80], "ERROR": str(e)[:120],
                    "elapsed_s": round(time.time() - t0, 2),
                }
                all_rows.append(row)
                continue
            elapsed = round(time.time() - t0, 2)
            paper_stats["elapsed_s"] += elapsed
            winner = tr.winner
            status = winner.status.value if winner else "none"
            kind = tr.winner_kind or "?"
            paper_stats["by_status"][status] += 1
            paper_stats["by_kind"][kind] += 1
            if status == "ok":
                paper_stats["n_ok"] += 1
            elif status == "partial":
                paper_stats["n_partial"] += 1
            elif status == "unsupported" and kind == "decorative":
                paper_stats["n_unsupported_decorative"] += 1
            all_rows.append({
                "slug": slug, "fig_idx": j, "img": img.name,
                "caption": (cap or "")[:80],
                "kind": kind, "status": status,
                "confidence": winner.confidence if winner else 0.0,
                "n_extractors_run": tr.n_extractors_run,
                "arbitration_reason": tr.arbitration_reason[:80],
                "elapsed_s": elapsed,
            })
        paper_stats["pct_ok"] = round(
            100 * paper_stats["n_ok"] / paper_stats["n_figs_extracted"], 1
        ) if paper_stats["n_figs_extracted"] else 0
        results.append(paper_stats)

    total_elapsed = round(time.time() - t_start, 1)
    n_figs = sum(r["n_figs_extracted"] for r in results)
    n_ok = sum(r["n_ok"] for r in results)
    n_partial = sum(r["n_partial"] for r in results)
    n_dec = sum(r["n_unsupported_decorative"] for r in results)
    n_other = n_figs - n_ok - n_partial - n_dec

    # By-kind across whole corpus
    by_kind = Counter()
    for r in results:
        for k, v in r["by_kind"].items():
            by_kind[k] += v
    by_status = Counter()
    for r in results:
        for s, v in r["by_status"].items():
            by_status[s] += v

    md = ["# Real corpus extraction bench", "",
            f"Ran `run_smart_extraction` on **{n_figs}** raster images "
            f"extracted from **{len(results)}** PDFs.",
            f"Total runtime: **{total_elapsed}s** "
            f"(~{round(total_elapsed/max(1,n_figs),2)}s per figure).",
            "",
            "## Aggregate",
            "",
            f"* **OK extractions:** {n_ok}/{n_figs} = "
            f"**{round(100*n_ok/max(1,n_figs),1)}%**",
            f"* PARTIAL extractions: {n_partial}/{n_figs} = "
            f"{round(100*n_partial/max(1,n_figs),1)}%",
            f"* Classified as decorative (skipped): {n_dec}/{n_figs} = "
            f"{round(100*n_dec/max(1,n_figs),1)}%",
            f"* Other (NO_BARS / NO_AXIS / ERROR): {n_other}/{n_figs} = "
            f"{round(100*n_other/max(1,n_figs),1)}%",
            "",
            "## By winning kind",
            "",
            "| kind | n |",
            "|---|---|"]
    for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append("## By status")
    md.append("")
    md.append("| status | n |")
    md.append("|---|---|")
    for s, v in sorted(by_status.items(), key=lambda kv: -kv[1]):
        md.append(f"| {s} | {v} |")
    md.append("")
    md.append("## Per-paper")
    md.append("")
    md.append("| paper | n figs | OK | partial | decorative | other | "
              "pct_ok | total s |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda r: -r["pct_ok"]):
        n_other_p = (r["n_figs_extracted"] - r["n_ok"] - r["n_partial"]
                      - r["n_unsupported_decorative"])
        md.append(f"| {r['slug']} | {r['n_figs_extracted']} | "
                   f"{r['n_ok']} | {r['n_partial']} | "
                   f"{r['n_unsupported_decorative']} | {n_other_p} | "
                   f"{r['pct_ok']}% | {round(r['elapsed_s'], 1)} |")
    md.append("")
    md.append("## Worst-N papers")
    md.append("")
    md.append("| paper | n figs | pct_ok | top failure status |")
    md.append("|---|---|---|---|")
    for r in sorted(results, key=lambda r: r["pct_ok"])[:10]:
        # Top non-ok status
        top_fail = "?"
        for s, v in sorted(r["by_status"].items(), key=lambda kv: -kv[1]):
            if s != "ok":
                top_fail = f"{s} ({v})"
                break
        md.append(f"| {r['slug']} | {r['n_figs_extracted']} | "
                   f"{r['pct_ok']}% | {top_fail} |")

    args.out.write_text("\n".join(md), encoding="utf-8")
    args.out_json.write_text(json.dumps({
        "n_figures": n_figs, "n_ok": n_ok, "n_partial": n_partial,
        "n_decorative": n_dec, "n_other": n_other,
        "total_elapsed_s": total_elapsed,
        "by_kind": dict(by_kind), "by_status": dict(by_status),
        "per_paper": [{**r, "by_status": dict(r["by_status"]),
                       "by_kind": dict(r["by_kind"])} for r in results],
        "rows": all_rows,
    }, indent=2), encoding="utf-8")
    print(f"\n{n_ok}/{n_figs} OK ({round(100*n_ok/max(1,n_figs),1)}%)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
