"""Backfill caption_text on already-converted output papers.

For papers that were converted before E3 (caption_pairing.py) was
shipped, ~74% of figures have empty caption_text. This script:

  1. Finds each output/<slug>/paper.json with empty-caption figures
  2. Locates the source PDF (NFC-normalised filename match)
  3. Runs caption_pairing on the PDF to get Figure-N → caption text
  4. Merges new captions into paper.json by caption_number, only
     filling fields that are currently empty (non-destructive)

Usage:
    python3 -m pipeline_v2.caption_backfill              # whole corpus
    python3 -m pipeline_v2.caption_backfill --paper baden-bohm-2023
    python3 -m pipeline_v2.caption_backfill --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_v2.caption_pairing import (find_captions_on_page,
                                              Caption)


# ----------------------------------------------------------------------
# PDF resolution (handles NFC/NFD diacritic mismatches)
# ----------------------------------------------------------------------

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _ngrams(s: str, n: int = 4) -> set:
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def resolve_pdf(slug: str, pdf_dir: Path) -> Optional[Path]:
    target = _norm(slug.replace("-", " "))
    target_g = _ngrams(target)
    best = None; best_score = 0
    for p in pdf_dir.glob("*.pdf"):
        pg = _ngrams(_norm(p.stem))
        score = len(pg & target_g)
        if score > best_score:
            best_score = score
            best = p
    return best if best_score >= 4 else None


# ----------------------------------------------------------------------
# Caption extraction
# ----------------------------------------------------------------------

def captions_from_pdf(pdf_path: Path) -> Dict[str, str]:
    """Returns {caption_number: caption_text} for the whole PDF.
    Numbers that appear more than once are joined with ' / '."""
    try:
        import fitz
    except ImportError:
        return {}
    out: Dict[str, List[str]] = {}
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return {}
    try:
        for i in range(doc.page_count):
            page = doc[i]
            caps = find_captions_on_page(page)
            for c in caps:
                # Strip any "Fig. 3:" header from the front of the
                # captured text -- find_captions_on_page already keeps
                # the full text.
                txt = c.text.strip()
                # Strip leading "Figure 3." / "Fig 3:" header
                txt = re.sub(
                    r"^\s*(?:Figure|Fig\.?)\s+\d+[A-Za-z]?\s*[.:\-—)]?\s*",
                    "", txt, flags=re.IGNORECASE)
                txt = " ".join(txt.split())
                if not txt:
                    continue
                # Preserve compound numbers (e.g. OECD's "Figure 1.3"
                # → key="1.3") AND emit a fallback integer key
                # ("Figure 1.3" → also matchable as "1"). This lets
                # papers with simple "Figure 3" still resolve while
                # giving compound-numbered papers their own keys.
                key = c.number.strip()
                out.setdefault(key, []).append(txt)
                # ALSO store under the integer prefix as a coarse
                # fallback (multiple captions joined by ' / ').
                int_match = re.match(r"\d+", key)
                if int_match:
                    int_key = int_match.group(0)
                    if int_key != key:
                        out.setdefault(int_key, []).append(txt)
    finally:
        doc.close()
    return {k: " / ".join(v) for k, v in out.items()}


# ----------------------------------------------------------------------
# Backfill
# ----------------------------------------------------------------------

def backfill_paper(paper_dir: Path, pdf_dir: Path,
                    dry_run: bool = False) -> Dict[str, Any]:
    pj = paper_dir / "paper.json"
    if not pj.exists():
        return {"slug": paper_dir.name, "skipped": "no paper.json"}
    paper = json.loads(pj.read_text(encoding="utf-8"))
    figs = paper.get("figures") or []
    if not figs:
        return {"slug": paper_dir.name, "skipped": "no figures"}
    n_empty_before = sum(1 for f in figs if not (f.get("caption_text") or "").strip())
    if n_empty_before == 0:
        return {"slug": paper_dir.name, "skipped": "all captions present",
                "n_empty_before": 0}
    pdf = resolve_pdf(paper_dir.name, pdf_dir)
    if pdf is None:
        return {"slug": paper_dir.name,
                "skipped": "couldn't resolve source PDF"}
    captions = captions_from_pdf(pdf)
    if not captions:
        return {"slug": paper_dir.name, "skipped": "no captions found in PDF",
                "n_empty_before": n_empty_before,
                "pdf": pdf.name}
    def _derive_key(fig):
        """Best-effort: extract the figure NUMBER for matching."""
        # 1. Use caption_number if present
        num = fig.get("caption_number")
        if num is not None:
            m = re.match(r"\d+", str(num))
            if m: return m.group(0)
        # 2. Parse from alt_text like "Figure 3 (page 5)"
        alt = fig.get("alt_text") or ""
        m = re.search(r"figure\s+(\d+)", alt, re.IGNORECASE)
        if m: return m.group(1)
        m = re.search(r"fig\.?\s+(\d+)", alt, re.IGNORECASE)
        if m: return m.group(1)
        # 3. Fall back to fig-NNN id -> N
        fid = fig.get("id", "")
        m = re.match(r"fig[-_]?0*(\d+)", fid, re.IGNORECASE)
        if m: return m.group(1)
        return None

    n_filled = 0
    for f in figs:
        if (f.get("caption_text") or "").strip():
            continue
        key = _derive_key(f)
        if key is None:
            continue
        if key in captions:
            f["caption_text"] = captions[key]
            # Also set caption_number if it was missing
            if f.get("caption_number") is None:
                f["caption_number"] = key
            n_filled += 1
    if not dry_run and n_filled > 0:
        pj.write_text(json.dumps(paper, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return {"slug": paper_dir.name,
             "pdf": pdf.name,
             "n_figs": len(figs),
             "n_empty_before": n_empty_before,
             "n_captions_found_in_pdf": len(captions),
             "n_filled": n_filled,
             "dry_run": dry_run}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    p.add_argument("--paper", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    targets = []
    if args.paper:
        targets = [args.output_dir / args.paper]
    else:
        targets = [d for d in sorted(args.output_dir.iterdir())
                   if d.is_dir() and not d.name.startswith("_")]
    total_filled = 0
    total_empty_before = 0
    n_papers_helped = 0
    rows = []
    for t in targets:
        r = backfill_paper(t, args.pdf_dir, dry_run=args.dry_run)
        rows.append(r)
        if "skipped" in r:
            print(f"SKIP {r['slug']}: {r['skipped']}")
            continue
        total_filled += r["n_filled"]
        total_empty_before += r["n_empty_before"]
        if r["n_filled"] > 0:
            n_papers_helped += 1
        print(f"{r['slug']}: filled {r['n_filled']}/{r['n_empty_before']} "
                f"empty (from {r['n_captions_found_in_pdf']} captions in PDF)")
    print(f"\n=== Summary ===")
    print(f"Papers improved: {n_papers_helped}/{len(targets)}")
    print(f"Captions filled: {total_filled}")
    print(f"Was empty before: {total_empty_before}")
    if total_empty_before:
        print(f"Backfill rate: {round(100 * total_filled / total_empty_before, 1)}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
