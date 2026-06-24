"""Eval harness: download arXiv papers (PDF + LaTeX source) for ground truth.

Pulls a small fixed set of arXiv IDs spanning multiple domains. Each
paper goes into ``eval_harness/corpus/<arxiv_id>/`` containing:

  * paper.pdf           -- the rendered PDF
  * source/             -- the unpacked LaTeX source tarball
  * ground_truth.txt    -- the LaTeX rendered to plain text by pandoc

Pandoc is required for the LaTeX → text conversion. Run on a host
with internet access.

Usage:
    python3 -m eval_harness.fetch_arxiv
    python3 -m eval_harness.fetch_arxiv --ids 1706.03762 2010.11929
"""
from __future__ import annotations

import argparse
import gzip
import io
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional


from .latex_text import latex_to_text as _latex_to_text


# A small curated set: cross-domain, single-author and multi-author,
# 2-col and 1-col, with and without heavy math. All have LaTeX source
# available on arXiv (verified by hand).
#
# Sept-2026 update: expanded beyond ML/CV to cover physics, math,
# economics, biology, and a couple of long/heavy-table papers so the
# harness isn't biased towards one community's layout conventions.
DEFAULT_IDS = [
    # --- ML/AI canonical ---
    "1706.03762",   # Attention Is All You Need (Vaswani et al., 2017)
    "1810.04805",   # BERT (Devlin et al., 2018)
    "2010.11929",   # ViT (Dosovitskiy et al., 2020)
    "1503.02531",   # Distilling the Knowledge (Hinton et al., 2015)
    "1909.11942",   # ALBERT (Lan et al., 2019)
    "1406.2661",    # GANs (Goodfellow et al., 2014)
    "1312.6114",    # VAE (Kingma & Welling, 2013)
    "1512.03385",   # ResNet (He et al., 2015)
    "1409.0473",    # NMT-Attention (Bahdanau et al., 2014)
    "2005.14165",   # GPT-3 (Brown et al., 2020) -- LONG (74 pages)
    # --- Non-ML domains ---
    "0710.5491",    # Quantum Computing intro (Wilde) -- physics, heavy math
    "0805.4452",    # Math overview, single-column
    "1209.3818",    # Galaxy formation (astrophysics)
    "1610.07854",   # Economics, single-col
    "1907.05047",   # Biology (genomics)
]


ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"


def _http_get(url: str, *, timeout: int = 60) -> bytes:
    """Download a URL with a friendly User-Agent (arXiv requires one)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "pdf-to-md-eval/1.0 (research/pdf-evaluation)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_pdf(arxiv_id: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 1024:
        return True
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        out.write_bytes(_http_get(url))
        return True
    except Exception as e:
        print(f"  ! PDF fetch failed: {e}")
        return False


def fetch_source(arxiv_id: str, out_dir: Path) -> bool:
    """Download and unpack the LaTeX source tarball."""
    if out_dir.exists() and any(out_dir.rglob("*.tex")):
        return True
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        data = _http_get(url)
    except Exception as e:
        print(f"  ! source fetch failed: {e}")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    # The "e-print" endpoint serves a gzipped tar OR a single .tex.gz.
    # Try tar.gz first, then plain gzip, then raw.
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(out_dir)
        return True
    except (tarfile.TarError, gzip.BadGzipFile, EOFError):
        pass
    try:
        decompressed = gzip.decompress(data)
        # If it looks like LaTeX, save as ms.tex
        if b"\\documentclass" in decompressed[:4096] or \
                b"\\begin" in decompressed[:4096]:
            (out_dir / "ms.tex").write_bytes(decompressed)
            return True
        # Otherwise might be a raw tar
        try:
            with tarfile.open(fileobj=io.BytesIO(decompressed),
                               mode="r:") as tf:
                tf.extractall(out_dir)
            return True
        except tarfile.TarError:
            pass
    except Exception:
        pass
    # Last resort: raw bytes
    (out_dir / "raw_eprint").write_bytes(data)
    return False


def find_main_tex(source_dir: Path) -> Optional[Path]:
    """Heuristic: find the main .tex (the one with \\documentclass)."""
    tex_files = list(source_dir.rglob("*.tex"))
    if not tex_files:
        return None
    # Prefer files containing \documentclass
    for p in tex_files:
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:4096]
            if "\\documentclass" in head:
                return p
        except Exception:
            continue
    # Fallback: longest .tex file
    return max(tex_files, key=lambda p: p.stat().st_size)


def latex_to_text(main_tex: Path, out: Path) -> str:
    """Convert LaTeX → text. Returns the method used ('pandoc' or
    'regex-stripper'). Always succeeds (writes SOMETHING)."""
    return _latex_to_text(main_tex, out)


def fetch_one(arxiv_id: str) -> dict:
    paper_dir = CORPUS_DIR / arxiv_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf = paper_dir / "paper.pdf"
    src = paper_dir / "source"
    gt = paper_dir / "ground_truth.txt"
    result = {"arxiv_id": arxiv_id, "ok": False, "steps": {}}

    print(f"[{arxiv_id}] downloading PDF...")
    result["steps"]["pdf"] = fetch_pdf(arxiv_id, pdf)
    time.sleep(0.5)
    if not result["steps"]["pdf"]:
        return result

    print(f"[{arxiv_id}] downloading LaTeX source...")
    result["steps"]["source"] = fetch_source(arxiv_id, src)
    time.sleep(0.5)
    if not result["steps"]["source"]:
        return result

    print(f"[{arxiv_id}] rendering LaTeX → text...")
    main_tex = find_main_tex(src)
    if main_tex is None:
        print(f"  ! no main .tex found in {src}")
        return result
    result["steps"]["tex_main"] = str(main_tex.relative_to(paper_dir))
    method = latex_to_text(main_tex, gt)
    result["steps"]["latex_to_text"] = method
    result["latex_method"] = method

    if not gt.exists() or gt.stat().st_size < 200:
        print(f"  ! latex conversion produced <200 chars, skipping")
        return result

    result["ground_truth_chars"] = gt.stat().st_size
    result["pdf_bytes"] = pdf.stat().st_size
    result["ok"] = True
    print(f"[{arxiv_id}] OK ({result['ground_truth_chars']:,} chars GT "
          f"via {method})")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ids", nargs="+", default=DEFAULT_IDS,
                    help="arXiv IDs to fetch")
    args = p.parse_args(argv)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for aid in args.ids:
        try:
            results.append(fetch_one(aid))
        except Exception as e:
            print(f"[{aid}] FAILED: {type(e).__name__}: {e}")
            results.append({"arxiv_id": aid, "ok": False, "error": str(e)})
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} papers ready")
    import json
    (CORPUS_DIR / "_fetch_report.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
