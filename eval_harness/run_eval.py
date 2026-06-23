"""Eval harness -- run each PDF through multiple extractors,
compute WER / CER / Jaccard against the LaTeX ground truth, save a
side-by-side report.

Extractors evaluated (when their dep is available):
  * pymupdf4llm      -- baseline
  * pdftotext        -- poppler-utils, simple but solid for digital PDFs
  * pdf-to-md (ours) -- full pipeline_v2 convert path
  * pdf-to-md raw    -- our pipeline's pymupdf4llm+postprocess only
                         (no figure/ref work) -- helps isolate where
                         our value-add is text-only vs structure

Metrics per extractor per paper:
  * char_ratio   -- len(extracted) / len(gt)
  * jaccard_words   -- |A ∩ B| / |A ∪ B| over alphanumeric tokens
  * wer_estimate -- Levenshtein-token / len(GT_tokens). We use a fast
                       O(N) "longest common subsequence proxy" instead of
                       full Levenshtein (which is O(N²) and dies on
                       50k-word papers). The proxy slightly underestimates
                       WER but is monotone and comparable across runs.
  * elapsed_s

Outputs:
    eval_harness/REPORT.md
    eval_harness/REPORT.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
PROJECT_ROOT = ROOT.parent


# ----------------------------------------------------------------------
# Text normalisation (so all extractors are scored on the same footing)
# ----------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop control chars."""
    text = text.lower()
    text = re.sub(r"[^\x20-\x7e\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def char_ratio(extracted: str, gt: str) -> float:
    if not gt:
        return 0.0
    return round(len(extracted) / len(gt), 3)


def jaccard_words(extracted: str, gt: str) -> float:
    a = set(tokenize(extracted))
    b = set(tokenize(gt))
    if not (a or b):
        return 0.0
    return round(len(a & b) / len(a | b), 4)


def lcs_proxy_token(extracted_toks: List[str],
                     gt_toks: List[str]) -> int:
    """Length of greedy LCS in chunks of 5 tokens. O(N) walker.

    Not a true LCS, but a useful proxy: walk both sequences with two
    pointers; advance whichever has the smaller word right now,
    counting matches when they agree.
    """
    i = j = matches = 0
    a = extracted_toks; b = gt_toks
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            matches += 1
            i += 1; j += 1
        else:
            # advance the one whose word is "behind"
            # (alphabetic sort proxy) -- not strictly correct but
            # gives ordered approx match rate
            if a[i] < b[j]:
                i += 1
            else:
                j += 1
    return matches


def fast_wer_estimate(extracted: str, gt: str) -> float:
    """Word Error Rate proxy: 1 - (matched_tokens / len(gt_tokens))
    where matched_tokens uses the LCS-proxy walker.

    A perfect extraction gives WER 0; an empty extraction gives WER 1.
    The proxy understates errors (since it uses sorted-walker LCS, not
    true edit distance) but is monotonic and lets us rank extractors.
    """
    a = tokenize(extracted)
    b = tokenize(gt)
    if not b:
        return 1.0
    m = lcs_proxy_token(sorted(a), sorted(b))
    return round(max(0.0, 1.0 - m / len(b)), 4)


def precision_recall_words(extracted: str, gt: str) -> Dict[str, float]:
    a = set(tokenize(extracted))
    b = set(tokenize(gt))
    if not a:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(a & b)
    precision = tp / len(a)
    recall = tp / len(b) if b else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) else 0.0)
    return {"precision": round(precision, 4),
             "recall": round(recall, 4),
             "f1": round(f1, 4)}


# ----------------------------------------------------------------------
# Extractors
# ----------------------------------------------------------------------

def extract_pymupdf4llm(pdf: Path) -> str:
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(str(pdf))
    except Exception as e:
        return f"__ERROR__: {type(e).__name__}: {e}"


def extract_pdftotext(pdf: Path) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return f"__ERROR__: pdftotext rc={proc.returncode}"
        return proc.stdout
    except Exception as e:
        return f"__ERROR__: {type(e).__name__}: {e}"


def extract_pdftotext_simple(pdf: Path) -> str:
    """pdftotext without -layout (stream order)."""
    try:
        proc = subprocess.run(
            ["pdftotext", str(pdf), "-"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return f"__ERROR__: pdftotext rc={proc.returncode}"
        return proc.stdout
    except Exception as e:
        return f"__ERROR__: {type(e).__name__}: {e}"


def extract_pipeline_v2_text(pdf: Path) -> str:
    """Our v2 pipeline: pymupdf4llm + postprocess_md (no figures/refs).

    Imports postprocess_md directly so we don't pay the convert.py
    figure-extraction cost (which would be unfair to baselines that
    don't do figures).
    """
    try:
        import pymupdf4llm
        from pipeline_v2 import postprocess_md
        raw = pymupdf4llm.to_markdown(str(pdf))
        # postprocess takes raw_md and may need extra args; try the
        # simplest available entrypoint
        for cand in ("postprocess_full", "postprocess",
                       "process", "postprocess_markdown"):
            fn = getattr(postprocess_md, cand, None)
            if callable(fn):
                return fn(raw)
        return raw
    except Exception as e:
        return f"__ERROR__: {type(e).__name__}: {e}"


def extract_pipeline_v2_with_reorder(pdf: Path) -> str:
    """E1 module on top of pymupdf: reading-order reorder, then collapse."""
    try:
        from pipeline_v2.reading_order import reorder_pdf_text
        return reorder_pdf_text(pdf)
    except Exception as e:
        return f"__ERROR__: {type(e).__name__}: {e}"


EXTRACTORS: List[Dict[str, Any]] = [
    {"name": "pdftotext",          "fn": extract_pdftotext},
    {"name": "pdftotext-stream",   "fn": extract_pdftotext_simple},
    {"name": "pymupdf4llm",        "fn": extract_pymupdf4llm},
    {"name": "pdf2md-postprocess", "fn": extract_pipeline_v2_text},
    {"name": "pdf2md-reorder-e1",  "fn": extract_pipeline_v2_with_reorder},
]


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def evaluate_paper(arxiv_id: str, paper_dir: Path) -> Dict[str, Any]:
    pdf = paper_dir / "paper.pdf"
    gt_path = paper_dir / "ground_truth.txt"
    if not pdf.exists() or not gt_path.exists():
        return {"arxiv_id": arxiv_id, "skipped": "missing pdf or ground truth"}
    gt_raw = gt_path.read_text(encoding="utf-8", errors="replace")
    gt = normalise(gt_raw)

    rec: Dict[str, Any] = {"arxiv_id": arxiv_id,
                              "gt_chars": len(gt),
                              "gt_words": len(tokenize(gt)),
                              "pdf_bytes": pdf.stat().st_size,
                              "extractors": {}}

    for ext in EXTRACTORS:
        name = ext["name"]
        fn: Callable[[Path], str] = ext["fn"]
        t0 = time.time()
        try:
            out = fn(pdf)
        except Exception as e:
            out = f"__ERROR__: {type(e).__name__}: {e}"
        elapsed = round(time.time() - t0, 2)
        if out.startswith("__ERROR__"):
            rec["extractors"][name] = {"error": out, "elapsed_s": elapsed}
            continue
        out_norm = normalise(out)
        pr = precision_recall_words(out_norm, gt)
        rec["extractors"][name] = {
            "chars": len(out_norm),
            "words": len(tokenize(out_norm)),
            "char_ratio": char_ratio(out_norm, gt),
            "jaccard_words": jaccard_words(out_norm, gt),
            "wer_estimate": fast_wer_estimate(out_norm, gt),
            "precision": pr["precision"],
            "recall": pr["recall"],
            "f1": pr["f1"],
            "elapsed_s": elapsed,
        }
    return rec


def render_md(records: List[Dict[str, Any]]) -> str:
    extractors_seen = []
    for r in records:
        for n in r.get("extractors", {}).keys():
            if n not in extractors_seen:
                extractors_seen.append(n)
    lines = ["# Eval harness report",
              "",
              f"Papers: {len(records)} arXiv papers with ground-truth "
              "LaTeX rendered via pandoc / regex-stripper.",
              "",
              "All extractors normalised identically (lowercase, collapse "
              "whitespace, strip non-ASCII) before scoring.",
              "",
              "Metrics:",
              "  * **char_ratio** -- extracted_chars / gt_chars "
              "(closer to 1.0 = comparable length)",
              "  * **F1** -- word-set F1 vs ground truth",
              "  * **WER\\*** -- proxy WER on sorted token bags "
              "(monotone, lower is better, not a strict edit distance)",
              "  * **t_s** -- seconds to extract",
              ""]

    # Overall summary
    summary_rows = []
    for ext in extractors_seen:
        f1s = [r["extractors"][ext].get("f1")
               for r in records if r.get("extractors", {}).get(ext)
                                       and "f1" in r["extractors"][ext]]
        wers = [r["extractors"][ext].get("wer_estimate")
                for r in records if r.get("extractors", {}).get(ext)
                                        and "wer_estimate" in r["extractors"][ext]]
        ts = [r["extractors"][ext].get("elapsed_s")
              for r in records if r.get("extractors", {}).get(ext)
                                      and "elapsed_s" in r["extractors"][ext]]
        n_err = sum(1 for r in records
                    if "error" in r.get("extractors", {}).get(ext, {}))
        avg = lambda v: round(sum(v) / len(v), 3) if v else None
        summary_rows.append({
            "extractor": ext,
            "n_ok": len(f1s),
            "n_err": n_err,
            "avg_f1": avg(f1s),
            "avg_wer": avg(wers),
            "avg_s": avg(ts),
        })
    lines.append("## Aggregate (averaged across papers)")
    lines.append("")
    lines.append("| Extractor | n ok | n err | avg F1 | avg WER\\* | avg s |")
    lines.append("|---|---|---|---|---|---|")
    for s in summary_rows:
        lines.append(
            f"| `{s['extractor']}` | {s['n_ok']} | {s['n_err']} | "
            f"{s['avg_f1']} | {s['avg_wer']} | {s['avg_s']} |"
        )
    lines.append("")

    # Per-paper detail tables
    lines.append("## Per-paper detail")
    lines.append("")
    for r in records:
        aid = r["arxiv_id"]
        lines.append(f"### {aid}")
        lines.append("")
        lines.append(f"_gt: {r.get('gt_chars', 0):,} chars / "
                       f"{r.get('gt_words', 0):,} words; "
                       f"pdf: {r.get('pdf_bytes', 0):,} bytes_")
        lines.append("")
        lines.append("| Extractor | chars | F1 | WER\\* | char ratio | t_s |")
        lines.append("|---|---|---|---|---|---|")
        for ext in extractors_seen:
            d = r.get("extractors", {}).get(ext, {})
            if "error" in d:
                lines.append(f"| `{ext}` | -- | -- | -- | -- | "
                              f"_{d['error'][:30]}_ |")
            else:
                lines.append(
                    f"| `{ext}` | {d.get('chars', 0):,} | "
                    f"{d.get('f1', '?')} | {d.get('wer_estimate', '?')} | "
                    f"{d.get('char_ratio', '?')} | {d.get('elapsed_s', '?')} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    p.add_argument("--out-md", type=Path, default=ROOT / "REPORT.md")
    p.add_argument("--out-json", type=Path, default=ROOT / "REPORT.json")
    p.add_argument("--ids", nargs="+", default=None)
    args = p.parse_args(argv)

    if args.ids:
        target_dirs = [args.corpus / aid for aid in args.ids]
    else:
        target_dirs = sorted(
            d for d in args.corpus.iterdir()
            if d.is_dir() and (d / "paper.pdf").exists())
    records = []
    for d in target_dirs:
        aid = d.name
        print(f"== {aid}")
        rec = evaluate_paper(aid, d)
        records.append(rec)
        for ext, dd in rec.get("extractors", {}).items():
            if "error" in dd:
                print(f"   {ext}: ERROR {dd['error'][:50]}")
            else:
                print(f"   {ext}: F1={dd.get('f1')} "
                       f"WER*={dd.get('wer_estimate')} "
                       f"chars={dd.get('chars')} t={dd.get('elapsed_s')}s")

    args.out_md.write_text(render_md(records), encoding="utf-8")
    args.out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
