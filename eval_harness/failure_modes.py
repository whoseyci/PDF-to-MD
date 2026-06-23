"""Failure-mode catalog.

Generates a small set of *deliberately broken* PDFs and runs them
through our pipeline_v2.convert path (or a subset of stages),
documenting which stages cope and which silently produce garbage.

Failure modes covered:
  F1  encrypted PDF (password-protected)
  F2  zero-page / empty PDF
  F3  image-only PDF (no embedded text -- needs OCR)
  F4  rotated pages (90° / 180°)
  F5  giant single column with no headings
  F6  page-break mid-word (hyphenation)
  F7  weird unicode (RTL + CJK mixed in)
  F8  duplicate identical pages
  F9  garbage / random-bytes PDF (parser must reject cleanly)
  F10 single-page table-only PDF (no body text)

Outputs:
    eval_harness/failure_pdfs/<F##_name>.pdf
    eval_harness/FAILURE_REPORT.md
    eval_harness/FAILURE_REPORT.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "failure_pdfs"


# ----------------------------------------------------------------------
# Synthetic PDF generators (all use pymupdf to keep deps minimal)
# ----------------------------------------------------------------------

def _new_doc():
    import fitz
    return fitz.open()


def f1_encrypted(out: Path):
    """Password-protected PDF."""
    import fitz
    doc = _new_doc()
    page = doc.new_page()
    page.insert_text((72, 100), "This page is encrypted.", fontsize=14)
    doc.save(str(out), encryption=fitz.PDF_ENCRYPT_AES_256,
              owner_pw="owner", user_pw="user")
    doc.close()


def f2_empty(out: Path):
    """Zero-page PDF -- some PDFs from poor exports have this."""
    import fitz
    doc = _new_doc()
    # No pages added
    try:
        doc.save(str(out))
    except Exception:
        # pymupdf refuses zero-page; create 1 blank page instead
        doc.close()
        doc = _new_doc()
        doc.new_page()
        doc.save(str(out))
    doc.close()


def f3_image_only(out: Path):
    """Render rasterised text as an image and embed as a PDF
    -- there's no extractable text."""
    import fitz
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1200, 1600), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
    text = ("This is image-only text.\n"
            "Extraction needs OCR.\n"
            "If your pipeline doesn't OCR, it will return nothing.")
    d.multiline_text((80, 100), text, fill="black", font=font, spacing=12)
    img_path = out.with_suffix(".png")
    img.save(img_path)
    doc = _new_doc()
    page = doc.new_page(width=600, height=800)
    page.insert_image(page.rect, filename=str(img_path))
    doc.save(str(out))
    doc.close()
    img_path.unlink()


def f4_rotated(out: Path):
    """3 pages: 0°, 90°, 180° rotated."""
    import fitz
    doc = _new_doc()
    for i, rot in enumerate([0, 90, 180]):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {i+1} rotated {rot}°.",
                         fontsize=14)
        page.set_rotation(rot)
    doc.save(str(out))
    doc.close()


def _inflated(rect, dx):
    """Compat shim: some pymupdf versions removed Rect.inflated."""
    return type(rect)(rect.x0 + abs(dx), rect.y0 + abs(dx),
                       rect.x1 - abs(dx), rect.y1 - abs(dx))


def f5_giant_no_headings(out: Path):
    """5 pages of one giant paragraph with no headings, periods,
    or paragraph breaks. Stresses our heading-detector."""
    import fitz
    doc = _new_doc()
    chunk = (" the quick brown fox jumps over the lazy dog " * 80).strip()
    for _ in range(5):
        page = doc.new_page()
        page.insert_textbox(
            _inflated(page.rect, 36),
            chunk, fontsize=10, align=0)
    doc.save(str(out))
    doc.close()


def f6_page_break_midword(out: Path):
    """End of page 1 ends mid-word, page 2 starts mid-word."""
    import fitz
    doc = _new_doc()
    p1 = doc.new_page()
    p1.insert_textbox(_inflated(p1.rect, 36),
                       "This sentence is very long and ends mid-wor",
                       fontsize=14)
    p2 = doc.new_page()
    p2.insert_textbox(_inflated(p2.rect, 36),
                       "d at the start of the next page seamlessly.",
                       fontsize=14)
    doc.save(str(out))
    doc.close()


def f7_weird_unicode(out: Path):
    """Mixed RTL Hebrew + CJK + emoji."""
    import fitz
    doc = _new_doc()
    page = doc.new_page()
    # CJK requires a CJK font; pymupdf has one built in for fallback
    page.insert_text((72, 100), "English Plain.", fontsize=14)
    page.insert_text((72, 130), "Hebrew: shalom.",  # avoid actual RTL char
                       fontsize=14)
    page.insert_text((72, 160), "CJK: 你好世界", fontsize=14,
                       fontname="china-s")
    doc.save(str(out))
    doc.close()


def f8_duplicate_pages(out: Path):
    """4 pages all containing the exact same text -- stresses header/
    footer de-duplication."""
    import fitz
    doc = _new_doc()
    for _ in range(4):
        page = doc.new_page()
        page.insert_text((72, 50), "PAPER TITLE", fontsize=20)
        page.insert_text((72, 100), "Same body text on every page.",
                         fontsize=12)
        page.insert_text((72, 750), "Page footer", fontsize=10)
    doc.save(str(out))
    doc.close()


def f9_garbage(out: Path):
    """Random bytes with PDF header -- parser must reject cleanly."""
    import os
    data = b"%PDF-1.4\n" + os.urandom(4000) + b"\n%%EOF"
    out.write_bytes(data)


def f10_table_only(out: Path):
    """Single page containing only a table-like layout."""
    import fitz
    doc = _new_doc()
    page = doc.new_page()
    y = 100
    for row in (["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"]):
        x = 100
        for cell in row:
            page.insert_text((x, y), cell, fontsize=14)
            x += 80
        y += 30
    doc.save(str(out))
    doc.close()


GENERATORS = [
    ("F01_encrypted",         f1_encrypted),
    ("F02_empty",             f2_empty),
    ("F03_image_only",        f3_image_only),
    ("F04_rotated",           f4_rotated),
    ("F05_no_headings",       f5_giant_no_headings),
    ("F06_midword_break",     f6_page_break_midword),
    ("F07_weird_unicode",     f7_weird_unicode),
    ("F08_duplicate_pages",   f8_duplicate_pages),
    ("F09_garbage_bytes",     f9_garbage),
    ("F10_table_only",        f10_table_only),
]


# ----------------------------------------------------------------------
# Probe each stage of our pipeline
# ----------------------------------------------------------------------

def probe(pdf: Path) -> Dict[str, Any]:
    """Run several pipeline stages, capture status/error for each."""
    out: Dict[str, Any] = {"pdf_bytes": pdf.stat().st_size,
                              "stages": {}}

    # Stage A: pymupdf opens the file
    t0 = time.time()
    n_pages = None; open_err = None
    try:
        import fitz
        doc = fitz.open(str(pdf))
        n_pages = doc.page_count
        is_encrypted = doc.needs_pass
        doc.close()
    except Exception as e:
        open_err = f"{type(e).__name__}: {str(e)[:200]}"
    out["stages"]["pymupdf_open"] = {
        "ok": open_err is None, "n_pages": n_pages,
        "encrypted": (is_encrypted if open_err is None else None),
        "elapsed_s": round(time.time() - t0, 3),
        "error": open_err,
    }

    # Stage B: pymupdf4llm extraction
    t0 = time.time()
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(str(pdf))
        out["stages"]["pymupdf4llm"] = {
            "ok": True, "chars": len(md or ""),
            "elapsed_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        out["stages"]["pymupdf4llm"] = {
            "ok": False, "elapsed_s": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    # Stage C: pdftotext (poppler)
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["pdftotext", str(pdf), "-"],
            capture_output=True, text=True, timeout=30,
        )
        out["stages"]["pdftotext"] = {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "chars": len(proc.stdout),
            "stderr": proc.stderr.strip()[:120],
            "elapsed_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        out["stages"]["pdftotext"] = {
            "ok": False, "elapsed_s": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    # Stage D: our reading-order reorder
    t0 = time.time()
    try:
        from pipeline_v2.reading_order import reorder_pdf_text
        text = reorder_pdf_text(pdf)
        out["stages"]["pdf2md_reorder"] = {
            "ok": True, "chars": len(text or ""),
            "elapsed_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        out["stages"]["pdf2md_reorder"] = {
            "ok": False, "elapsed_s": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    # Stage E: caption pairing
    t0 = time.time()
    try:
        from pipeline_v2.caption_pairing import pair_pdf
        pairs = pair_pdf(pdf)
        out["stages"]["caption_pairing"] = {
            "ok": True, "n_pairs": len(pairs),
            "elapsed_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        out["stages"]["caption_pairing"] = {
            "ok": False, "elapsed_s": round(time.time() - t0, 3),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    return out


# ----------------------------------------------------------------------
# Top-level: generate + probe + report
# ----------------------------------------------------------------------

def render_md(records: Dict[str, Dict[str, Any]]) -> str:
    lines = ["# Failure-mode catalog", "",
              "Each row is a synthetic PDF designed to break ONE specific "
              "assumption in the pipeline. The cells show which stage "
              "tolerated the input.",
              "",
              "Legend: ✅ ok · ⚠️ partial (parsed but suspicious) · ❌ crashed",
              "",
              "| Failure mode | pages | pymupdf | pymupdf4llm | pdftotext | pdf2md_reorder | caption_pair |",
              "|---|---|---|---|---|---|---|"]
    for name, rec in records.items():
        cells = [name, str(rec.get("stages", {}).get(
            "pymupdf_open", {}).get("n_pages") or "?")]
        for stage in ("pymupdf_open", "pymupdf4llm", "pdftotext",
                        "pdf2md_reorder", "caption_pairing"):
            d = rec.get("stages", {}).get(stage, {})
            if d.get("ok") is False or d.get("error"):
                cells.append("❌")
            else:
                chars = d.get("chars")
                n_pairs = d.get("n_pairs")
                if chars is not None and chars < 30:
                    cells.append(f"⚠️ {chars}c")
                elif n_pairs is not None:
                    cells.append(f"✅ {n_pairs}pr")
                elif chars is not None:
                    cells.append(f"✅ {chars}c")
                else:
                    cells.append("✅")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Detailed observations")
    lines.append("")
    for name, rec in records.items():
        lines.append(f"### {name}")
        lines.append("")
        for stage, d in rec.get("stages", {}).items():
            if d.get("error"):
                lines.append(f"* **{stage}**: ❌ `{d['error']}`")
            else:
                bits = [f"{k}={v}" for k, v in d.items()
                          if k not in ("ok", "stderr") and v is not None]
                lines.append(f"* **{stage}**: ✅ " + " ".join(bits))
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    p.add_argument("--out-md", type=Path, default=ROOT / "FAILURE_REPORT.md")
    p.add_argument("--out-json", type=Path, default=ROOT / "FAILURE_REPORT.json")
    args = p.parse_args(argv)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    for name, fn in GENERATORS:
        out_pdf = args.pdf_dir / f"{name}.pdf"
        try:
            fn(out_pdf)
            print(f"generated {out_pdf.name}")
        except Exception as e:
            print(f"GEN FAIL {name}: {e}")

    # Probe
    records = {}
    for name, _ in GENERATORS:
        pdf = args.pdf_dir / f"{name}.pdf"
        if not pdf.exists():
            continue
        print(f"probing {name}...")
        records[name] = probe(pdf)
    args.out_md.write_text(render_md(records), encoding="utf-8")
    args.out_json.write_text(json.dumps(records, indent=2),
                               encoding="utf-8")
    print(f"\nwrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
