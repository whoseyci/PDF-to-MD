"""Generate a 'scanned' version of an existing arXiv paper.

Rasterises each page of the source PDF at low DPI then re-embeds the
images into a new PDF -- the result has NO extractable text, only
images. Used to test the OCR-fallback path of the smart dispatcher.

Usage:
    python3 -m eval_harness.make_scanned 1503.02531 --pages 8
"""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"


def rasterise(src_pdf: Path, out_pdf: Path,
                *, dpi: int = 150, max_pages: int = 0):
    import fitz
    src = fitz.open(str(src_pdf))
    out = fitz.open()
    n_pages = src.page_count if not max_pages else min(max_pages, src.page_count)
    for i in range(n_pages):
        page = src[i]
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        new_page = out.new_page(width=page.rect.width,
                                 height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=png_bytes)
    out.save(str(out_pdf))
    out.close(); src.close()
    return n_pages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arxiv_id")
    p.add_argument("--pages", type=int, default=0,
                    help="Max pages to rasterise (0 = all)")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    src_dir = CORPUS / args.arxiv_id
    src = src_dir / "paper.pdf"
    if not src.exists():
        print(f"No source PDF at {src}")
        return 1
    out_dir = CORPUS / f"{args.arxiv_id}_scanned"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "paper.pdf"
    n = rasterise(src, out_pdf, dpi=args.dpi, max_pages=args.pages)
    # Copy ground truth across (we use the same GT)
    gt = src_dir / "ground_truth.txt"
    if gt.exists():
        (out_dir / "ground_truth.txt").write_text(
            gt.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"rasterised {n} pages -> {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
