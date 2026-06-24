"""E3 -- PDFigCapX-style figure ↔ caption pairing.

Re-implements the core PDFigCapX heuristic in pure python on top of
pymupdf. Compared to "nearest text below the image" pairing (what we
do today), this version:

  1. Finds every "Figure N" / "Fig. N" caption header on the page.
  2. Finds every image bbox AND every large empty rectangle ("negative
     space") on the page -- the page region a figure occupies even
     when no raster image exists.
  3. For each caption header, finds the closest negative-space region
     whose bbox is in the expected direction (typically above the
     caption text) AND whose width matches the caption's column.
  4. Pairs them up. Unpaired captions default back to nearest-image.

Pure stdlib + pymupdf.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_CAPTION_RE = re.compile(
    # Accept: "Figure 3", "Fig. 3", "Figure 3.2", "Box 3.1",
    # "Annex Figure 5", "Table 4" (callers can filter); supports
    # OECD-style 'Figure 3.1' compound numbers.
    r"^\s*(?:Annex\s+)?(?:Figure|Fig\.|Fig|Chart|Box)\s+"
    r"(\d+(?:\.\d+)?[A-Za-z]?)[.:\)]?\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Caption:
    page: int
    number: str
    text: str
    bbox: Tuple[float, float, float, float]   # x0,y0,x1,y1


@dataclass
class Region:
    page: int
    bbox: Tuple[float, float, float, float]
    is_image: bool       # True = raster image, False = empty-space figure
    image_id: Optional[str] = None


@dataclass
class Pairing:
    page: int
    caption: Caption
    region: Optional[Region]
    distance: float
    method: str   # "nearest-above" | "nearest-image" | "unpaired"


# ----------------------------------------------------------------------
# Extraction primitives
# ----------------------------------------------------------------------

def find_captions_on_page(page) -> List[Caption]:
    """Find figure caption headers on a page."""
    out = []
    blocks = page.get_text("blocks")
    for tpl in blocks:
        if len(tpl) < 5:
            continue
        x0, y0, x1, y1, text = tpl[0], tpl[1], tpl[2], tpl[3], tpl[4]
        if not isinstance(text, str):
            continue
        m = _CAPTION_RE.match(text.lstrip())
        if not m:
            continue
        out.append(Caption(
            page=page.number + 1,
            number=m.group(1),
            text=text.strip(),
            bbox=(float(x0), float(y0), float(x1), float(y1)),
        ))
    return out


def find_image_regions_on_page(page) -> List[Region]:
    """Find raster image bboxes on a page."""
    out = []
    try:
        images = page.get_images(full=True)
    except Exception:
        images = []
    seen = set()
    for img in images:
        try:
            xref = img[0]
        except Exception:
            continue
        if xref in seen:
            continue
        seen.add(xref)
        try:
            for r in page.get_image_rects(xref):
                out.append(Region(
                    page=page.number + 1,
                    bbox=(r.x0, r.y0, r.x1, r.y1),
                    is_image=True,
                    image_id=str(xref),
                ))
        except Exception:
            pass
    return out


def find_negative_space_regions(page,
                                 min_area_frac: float = 0.05) -> List[Region]:
    """Find rectangles of empty page space that could host a vector
    figure (no text, no images). We grid-scan the page at 10x10 and
    coalesce empty cells into bounding boxes."""
    rect = page.rect
    W, H = rect.width, rect.height
    nx, ny = 20, 28
    cell_w = W / nx; cell_h = H / ny
    blocks = page.get_text("blocks")
    grid = [[True] * nx for _ in range(ny)]
    for tpl in blocks:
        if len(tpl) < 5:
            continue
        x0, y0, x1, y1, text = tpl[0], tpl[1], tpl[2], tpl[3], tpl[4]
        if not isinstance(text, str) or not text.strip():
            continue
        for j in range(int(y0 / cell_h), min(ny, int(y1 / cell_h) + 1)):
            for i in range(int(x0 / cell_w), min(nx, int(x1 / cell_w) + 1)):
                if 0 <= i < nx and 0 <= j < ny:
                    grid[j][i] = False
    # Also mark image rectangles as occupied (so we don't double-count)
    for img_r in find_image_regions_on_page(page):
        x0, y0, x1, y1 = img_r.bbox
        for j in range(int(y0 / cell_h), min(ny, int(y1 / cell_h) + 1)):
            for i in range(int(x0 / cell_w), min(nx, int(x1 / cell_w) + 1)):
                if 0 <= i < nx and 0 <= j < ny:
                    grid[j][i] = False
    # Find maximal rectangles of True cells (greedy)
    out = []
    visited = [[False] * nx for _ in range(ny)]
    for j in range(ny):
        for i in range(nx):
            if visited[j][i] or not grid[j][i]:
                continue
            # Greedy expand right and down
            ii = i
            while ii < nx and grid[j][ii] and not visited[j][ii]:
                ii += 1
            jj = j + 1
            while jj < ny:
                ok = all(grid[jj][k] and not visited[jj][k]
                          for k in range(i, ii))
                if not ok: break
                jj += 1
            for jj2 in range(j, jj):
                for ii2 in range(i, ii):
                    visited[jj2][ii2] = True
            area = (ii - i) * (jj - j) * cell_w * cell_h
            if area < min_area_frac * W * H:
                continue
            out.append(Region(
                page=page.number + 1,
                bbox=(i * cell_w, j * cell_h, ii * cell_w, jj * cell_h),
                is_image=False,
            ))
    return out


# ----------------------------------------------------------------------
# Pairing logic
# ----------------------------------------------------------------------

def _bbox_y_centre(b): return (b[1] + b[3]) / 2
def _bbox_x_centre(b): return (b[0] + b[2]) / 2
def _bbox_w(b): return b[2] - b[0]


def pair_captions(captions: List[Caption],
                  image_regions: List[Region],
                  empty_regions: List[Region]) -> List[Pairing]:
    """Pair each caption with the most likely figure region.

    Algorithm:
      1. Try to find a region (image or empty) whose bbox is ABOVE the
         caption (region.y1 <= caption.y0 + small slack) AND horizontally
         overlapping the caption's column. Pick the one with the
         smallest vertical gap.
      2. Fallback: nearest image by Euclidean distance.
      3. If no images at all on the page, return unpaired.
    """
    out = []
    all_regions = list(image_regions) + list(empty_regions)
    for cap in captions:
        cx0, cy0, cx1, cy1 = cap.bbox
        cap_xc = _bbox_x_centre(cap.bbox)
        cap_w = _bbox_w(cap.bbox)
        # Step 1
        candidates = []
        for r in all_regions:
            rx0, ry0, rx1, ry1 = r.bbox
            if ry1 > cy0 + 20:  # not above
                continue
            # Horizontal overlap
            overlap = min(cx1, rx1) - max(cx0, rx0)
            if overlap < 0.3 * cap_w:
                continue
            gap = cy0 - ry1
            candidates.append((gap, r))
        if candidates:
            candidates.sort(key=lambda kr: kr[0])
            _, best = candidates[0]
            out.append(Pairing(
                page=cap.page, caption=cap, region=best,
                distance=candidates[0][0],
                method=("nearest-above-image" if best.is_image else
                        "nearest-above-empty"),
            ))
            continue
        # Step 2 fallback
        if image_regions:
            def _d(r):
                rxc = _bbox_x_centre(r.bbox)
                ryc = _bbox_y_centre(r.bbox)
                return ((cap_xc - rxc) ** 2 + (cy0 - ryc) ** 2) ** 0.5
            best_img = min(image_regions, key=_d)
            out.append(Pairing(
                page=cap.page, caption=cap, region=best_img,
                distance=_d(best_img),
                method="nearest-image",
            ))
        else:
            out.append(Pairing(
                page=cap.page, caption=cap, region=None,
                distance=-1.0, method="unpaired",
            ))
    return out


def pair_pdf(pdf_path: Path) -> List[Pairing]:
    try:
        import fitz
    except ImportError:
        return []
    doc = fitz.open(str(pdf_path))
    try:
        pairings = []
        for p in range(doc.page_count):
            page = doc[p]
            caps = find_captions_on_page(page)
            if not caps:
                continue
            imgs = find_image_regions_on_page(page)
            empty = find_negative_space_regions(page)
            pairings.extend(pair_captions(caps, imgs, empty))
        return pairings
    finally:
        doc.close()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pdf", type=Path)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    pairs = pair_pdf(args.pdf)
    if args.json:
        out = [{
            "page": p.page,
            "caption_number": p.caption.number,
            "caption_text": p.caption.text[:100],
            "region_bbox": list(p.region.bbox) if p.region else None,
            "is_image": p.region.is_image if p.region else None,
            "distance": round(p.distance, 1),
            "method": p.method,
        } for p in pairs]
        print(json.dumps(out, indent=2))
    else:
        for p in pairs:
            tag = (f"image" if (p.region and p.region.is_image) else
                    ("empty" if p.region else "UNPAIRED"))
            print(f"p{p.page}: Fig {p.caption.number} -> "
                    f"{tag} dist={p.distance:.0f} method={p.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
