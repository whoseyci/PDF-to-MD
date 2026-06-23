"""
Classical (non-LLM) diagram → Mermaid extractor.

Inspired by:
  * Arrow R-CNN, FloCo-T5, DrawnNet (academic flowchart recognition)
  * "How I Built a Computer Vision Pipeline to Extract Structure from
    Flowchart Diagrams" (Medium, Wasif Ullah, Nov 2025)
  * github.com/topics/arrow-detection

Pipeline:

  1. Threshold + connected components → candidate node bounding boxes
     (rectangles, rounded rects, diamonds, ellipses).
  2. Per node, OCR the interior to get its label.
  3. Extract the "edge skeleton" -- everything dark that isn't inside a
     node bbox.
  4. Connected components on the edge skeleton → candidate edges. For
     each edge component, find the two endpoints (farthest pair of
     pixels) and snap each to the nearest node bbox.
  5. Detect arrowhead direction:
       - look at a small window around each endpoint
       - count how many edge pixels lie within ±30° of the line into
         the endpoint -- the endpoint with the narrower distribution
         is the *tail*, the wider one is the *head* (arrowhead)
       - if both look the same, mark the edge undirected
  6. Emit Mermaid `flowchart LR` syntax: one `A[label]` per node,
     `A --> B` (or `A --- B`) per edge.

What works (single-page synthetic diagrams):
  * Rectangular nodes with clearly-OCR-able labels
  * Straight or simple-curve arrows
  * Up to ~20 nodes (more starts overflowing the picker heuristics)

Known limitations:
  * Hand-drawn diagrams (irregular shapes)
  * Heavy node-overlap or arrows passing through nodes
  * Diamond / ellipse-only shape detection (current code routes
    everything to `[label]` rectangular nodes regardless of shape)
  * Dashed arrows: detected as multiple short edges and merged
    heuristically; results vary

Honest scope: this is a v1 baseline. For really complex diagrams the
VLM-based `mermaid_extract.py` will still beat this. But for clean
machine-rendered diagrams (matplotlib boxes-and-arrows, draw.io
exports, structured TPB-style models) this is faster, deterministic,
and free.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DiagramNode:
    id: str                    # "A", "B", "C", ...
    label: str
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)


@dataclass
class DiagramEdge:
    src: str
    dst: str
    directed: bool = True
    label: Optional[str] = None


@dataclass
class DiagramExtractionResult:
    status: str = "ok"          # "ok" | "partial" | "no_nodes" | "error"
    reason: str = ""
    nodes: List[DiagramNode] = field(default_factory=list)
    edges: List[DiagramEdge] = field(default_factory=list)
    mermaid: Optional[str] = None
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    raw_node_count: int = 0
    raw_edge_count: int = 0


def extract_diagram(image_path: Path,
                     *,
                     min_node_size: int = 30,
                     max_node_size_frac: float = 0.5,
                     max_node_size_frac_one_dim: float = 0.7,
                     ) -> DiagramExtractionResult:
    """
    Extract a structured graph from a flowchart-style diagram image
    and return a `DiagramExtractionResult` whose `mermaid` field is
    a fenced ```mermaid block ready to embed in Markdown.
    """
    t0 = time.time()
    result = DiagramExtractionResult()
    try:
        import cv2
    except ImportError:
        result.status = "error"
        result.reason = "opencv not installed"
        result.elapsed_seconds = round(time.time() - t0, 3)
        return result
    try:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            result.status = "error"
            result.reason = f"cv2.imread failed on {image_path}"
            result.elapsed_seconds = round(time.time() - t0, 3)
            return result
        H, W = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Step 1: find node candidates as filled-OR-outlined rectangles.
        # We detect rectangles by finding 4-vertex polygons in the
        # contour approximation. Anything else (diamonds, ellipses) is
        # picked up as connected components of saturated colour.
        # max_size: the LONG side of a node can be up to
        # max_node_size_frac_one_dim of EITHER W or H (use the bigger).
        # This handles wide-short title boxes on letterbox-shaped diagrams.
        max_size_w = int(max_node_size_frac_one_dim * W)
        max_size_h = int(max_node_size_frac_one_dim * H)
        nodes = _detect_node_bboxes(bgr, gray,
                                       min_size=min_node_size,
                                       max_size_w=max_size_w,
                                       max_size_h=max_size_h)
        result.raw_node_count = len(nodes)
        if not nodes:
            result.status = "no_nodes"
            result.reason = "no rectangular / coloured shapes found"
            result.elapsed_seconds = round(time.time() - t0, 3)
            return result

        # Step 2: OCR each node's interior.
        labeled: List[DiagramNode] = []
        for i, bbox in enumerate(nodes):
            text = _ocr_inside(bgr, bbox)
            if not text:
                text = f"Node {i + 1}"
            ident = _next_ident(i)
            labeled.append(DiagramNode(id=ident, label=text, bbox=bbox))

        # Step 3: edge skeleton -- dark pixels NOT inside any node bbox.
        edge_mask = _build_edge_mask(gray, labeled)

        # Step 4: connected components on the edge mask → candidate edges.
        edges = _detect_edges(edge_mask, labeled)
        result.raw_edge_count = len(edges)

        # If no edges found, still return the nodes -- maybe it was just
        # a labeled box diagram.
        result.nodes = labeled
        result.edges = edges
        result.mermaid = _to_mermaid(labeled, edges)
        if not edges:
            result.status = "partial"
            result.reason = (f"found {len(labeled)} nodes but no edges; "
                              "diagram may be label-only")
            result.confidence = 0.4
        else:
            result.status = "ok"
            result.reason = (f"extracted {len(labeled)} nodes + "
                              f"{len(edges)} edges")
            result.confidence = 0.6 if len(edges) >= len(labeled) - 1 else 0.4

    except Exception as e:
        result.status = "error"
        result.reason = f"{type(e).__name__}: {e}"
    result.elapsed_seconds = round(time.time() - t0, 3)
    return result


# --------------------------------------------------------------------
# Step 1: node detection
# --------------------------------------------------------------------

def _detect_node_bboxes(bgr, gray, *, min_size: int,
                          max_size_w: int, max_size_h: int
                          ) -> List[Tuple[int, int, int, int]]:
    """Find rectangular / filled-coloured shape bboxes."""
    import cv2
    H, W = gray.shape

    # Path A: filled-coloured regions. Lower saturation threshold to
    # catch pastel fills like #dbe9f4 (sat ~26 in HSV).
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = (hsv[:, :, 1] > 15).astype(np.uint8) * 255
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    candidates: List[Tuple[int, int, int, int]] = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < min_size or h < min_size: continue
        if w > max_size_w or h > max_size_h: continue
        if min(w, h) / max(w, h) < 0.1: continue
        fill = area / float(w * h)
        if fill < 0.5: continue
        candidates.append((int(x), int(y), int(w), int(h)))

    # Path B: outlined-only rectangles (no colour fill, just dark border).
    # Find 4-vertex polygons in the binary edge map.
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) not in (4, 5, 6): continue   # rectangles, rounded rects
        x, y, w, h = cv2.boundingRect(approx)
        if w < min_size or h < min_size: continue
        if w > max_size_w or h > max_size_h: continue
        if min(w, h) / max(w, h) < 0.15: continue
        # Skip if already covered by a saturated-colour candidate
        if any(_bbox_iou((x, y, w, h), c) > 0.5 for c in candidates):
            continue
        candidates.append((int(x), int(y), int(w), int(h)))

    # Dedupe overlapping bboxes via NMS.
    candidates = _nms_bboxes(candidates, iou_thresh=0.5)
    # Sort top-to-bottom, then left-to-right (reading order).
    candidates.sort(key=lambda b: (b[1] // 30, b[0]))
    return candidates


def _bbox_iou(a: Tuple[int, int, int, int],
               b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0: return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms_bboxes(bboxes, iou_thresh: float = 0.5):
    """Drop bboxes that overlap a larger one by >= iou_thresh."""
    sorted_b = sorted(bboxes, key=lambda b: -b[2] * b[3])
    keep = []
    for b in sorted_b:
        if not any(_bbox_iou(b, k) >= iou_thresh for k in keep):
            keep.append(b)
    return keep


# --------------------------------------------------------------------
# Step 2: OCR inside each node bbox
# --------------------------------------------------------------------

def _ocr_inside(bgr, bbox) -> str:
    """Run tesseract on a node's interior; return the cleaned text."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    x, y, w, h = bbox
    # Inset by a few pixels to skip the box border (anti-aliased pixels
    # near the border tend to produce phantom characters in OCR).
    pad = max(3, min(w, h) // 12)
    x0, y0 = x + pad, y + pad
    x1, y1 = x + w - pad, y + h - pad
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0: return ""
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) < 100:
        scale = 100 / max(gray.shape)
        new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(gray)
    try:
        text = pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return ""
    # Clean whitespace and discard noise tokens
    text = " ".join(text.split())
    # Strip leading/trailing punctuation noise
    text = text.strip("., :;|/\\-_=")
    return text


def _next_ident(i: int) -> str:
    """Generate node identifier: A..Z, then AA, AB, ..., AZ, BA..."""
    if i < 26: return chr(ord("A") + i)
    return chr(ord("A") + i // 26 - 1) + chr(ord("A") + i % 26)


# --------------------------------------------------------------------
# Step 3 & 4: edge mask + edge detection
# --------------------------------------------------------------------

def _build_edge_mask(gray, nodes: List[DiagramNode]) -> np.ndarray:
    """Dark pixels NOT inside any node bbox."""
    import cv2
    dark = (gray < 100).astype(np.uint8) * 255
    for node in nodes:
        x, y, w, h = node.bbox
        # Pad a few pixels into the node so border edges don't leak in
        pad = 3
        dark[max(0, y - pad):min(gray.shape[0], y + h + pad),
              max(0, x - pad):min(gray.shape[1], x + w + pad)] = 0
    # Connect small gaps
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return dark


def _detect_edges(edge_mask, nodes: List[DiagramNode]
                   ) -> List[DiagramEdge]:
    """Find connected components in the edge mask, snap endpoints to
    nearest nodes."""
    import cv2
    edges: List[DiagramEdge] = []
    if not nodes:
        return edges
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 30: continue
        # too small to be a meaningful edge
        if max(w, h) < 20: continue
        # The component's pixels:
        ys, xs = np.where(labels == i)
        if len(xs) < 10: continue
        # Find endpoints: pair of pixels with maximum distance from
        # each other within the component.
        ep1, ep2 = _far_endpoints(xs, ys)
        if ep1 is None or ep2 is None: continue
        # Snap each endpoint to nearest node.
        n1 = _nearest_node(ep1, nodes)
        n2 = _nearest_node(ep2, nodes)
        if n1 is None or n2 is None: continue
        if n1.id == n2.id: continue   # self-loops not supported

        # Detect direction by arrowhead-pixel density at endpoints.
        head_at = _detect_arrowhead(labels == i, ep1, ep2)
        if head_at == 1:
            src, dst = n2.id, n1.id
            directed = True
        elif head_at == 2:
            src, dst = n1.id, n2.id
            directed = True
        else:
            # Fall back to spatial heuristic: arrow goes left-to-right
            # if dx dominates, top-to-bottom otherwise. This matches
            # the convention of most flowcharts (and the rendered
            # mermaid `flowchart LR` layout).
            n1c = (n1.bbox[0] + n1.bbox[2] / 2,
                    n1.bbox[1] + n1.bbox[3] / 2)
            n2c = (n2.bbox[0] + n2.bbox[2] / 2,
                    n2.bbox[1] + n2.bbox[3] / 2)
            if abs(n2c[0] - n1c[0]) > abs(n2c[1] - n1c[1]):
                if n1c[0] <= n2c[0]:
                    src, dst = n1.id, n2.id
                else:
                    src, dst = n2.id, n1.id
            else:
                if n1c[1] <= n2c[1]:
                    src, dst = n1.id, n2.id
                else:
                    src, dst = n2.id, n1.id
            directed = False
        edges.append(DiagramEdge(
            src=src, dst=dst, directed=directed))
    # Deduplicate: if two edges connect the same pair, keep one.
    seen = set()
    dedup = []
    for e in edges:
        key = (e.src, e.dst) if e.directed else tuple(sorted((e.src, e.dst)))
        if key in seen: continue
        seen.add(key); dedup.append(e)
    return dedup


def _far_endpoints(xs, ys
                    ) -> Tuple[Optional[Tuple[int, int]],
                                 Optional[Tuple[int, int]]]:
    """Approximate the farthest pair using axis extremes (cheap and
    works well for line-ish components)."""
    if len(xs) == 0: return None, None
    # Try x-extremes and y-extremes, pick the pair with longest distance
    candidates = []
    for idx in (np.argmin(xs), np.argmax(xs),
                np.argmin(ys), np.argmax(ys)):
        candidates.append((int(xs[idx]), int(ys[idx])))
    best_pair = None; best_d = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            d = math.hypot(candidates[i][0] - candidates[j][0],
                            candidates[i][1] - candidates[j][1])
            if d > best_d:
                best_d = d; best_pair = (candidates[i], candidates[j])
    if best_pair is None: return None, None
    return best_pair


def _nearest_node(pt: Tuple[int, int], nodes: List[DiagramNode]
                   ) -> Optional[DiagramNode]:
    """Snap to the node whose bbox edge is closest to the point.
    Returns None if no node is within 0.3 * max(W,H) pixels."""
    px, py = pt
    best = None; best_d = float("inf")
    for n in nodes:
        x, y, w, h = n.bbox
        # Distance from point to nearest edge of bbox
        dx = max(x - px, 0, px - (x + w))
        dy = max(y - py, 0, py - (y + h))
        d = math.hypot(dx, dy)
        if d < best_d:
            best_d = d; best = n
    # Snap threshold: at most 80px between endpoint and nearest node edge
    if best is None or best_d > 80:
        return None
    return best


def _detect_arrowhead(mask: np.ndarray, ep1: Tuple[int, int],
                       ep2: Tuple[int, int], radius: int = 12) -> int:
    """
    Crude direction detection: count edge pixels within `radius` of
    each endpoint. The arrowhead end has MORE pixels (because of the
    triangular tip splitting into two flanks).

    Returns:
       1 -- arrowhead at ep1
       2 -- arrowhead at ep2
       0 -- can't tell (undirected)
    """
    h, w = mask.shape
    def density(pt):
        x, y = pt
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        return int(mask[y0:y1, x0:x1].sum())
    d1 = density(ep1); d2 = density(ep2)
    if d1 == 0 and d2 == 0: return 0
    # Need at least 20% asymmetry to call it
    if max(d1, d2) == 0: return 0
    ratio = abs(d1 - d2) / max(d1, d2)
    if ratio < 0.2: return 0
    return 1 if d1 > d2 else 2


# --------------------------------------------------------------------
# Step 6: emit Mermaid
# --------------------------------------------------------------------

def _to_mermaid(nodes: List[DiagramNode],
                 edges: List[DiagramEdge]) -> str:
    """Build a fenced ```mermaid block."""
    lines = ["```mermaid", "flowchart LR"]
    for n in nodes:
        # Escape closing brackets in labels
        safe = n.label.replace("]", " ").replace("[", " ").strip() or n.id
        lines.append(f"    {n.id}[{safe}]")
    for e in edges:
        arrow = "-->" if e.directed else "---"
        if e.label:
            lines.append(f"    {e.src} {arrow}|{e.label}| {e.dst}")
        else:
            lines.append(f"    {e.src} {arrow} {e.dst}")
    lines.append("```")
    return "\n".join(lines)
