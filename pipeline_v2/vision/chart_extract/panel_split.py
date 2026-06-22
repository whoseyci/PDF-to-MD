"""Multi-panel figure splitter."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .axis_ocr import ocr_words, Word


_PANEL_LABEL_RE = re.compile(r"^[\(\[]?([A-Za-z])[\)\]\.\,\:]?$")


@dataclass
class Panel:
    x0: int; y0: int; x1: int; y1: int
    label: Optional[str] = None
    row: int = 0; col: int = 0
    def as_tuple(self): return (self.x0, self.y0, self.x1, self.y1)


def detect_panels(image_path, *, image_size=None, words=None,
                    min_panel_dim_frac=0.18):
    try:
        if image_size is None:
            from PIL import Image
            with Image.open(image_path) as im:
                W, H = im.size
        else:
            W, H = image_size
    except Exception:
        return [Panel(0, 0, 0, 0)]
    if words is None:
        try: words = ocr_words(image_path)
        except Exception: words = []
    cand = _find_panel_label_candidates(words, (W, H))
    panels = _cluster_panel_labels_to_grid(cand, (W, H),
                                              min_dim_frac=min_panel_dim_frac)
    if panels and len(panels) > 1:
        return panels
    panels = _split_by_whitespace_bands(image_path, (W, H),
                                          min_dim_frac=min_panel_dim_frac)
    if panels and len(panels) > 1:
        return panels
    return [Panel(0, 0, W, H)]


def _find_panel_label_candidates(words, image_size):
    W, H = image_size
    candidates = []
    for w in words:
        s = w.text.strip()
        if not _PANEL_LABEL_RE.match(s): continue
        if w.h < 0.018 * H or w.h > 0.10 * H: continue
        candidates.append(w)
    if not candidates: return []
    max_h = max(w.h for w in candidates)
    consistent = [w for w in candidates if w.h >= 0.85 * max_h]
    if len(consistent) >= 2:
        return consistent
    return candidates


def _cluster_panel_labels_to_grid(labels, image_size, min_dim_frac):
    if not labels: return []
    W, H = image_size

    def cluster_1d(vals, tol):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        groups = []
        for i in order:
            if not groups: groups.append([i]); continue
            mean = sum(vals[j] for j in groups[-1]) / len(groups[-1])
            if abs(vals[i] - mean) <= tol: groups[-1].append(i)
            else: groups.append([i])
        return groups

    cys = [w.cy for w in labels]; cxs = [w.cx for w in labels]
    row_groups = cluster_1d(cys, tol=0.04 * H)
    col_groups = cluster_1d(cxs, tol=0.04 * W)
    n_rows = len(row_groups); n_cols = len(col_groups)
    if n_rows * n_cols < 2: return []

    row_centers = sorted([(min(cys[i] for i in g), g) for g in row_groups])
    col_centers = sorted([(min(cxs[i] for i in g), g) for g in col_groups])
    row_ys = [c for c, _ in row_centers]
    col_xs = [c for c, _ in col_centers]

    def bounds_from_label_pos(arr, end, edge_offset):
        return [0] + [max(0, a - edge_offset) for a in arr[1:]] + [end]

    row_bounds = bounds_from_label_pos(row_ys, H, int(0.02 * H))
    col_bounds = bounds_from_label_pos(col_xs, W, int(0.02 * W))

    label_rc = {}
    for r, (_c, g) in enumerate(row_centers):
        for i in g:
            label_rc[i] = label_rc.get(i, (r, None))
            label_rc[i] = (r, label_rc[i][1])
    for c, (_x, g) in enumerate(col_centers):
        for i in g:
            r0, _ = label_rc.get(i, (None, None))
            label_rc[i] = (r0, c)

    placed_with_label = {}
    for i, (r, c) in label_rc.items():
        if r is None or c is None: continue
        if (r, c) in placed_with_label: continue
        placed_with_label[(r, c)] = labels[i].text.strip("()[].,: ")

    fillable = (n_rows * n_cols >= 4 and
                len(placed_with_label) >= 0.5 * n_rows * n_cols)
    panels = []
    for r in range(n_rows):
        for c in range(n_cols):
            label = placed_with_label.get((r, c))
            if label is None and not fillable: continue
            x0 = int(col_bounds[c]); x1 = int(col_bounds[c + 1])
            y0 = int(row_bounds[r]); y1 = int(row_bounds[r + 1])
            if x1 - x0 < min_dim_frac * W or y1 - y0 < min_dim_frac * H:
                return []
            panels.append(Panel(x0=x0, y0=y0, x1=x1, y1=y1,
                                  label=label, row=r, col=c))
    if len(panels) < 2: return []
    panels.sort(key=lambda p: (p.row, p.col))
    return panels


def _split_by_whitespace_bands(image_path, image_size, *, min_dim_frac):
    try: import cv2
    except ImportError: return []
    try:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None: return []
    except Exception:
        return []
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    is_bg = ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 230)).astype(np.uint8)
    h_seps = _find_separator_bands(is_bg.mean(axis=1),
                                     min_band=int(0.012 * H),
                                     min_gap=int(min_dim_frac * H),
                                     thresh=0.98)
    v_seps = _find_separator_bands(is_bg.mean(axis=0),
                                     min_band=int(0.012 * W),
                                     min_gap=int(min_dim_frac * W),
                                     thresh=0.98)
    if not h_seps and not v_seps: return []
    def sep_to_bounds(seps, end):
        bs = [0]
        for (a, b) in seps:
            bs.append((a + b) // 2)
        bs.append(end); return bs
    rb = sep_to_bounds(h_seps, H); cb = sep_to_bounds(v_seps, W)
    min_w = min_dim_frac * W; min_h = min_dim_frac * H
    while len(rb) > 2 and rb[1] - rb[0] < min_h: rb = rb[1:]
    while len(rb) > 2 and rb[-1] - rb[-2] < min_h: rb = rb[:-1]
    while len(cb) > 2 and cb[1] - cb[0] < min_w: cb = cb[1:]
    while len(cb) > 2 and cb[-1] - cb[-2] < min_w: cb = cb[:-1]
    def _collapse_thin_interior(bounds, min_dim):
        changed = True
        while changed and len(bounds) > 3:
            changed = False
            for k in range(1, len(bounds) - 1):
                left_w = bounds[k] - bounds[k - 1]
                right_w = bounds[k + 1] - bounds[k]
                if right_w < min_dim and k + 1 < len(bounds):
                    bounds = bounds[:k] + bounds[k + 1:]
                    changed = True; break
                if left_w < min_dim and k > 0:
                    bounds = bounds[:k] + bounds[k + 1:]
                    changed = True; break
        return bounds
    rb = _collapse_thin_interior(rb, min_h)
    cb = _collapse_thin_interior(cb, min_w)
    if len(rb) > 2:
        row_heights = [rb[i + 1] - rb[i] for i in range(len(rb) - 1)]
        if max(row_heights) > 0 and min(row_heights) / max(row_heights) < 0.7:
            return []
    if len(cb) > 2:
        col_widths = [cb[i + 1] - cb[i] for i in range(len(cb) - 1)]
        if max(col_widths) > 0 and min(col_widths) / max(col_widths) < 0.7:
            return []
    panels = []
    for r in range(len(rb) - 1):
        for c in range(len(cb) - 1):
            x0, x1 = cb[c], cb[c + 1]
            y0, y1 = rb[r], rb[r + 1]
            if x1 - x0 < min_w or y1 - y0 < min_h:
                return []
            panels.append(Panel(x0=x0, y0=y0, x1=x1, y1=y1, row=r, col=c))
    return panels if len(panels) >= 2 else []


def _find_separator_bands(profile, *, min_band, min_gap, thresh):
    n = len(profile)
    is_sep = profile >= thresh
    runs = []; start = None
    for i in range(n):
        if is_sep[i] and start is None: start = i
        elif not is_sep[i] and start is not None:
            if i - start >= min_band: runs.append((start, i - 1))
            start = None
    if start is not None and n - start >= min_band:
        runs.append((start, n - 1))
    if not runs: return []
    merge_thresh = max(min_band, min_gap // 3)
    merged = [runs[0]]
    for (a, b) in runs[1:]:
        pa, pb = merged[-1]
        if a - pb - 1 < merge_thresh:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    content_thresh = max(min_band, min_gap // 4)
    keep = []
    for j, (a, b) in enumerate(merged):
        if j == 0:
            content_above = int(np.sum(~is_sep[:a]))
        else:
            content_above = int(np.sum(~is_sep[merged[j - 1][1] + 1:a]))
        if content_above < content_thresh: continue
        if j == len(merged) - 1:
            content_below = int(np.sum(~is_sep[b + 1:]))
        else:
            content_below = int(np.sum(~is_sep[b + 1:merged[j + 1][0]]))
        if content_below < content_thresh: continue
        keep.append((a, b))
    return keep


def crop_panel(image_path, panel, out_path):
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            im.crop((panel.x0, panel.y0, panel.x1, panel.y1)).save(out_path)
        return True
    except Exception:
        return False
