"""Shared legend-mapping helper."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from .axis_ocr import _parse_num, Word
from .colour_utils import colour_distance, quantise_colour


def build_legend_map(full_rgb, words, palette, *, exclude_box=None,
                      exclude_circle=None):
    import cv2
    H, W = full_rgb.shape[:2]
    hsv = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2HSV)
    sat = (hsv[:, :, 1] > 80).astype(np.uint8) * 255
    if exclude_box is not None:
        x0, y0, x1, y1 = exclude_box
        sat[y0:y1 + 1, x0:x1 + 1] = 0
    if exclude_circle is not None:
        cx, cy, radius = exclude_circle
        pad = int(radius * 1.05)
        sat[max(0, cy - pad):min(H, cy + pad),
              max(0, cx - pad):min(W, cx + pad)] = 0

    n, _, stats, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    swatches = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 or area > 4000: continue
        if max(w, h) > 80: continue
        ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if ratio < 0.3: continue
        region = full_rgb[y:y + h, x:x + w]
        flat = region.reshape(-1, 3)
        flat = flat[~np.all(flat > 220, axis=1)]
        if len(flat) == 0: continue
        modal = tuple(int(c) for c in quantise_colour(
            tuple(int(c) for c in np.median(flat, axis=0))))
        swatches.append((modal, (int(x + w // 2), int(y + h // 2))))
    if not swatches: return {}

    swatch_labels = []
    for color, (sx, sy) in swatches:
        best = None; best_d = 1e18
        for w in words:
            if _parse_num(w.text) is not None: continue
            if len(w.text) < 2: continue
            dx = w.cx - sx; dy = abs(w.cy - sy)
            if dx < 0 or dx > 200 or dy > 25: continue
            d = dx + dy * 2
            if d < best_d: best_d = d; best = w
        if best is not None:
            swatch_labels.append((color, best.text))

    out = {}
    for pc in palette:
        best = None; best_d = 1e18
        for (sc, label) in swatch_labels:
            d = colour_distance(pc, sc)
            if d < best_d: best_d = d; best = label
        if best is not None and best_d < 80:
            out[pc] = best
    return out
