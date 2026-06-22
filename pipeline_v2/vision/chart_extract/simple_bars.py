"""SimpleBarsExtractor -- single-panel bar chart with one numeric axis."""
from __future__ import annotations
import time
from typing import List, Tuple
import numpy as np

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus
from .axis_ocr import ocr_words, find_axes, find_label_text, _parse_num, Word, AxisCalibration


class SimpleBarsExtractor(ChartExtractor):
    name = "simple_bars/v1"

    def extract(self, image_path, *, caption=None, ocr_text=None):
        t0 = time.time()
        r = ChartExtractionResult(extractor=self.name, status=ExtractionStatus.ERROR)
        try: import cv2
        except ImportError:
            r.reason = "opencv missing"; r.elapsed_seconds = time.time() - t0; return r
        try:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                r.reason = "cv2.imread failed"; r.elapsed_seconds = time.time() - t0; return r
            H, W = bgr.shape[:2]
            words = ocr_words(image_path)
            if not words:
                r.status = ExtractionStatus.OCR_FAILED; r.reason = "no words"
                r.elapsed_seconds = time.time() - t0; return r
            x_axis, y_axis = find_axes(words, (W, H))
            def _sc(a): return (len(a.ticks), a.confidence) if a else (-1, -1)
            if y_axis and (not x_axis or _sc(y_axis) >= _sc(x_axis)):
                value_axis, orientation = y_axis, "vertical"
                cap = max(p for p, _ in y_axis.ticks)
            elif x_axis:
                value_axis, orientation = x_axis, "horizontal"
                cap = min(p for p, _ in x_axis.ticks)
            else:
                r.status = ExtractionStatus.NO_AXIS; r.reason = "no axis"
                r.elapsed_seconds = time.time() - t0; return r
            r.orientation = orientation; r.value_axis = value_axis.axis
            r.calibration = {"axis": value_axis.axis,
                              "slope": value_axis.p_to_v[0],
                              "intercept": value_axis.p_to_v[1],
                              "r2": value_axis.confidence,
                              "ticks": value_axis.ticks,
                              "axis_band_pixel": value_axis.perp_band_pixel}
            r.value_label = find_label_text(words, near=value_axis.axis,
                                              axis_cal=value_axis,
                                              image_size=(W, H), skip_first_band=False)
            cat_axis_name = "x" if value_axis.axis == "y" else "y"
            fake = AxisCalibration(cat_axis_name, (0, 0), [], 0.0, cap)
            r.category_label = find_label_text(words, near=cat_axis_name,
                                                 axis_cal=fake, image_size=(W, H),
                                                 skip_first_band=True)
            tick_pix = [t[0] for t in value_axis.ticks]
            if orientation == "vertical":
                x0, x1 = value_axis.perp_band_pixel + 2, W
                y0, y1 = min(tick_pix), max(tick_pix)
            else:
                y0, y1 = 0, value_axis.perp_band_pixel - 2
                x0, x1 = min(tick_pix), max(tick_pix)
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 20 or y1 - y0 < 20:
                r.status = ExtractionStatus.NO_BARS; r.reason = "plot degenerate"
                r.elapsed_seconds = time.time() - t0; return r
            r.plot_area = [int(x0), int(y0), int(x1), int(y1)]
            plot = bgr[y0:y1, x0:x1]
            bars = _find_bars(plot, orientation)
            if not bars:
                r.status = ExtractionStatus.NO_BARS; r.reason = "no bars"
                r.elapsed_seconds = time.time() - t0; return r
            bars_g = [(bx + x0, by + y0, bw, bh) for (bx, by, bw, bh) in bars]
            r.bar_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in bars_g]
            if orientation == "vertical":
                bars_g.sort(key=lambda b: b[0] + b[2] / 2)
            else:
                bars_g.sort(key=lambda b: b[1] + b[3] / 2)
            slope, intercept = value_axis.p_to_v
            if abs(slope) < 1e-12:
                r.status = ExtractionStatus.NO_AXIS; r.reason = "slope ~0"
                r.elapsed_seconds = time.time() - t0; return r
            zero_pixel = -intercept / slope
            values = []; centroids = []
            for (bx, by, bw, bh) in bars_g:
                if orientation == "vertical":
                    v = value_axis.pixel_to_value(by)
                    if abs((by + bh) - zero_pixel) > 0.05 * (y1 - y0):
                        v = value_axis.pixel_to_value(by) - value_axis.pixel_to_value(by + bh)
                    values.append(round(v, 3)); centroids.append(int(bx + bw / 2))
                else:
                    v = value_axis.pixel_to_value(bx + bw)
                    if abs(bx - zero_pixel) > 0.05 * (x1 - x0):
                        v = value_axis.pixel_to_value(bx + bw) - value_axis.pixel_to_value(bx)
                    values.append(round(v, 3)); centroids.append(int(by + bh / 2))
            cat_labels = _assign_categories(words, centroids,
                                              orientation=orientation,
                                              axis_pixel=cap, image_size=(W, H))
            n_known = sum(1 for c in cat_labels if not c.startswith("Category "))
            r.categories = cat_labels; r.values = values
            if n_known == len(values):
                r.status = ExtractionStatus.OK
                r.confidence = min(0.95, 0.5 + 0.5 * value_axis.confidence)
                r.reason = "extracted"
            else:
                r.status = ExtractionStatus.PARTIAL
                r.confidence = 0.4 + 0.3 * (n_known / max(1, len(values)))
                r.reason = f"{len(values) - n_known}/{len(values)} labels missing"
                r.warnings.append("some labels unreadable")
        except Exception as e:
            r.status = ExtractionStatus.ERROR; r.reason = f"{type(e).__name__}: {e}"
        r.elapsed_seconds = round(time.time() - t0, 3); return r


def _find_bars(plot_bgr, orientation):
    import cv2
    H, W = plot_bgr.shape[:2]
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]; v = hsv[:, :, 2]
    mask = ((s > 40) | ((v < 80) & (s < 40))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bars = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.001 * H * W: continue
        if w < 3 or h < 3: continue
        aspect = (h / w) if orientation == "vertical" else (w / h)
        if aspect < 0.3: continue
        if orientation == "vertical":
            if h < 0.05 * H or w > 0.5 * W: continue
        else:
            if w < 0.05 * W or h > 0.5 * H: continue
        fill = area / float(w * h)
        if fill < 0.55: continue
        bars.append((int(x), int(y), int(w), int(h)))
    if len(bars) == 1 and bars[0][2] * bars[0][3] > 0.4 * W * H:
        return []
    return bars


def _assign_categories(words, centroids, *, orientation, axis_pixel, image_size):
    W, H = image_size
    candidates = []
    for w in words:
        if _parse_num(w.text) is not None: continue
        if len(w.text) < 1: continue
        if orientation == "vertical":
            if w.cy <= axis_pixel - 5 or w.cy > axis_pixel + 0.4 * H: continue
        else:
            if w.cx >= axis_pixel + 5 or w.cx < axis_pixel - 0.4 * W: continue
        candidates.append(w)
    if not candidates:
        return [f"Category {i + 1}" for i in range(len(centroids))]
    if orientation == "vertical":
        nearest = min(w.cy - axis_pixel for w in candidates)
        band_tol = max(20.0, 0.04 * H)
        candidates = [w for w in candidates if (w.cy - axis_pixel) <= nearest + band_tol]
    else:
        nearest = min(axis_pixel - w.cx for w in candidates)
        band_tol = max(20.0, 0.04 * W)
        candidates = [w for w in candidates if (axis_pixel - w.cx) <= nearest + band_tol]
    out = []; used = set()
    spacing = ((max(centroids) - min(centroids)) / max(1, len(centroids) - 1)
                if len(centroids) > 1 else max(W, H))
    for i, c in enumerate(centroids):
        best = None; best_d = 1e18
        for w in candidates:
            wc = w.cx if orientation == "vertical" else w.cy
            d = abs(wc - c)
            if d < best_d and id(w) not in used:
                best_d = d; best = w
        if best is not None and best_d < spacing * 0.7:
            out.append(best.text); used.add(id(best))
        else:
            out.append(f"Category {i + 1}")
    return out
