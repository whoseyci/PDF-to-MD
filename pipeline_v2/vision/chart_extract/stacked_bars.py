"""E8 -- Stacked-bar geometric extractor.

Approach:
  1. Find the value axis (reuse axis_ocr).
  2. Find candidate bars as columns of contiguous coloured segments.
  3. For each bar column, segment by colour bands → per-series values.
  4. Map pixel heights → values via the axis calibration.
"""
from __future__ import annotations
import time
from collections import Counter
from typing import List, Tuple

import numpy as np

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus
from .axis_ocr import ocr_words, find_axes, find_label_text, _parse_num


class StackedBarsExtractor(ChartExtractor):
    name = "stacked_bars/v1"
    why = ""

    def extract(self, image_path, *, caption=None, ocr_text=None):
        t0 = time.time()
        r = ChartExtractionResult(extractor=self.name,
                                    status=ExtractionStatus.ERROR)
        try:
            import cv2
        except ImportError:
            r.reason = "opencv missing"
            r.elapsed_seconds = time.time() - t0; return r
        try:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                r.reason = "imread failed"
                r.elapsed_seconds = time.time() - t0; return r
            H, W = bgr.shape[:2]
            words = ocr_words(image_path)
            if not words:
                r.status = ExtractionStatus.OCR_FAILED
                r.elapsed_seconds = time.time() - t0; return r
            _, y_axis = find_axes(words, (W, H))
            if not y_axis:
                r.status = ExtractionStatus.NO_AXIS
                r.reason = "no y-axis"
                r.elapsed_seconds = time.time() - t0; return r
            axis = y_axis
            r.value_axis = axis.axis
            r.orientation = "vertical"
            r.calibration = {"slope": axis.p_to_v[0],
                              "intercept": axis.p_to_v[1],
                              "r2": axis.confidence}

            tick_pix = [t[0] for t in axis.ticks]
            x0 = axis.perp_band_pixel + 2
            x1 = W
            y0 = min(tick_pix); y1 = max(tick_pix)
            if x1 - x0 < 20 or y1 - y0 < 20:
                r.status = ExtractionStatus.NO_BARS
                r.elapsed_seconds = time.time() - t0; return r
            plot = bgr[y0:y1, x0:x1].copy()
            r.plot_area = [int(x0), int(y0), int(x1), int(y1)]
            ph, pw = plot.shape[:2]

            hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
            saturated = (hsv[:, :, 1] > 60).astype(np.uint8)
            col_count = saturated.sum(axis=0)
            col_active = col_count > max(8, 0.20 * ph)
            runs: List[Tuple[int, int]] = []
            i = 0
            while i < pw:
                if col_active[i]:
                    j = i
                    while j < pw and col_active[j]:
                        j += 1
                    if (j - i) >= 8:
                        runs.append((i, j))
                    i = j
                else:
                    i += 1
            runs = [(a, b) for (a, b) in runs if (b - a) < 0.40 * pw]
            if len(runs) < 2:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "no bar columns"
                r.elapsed_seconds = time.time() - t0; return r

            def px_to_val(p_in_plot):
                return axis.pixel_to_value(p_in_plot + y0)

            def colour_key(hsv_px):
                h, s, v = int(hsv_px[0]), int(hsv_px[1]), int(hsv_px[2])
                if s < 30 and v > 200: return None
                if s < 30 and v < 60: return None
                return (h // 18,)

            all_series_keys: Counter = Counter()
            per_bar_segments = []
            for (xa, xb) in runs:
                strip_hsv = hsv[:, xa:xb]
                row_keys = []
                for y in range(ph):
                    row_pixels = strip_hsv[y]
                    keys = [colour_key(px) for px in row_pixels]
                    keys = [k for k in keys if k is not None]
                    if not keys:
                        row_keys.append(None); continue
                    most = Counter(keys).most_common(1)[0][0]
                    row_keys.append(most)
                segs: List[Tuple[object, int, int]] = []
                i = 0
                while i < ph:
                    if row_keys[i] is None:
                        i += 1; continue
                    k = row_keys[i]
                    j = i
                    while j < ph and row_keys[j] == k:
                        j += 1
                    if (j - i) >= 3:
                        segs.append((k, i, j))
                    i = j
                segs = [(k, a, b) for (k, a, b) in segs if (b - a) >= 4]
                seg_map = {}
                for (k, a, b) in segs:
                    if k in seg_map:
                        oa, ob = seg_map[k]
                        seg_map[k] = (min(oa, a), max(ob, b))
                    else:
                        seg_map[k] = (a, b)
                    all_series_keys[k] += 1
                per_bar_segments.append(seg_map)

            top_keys = [k for k, _ in all_series_keys.most_common(8)]
            if not top_keys:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "no colour segments"
                r.elapsed_seconds = time.time() - t0; return r
            avg_y = {k: 0.0 for k in top_keys}
            count_y = {k: 0 for k in top_keys}
            for seg_map in per_bar_segments:
                for k, (a, b) in seg_map.items():
                    if k in avg_y:
                        avg_y[k] += (a + b) / 2; count_y[k] += 1
            sorted_keys = sorted(top_keys,
                                  key=lambda k: avg_y[k] / max(1, count_y[k]))
            series_labels = [f"Series {i + 1}" for i in range(len(sorted_keys))]

            matrix = []
            for seg_map in per_bar_segments:
                row = []
                for k in sorted_keys:
                    if k in seg_map:
                        a, b = seg_map[k]
                        v_top = px_to_val(a)
                        v_bot = px_to_val(b)
                        row.append(round(abs(v_top - v_bot), 3))
                    else:
                        row.append(0.0)
                matrix.append(row)

            centroids = [(a + b) / 2 + x0 for (a, b) in runs]
            cat_words = [w for w in words
                          if w.cy > y1 - 2
                          and w.cy < y1 + 0.18 * H
                          and _parse_num(w.text) is None]
            cat_labels = []
            for c in centroids:
                near = sorted(cat_words, key=lambda w: abs(w.cx - c))
                if near and abs(near[0].cx - c) < (x1 - x0) / max(1, len(runs)):
                    cat_labels.append(near[0].text)
                else:
                    cat_labels.append(f"Cat {len(cat_labels) + 1}")

            r.categories = cat_labels
            r.series = series_labels
            r.matrix = matrix
            r.values = [round(sum(row), 3) for row in matrix]
            r.value_label = find_label_text(words, near=axis.axis,
                                              axis_cal=axis, image_size=(W, H),
                                              skip_first_band=False)
            n_known = sum(1 for c in cat_labels if not c.startswith("Cat "))
            if n_known >= max(1, int(0.6 * len(cat_labels))) and len(matrix) >= 2:
                r.status = ExtractionStatus.OK
                r.confidence = min(0.90, 0.4 + 0.05 * len(sorted_keys)
                                    + 0.5 * axis.confidence)
                r.reason = "extracted"
            else:
                r.status = ExtractionStatus.PARTIAL
                r.confidence = 0.4
                r.reason = "partial labels"
        except Exception as e:
            r.status = ExtractionStatus.ERROR
            r.reason = f"{type(e).__name__}: {e}"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r
