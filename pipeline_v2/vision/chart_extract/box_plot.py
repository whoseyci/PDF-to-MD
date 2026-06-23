"""E8 -- Box-plot geometric extractor."""
from __future__ import annotations
import time

import numpy as np

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus
from .axis_ocr import ocr_words, find_axes, find_label_text, _parse_num


class BoxPlotExtractor(ChartExtractor):
    name = "box_plot/v1"

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
                r.elapsed_seconds = time.time() - t0; return r
            x0 = y_axis.perp_band_pixel + 2
            x1 = W
            tick_pix = [t[0] for t in y_axis.ticks]
            y0 = min(tick_pix); y1 = max(tick_pix)
            if x1 - x0 < 30 or y1 - y0 < 30:
                r.status = ExtractionStatus.NO_BARS
                r.elapsed_seconds = time.time() - t0; return r
            r.plot_area = [int(x0), int(y0), int(x1), int(y1)]
            plot = bgr[y0:y1, x0:x1]
            ph, pw = plot.shape[:2]
            gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            n_comp, _, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
            box_candidates = []
            for i in range(1, n_comp):
                x, y, w, h, area = stats[i]
                if w < 14 or h < max(10, 0.08 * ph): continue
                if w > 0.30 * pw: continue
                if h > 0.70 * ph: continue
                aspect = w / max(1, h)
                if aspect > 4 or aspect < 0.20: continue
                fill = area / max(1, w * h)
                if fill < 0.30: continue
                box_candidates.append((x, y, w, h, area))
            box_candidates.sort(key=lambda b: b[0])
            coalesced = []
            for c in box_candidates:
                if coalesced and (c[0] < coalesced[-1][0] + coalesced[-1][2]):
                    if c[2] * c[3] > coalesced[-1][2] * coalesced[-1][3]:
                        coalesced[-1] = c
                else:
                    coalesced.append(c)
            box_candidates = coalesced
            if not box_candidates:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "no box candidates"
                r.elapsed_seconds = time.time() - t0; return r

            def y_px_to_val(yp): return y_axis.pixel_to_value(yp + y0)

            box_stats = []
            centroids = []
            for (bx, by, bw, bh, _area) in box_candidates:
                q1 = float(y_px_to_val(by + bh))
                q3 = float(y_px_to_val(by))
                sub = gray[by:by + bh, bx:bx + bw]
                row_dark = (sub < 100).sum(axis=1)
                if len(row_dark):
                    median_row = int(np.argmax(row_dark))
                    median_val = float(y_px_to_val(by + median_row))
                else:
                    median_val = (q1 + q3) / 2
                col_center = bx + bw // 2
                col_slice = thr[:, col_center - 2:col_center + 3]
                above = col_slice[:by]
                below = col_slice[by + bh:]
                whisker_top_px = by
                rows_above = np.where(above.sum(axis=1) > 0)[0]
                if len(rows_above):
                    whisker_top_px = rows_above[0]
                whisker_bottom_px = by + bh
                rows_below = np.where(below.sum(axis=1) > 0)[0]
                if len(rows_below):
                    whisker_bottom_px = by + bh + rows_below[-1]
                w_high = float(y_px_to_val(whisker_top_px))
                w_low = float(y_px_to_val(whisker_bottom_px))
                box_stats.append({
                    "label": None,
                    "q1": round(q1, 3),
                    "q3": round(q3, 3),
                    "median": round(median_val, 3),
                    "whisker_low": round(min(w_high, w_low), 3),
                    "whisker_high": round(max(w_high, w_low), 3),
                })
                centroids.append(bx + bw // 2 + x0)

            cat_words = [w for w in words
                          if w.cy > y1 - 2
                          and w.cy < y1 + 0.18 * H
                          and _parse_num(w.text) is None]
            for i, c in enumerate(centroids):
                near = sorted(cat_words, key=lambda w: abs(w.cx - c))
                if near and abs(near[0].cx - c) < (x1 - x0) / max(1, len(centroids)):
                    box_stats[i]["label"] = near[0].text
                else:
                    box_stats[i]["label"] = f"Group {i + 1}"

            r.box_stats = box_stats
            r.categories = [b["label"] for b in box_stats]
            r.values = [b["median"] for b in box_stats]
            r.value_label = find_label_text(words, near=y_axis.axis,
                                              axis_cal=y_axis,
                                              image_size=(W, H),
                                              skip_first_band=False)
            r.status = ExtractionStatus.OK if len(box_stats) >= 2 \
                else ExtractionStatus.PARTIAL
            r.confidence = min(0.85, 0.4 + 0.05 * len(box_stats)
                                + 0.4 * y_axis.confidence)
            r.reason = f"{len(box_stats)} boxes"
        except Exception as e:
            r.status = ExtractionStatus.ERROR
            r.reason = f"{type(e).__name__}: {e}"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r
