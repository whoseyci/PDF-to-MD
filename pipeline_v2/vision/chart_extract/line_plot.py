"""E8 -- Line plot geometric extractor."""
from __future__ import annotations
import time
from collections import defaultdict
from statistics import median

import numpy as np

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus
from .axis_ocr import ocr_words, find_axes, find_label_text


class LinePlotExtractor(ChartExtractor):
    name = "line_plot/v1"

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
            x_axis, y_axis = find_axes(words, (W, H))
            if not x_axis or not y_axis:
                r.status = ExtractionStatus.NO_AXIS
                r.elapsed_seconds = time.time() - t0; return r
            x0 = y_axis.perp_band_pixel + 2
            y1 = x_axis.perp_band_pixel - 2
            xt = [t[0] for t in x_axis.ticks]
            yt = [t[0] for t in y_axis.ticks]
            x1 = max(xt) if xt else W - 1
            y0 = min(yt) if yt else 0
            if x1 - x0 < 30 or y1 - y0 < 30:
                r.status = ExtractionStatus.NO_BARS
                r.elapsed_seconds = time.time() - t0; return r
            r.plot_area = [int(x0), int(y0), int(x1), int(y1)]
            plot = bgr[y0:y1, x0:x1]
            hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
            non_white = ((hsv[:, :, 1] > 35) | (hsv[:, :, 2] < 200))
            ys_, xs_ = np.where(non_white)
            if len(xs_) < 20:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "no line pixels"
                r.elapsed_seconds = time.time() - t0; return r

            def key_for(px):
                h, s, v = int(px[0]), int(px[1]), int(px[2])
                if s < 35 or v < 40:
                    return ("k",)
                return (int(h) // 18,)

            series_pixels: dict = defaultdict(list)
            for px, py in zip(xs_, ys_):
                k = key_for(hsv[py, px])
                series_pixels[k].append((px, py))
            series_pixels.pop(("k",), None)
            kept = []
            for k, pts in series_pixels.items():
                if len(pts) < 60: continue
                xs_only = [p[0] for p in pts]
                if (max(xs_only) - min(xs_only)) < 0.4 * (x1 - x0): continue
                kept.append((k, pts))
            kept.sort(key=lambda kp: -len(kp[1]))
            kept = kept[:6]
            if not kept:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "no series"
                r.elapsed_seconds = time.time() - t0; return r

            x_slope, x_int = x_axis.p_to_v
            y_slope, y_int = y_axis.p_to_v
            line_series = []
            for k, pts in kept:
                col: dict = defaultdict(list)
                for (px, py) in pts:
                    col[px].append(py)
                xs_sorted = sorted(col.keys())
                n_samples = min(12, len(xs_sorted))
                if n_samples < 3: continue
                idx = np.linspace(0, len(xs_sorted) - 1, n_samples).astype(int)
                samples = []
                for i in idx:
                    px = xs_sorted[i]
                    py = median(col[px])
                    xv = x_slope * (px + x0) + x_int
                    yv = y_slope * (py + y0) + y_int
                    samples.append([round(float(xv), 3),
                                      round(float(yv), 3)])
                line_series.append({
                    "color_key": list(k),
                    "n_points_sampled": len(samples),
                    "samples": samples,
                })
            if not line_series:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "no usable series"
                r.elapsed_seconds = time.time() - t0; return r
            r.line_series = line_series
            r.value_axis = "y"
            r.value_label = find_label_text(words, near="y",
                                              axis_cal=y_axis,
                                              image_size=(W, H),
                                              skip_first_band=False)
            r.category_label = find_label_text(words, near="x",
                                                 axis_cal=x_axis,
                                                 image_size=(W, H),
                                                 skip_first_band=False)
            r.status = ExtractionStatus.OK
            r.confidence = min(0.85,
                                0.3 + 0.1 * len(line_series)
                                + 0.4 * (x_axis.confidence + y_axis.confidence) / 2)
            r.reason = f"{len(line_series)} series"
        except Exception as e:
            r.status = ExtractionStatus.ERROR
            r.reason = f"{type(e).__name__}: {e}"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r
