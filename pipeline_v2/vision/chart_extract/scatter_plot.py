"""E8 -- Scatter plot geometric extractor."""
from __future__ import annotations
import time

import numpy as np

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus
from .axis_ocr import ocr_words, find_axes, find_label_text


class ScatterExtractor(ChartExtractor):
    name = "scatter/v1"

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
            if x1 - x0 < 20 or y1 - y0 < 20:
                r.status = ExtractionStatus.NO_BARS
                r.elapsed_seconds = time.time() - t0; return r
            r.plot_area = [int(x0), int(y0), int(x1), int(y1)]
            plot = bgr[y0:y1, x0:x1]
            hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
            non_white = ((hsv[:, :, 1] > 35) | (hsv[:, :, 2] < 220)).astype(np.uint8) * 255
            non_white = cv2.morphologyEx(
                non_white, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            n_comp, _, stats, centroids = cv2.connectedComponentsWithStats(
                non_white, 8)
            markers = []
            for i in range(1, n_comp):
                x, y, w, h, area = stats[i]
                if area < 4 or area > 0.01 * (x1 - x0) * (y1 - y0):
                    continue
                if max(w, h) > 30:
                    continue
                aspect = max(w, h) / max(1, min(w, h))
                if aspect > 2.5:
                    continue
                cx, cy = centroids[i]
                xx, yy = int(cx), int(cy)
                xx = max(0, min(plot.shape[1] - 1, xx))
                yy = max(0, min(plot.shape[0] - 1, yy))
                h_, s_, v_ = hsv[yy, xx]
                key = (int(h_) // 12, int(s_) // 64, int(v_) // 64)
                markers.append((float(cx) + x0, float(cy) + y0, key))
            if len(markers) < 3:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "too few points"
                r.elapsed_seconds = time.time() - t0; return r
            # Self-rejection: if markers form a continuous line
            # (1 y-value per x-bin across most of the x-range),
            # this is a line plot, not a scatter.
            try:
                xs = np.array([m[0] for m in markers])
                ys = np.array([m[1] for m in markers])
                x_span = float(xs.max() - xs.min())
                if x_span > 0:
                    n_x_bins = max(10, int(x_span / 6))
                    bins = np.linspace(xs.min(), xs.max(), n_x_bins + 1)
                    bin_ids = np.digitize(xs, bins)
                    from collections import defaultdict as _dd
                    by_bin = _dd(list)
                    for bi, yv in zip(bin_ids, ys):
                        by_bin[bi].append(yv)
                    if by_bin:
                        avg_ys_per_bin = np.mean(
                            [len(v) for v in by_bin.values() if v])
                        n_bins_with_markers = sum(
                            1 for v in by_bin.values() if v)
                        if (avg_ys_per_bin < 1.5
                                and n_bins_with_markers > 0.7 * n_x_bins):
                            r.status = ExtractionStatus.NO_BARS
                            r.reason = (f"markers form a line "
                                          f"({n_bins_with_markers}/"
                                          f"{n_x_bins} bins, "
                                          f"{round(avg_ys_per_bin,2)} y/bin)"
                                          f"; not a scatter")
                            r.elapsed_seconds = time.time() - t0; return r
            except Exception:
                pass
            clusters: dict = {}
            for (mx, my, k) in markers:
                clusters.setdefault(k, []).append((mx, my))
            x_slope, x_int = x_axis.p_to_v
            y_slope, y_int = y_axis.p_to_v
            scatter_summary = []
            for k, pts in clusters.items():
                xs_v = [x_slope * x + x_int for (x, _) in pts]
                ys_v = [y_slope * y + y_int for (_, y) in pts]
                scatter_summary.append({
                    "color_key": list(k),
                    "n_points": len(pts),
                    "x_mean": round(float(np.mean(xs_v)), 3),
                    "y_mean": round(float(np.mean(ys_v)), 3),
                    "x_min": round(float(np.min(xs_v)), 3),
                    "x_max": round(float(np.max(xs_v)), 3),
                    "y_min": round(float(np.min(ys_v)), 3),
                    "y_max": round(float(np.max(ys_v)), 3),
                })
            scatter_summary.sort(key=lambda d: -d["n_points"])
            r.scatter_summary = scatter_summary
            r.value_axis = "y"
            r.value_label = find_label_text(words, near="y",
                                              axis_cal=y_axis,
                                              image_size=(W, H),
                                              skip_first_band=False)
            r.category_label = find_label_text(words, near="x",
                                                 axis_cal=x_axis,
                                                 image_size=(W, H),
                                                 skip_first_band=False)
            r.status = ExtractionStatus.OK if len(markers) >= 5 \
                else ExtractionStatus.PARTIAL
            r.confidence = min(0.85,
                                0.3 + 0.05 * len(scatter_summary)
                                + 0.4 * (x_axis.confidence + y_axis.confidence) / 2)
            r.reason = f"{len(markers)} markers, {len(scatter_summary)} clusters"
        except Exception as e:
            r.status = ExtractionStatus.ERROR
            r.reason = f"{type(e).__name__}: {e}"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r
