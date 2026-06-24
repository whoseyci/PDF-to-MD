"""E8 -- Pie-chart geometric extractor."""
from __future__ import annotations
import math
import time

import numpy as np

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus


class PieChartExtractor(ChartExtractor):
    name = "pie_chart/v1"

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
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                          minDist=max(W, H),
                                          param1=80, param2=30,
                                          minRadius=int(0.10 * min(W, H)),
                                          maxRadius=int(0.45 * min(W, H)))
            if circles is None:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "no pie circle"
                r.elapsed_seconds = time.time() - t0; return r
            cx, cy, rad = circles[0][0]
            cx, cy, rad = int(cx), int(cy), int(rad)
            r.plot_area = [cx - rad, cy - rad, cx + rad, cy + rad]
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            def colour_key(px):
                h, s, v = int(px[0]), int(px[1]), int(px[2])
                if s < 30 and v > 200: return None
                if v < 40: return ("k",)
                return (h // 12, s // 64, v // 64)

            sample_r = int(rad * 0.7)
            n_angles = 720
            keys = []
            for k in range(n_angles):
                theta = (k / n_angles) * 2 * math.pi - math.pi / 2
                x = int(round(cx + sample_r * math.cos(theta)))
                y = int(round(cy + sample_r * math.sin(theta)))
                if 0 <= x < W and 0 <= y < H:
                    keys.append(colour_key(hsv[y, x]))
                else:
                    keys.append(None)

            wedges = []
            i = 0
            while i < n_angles:
                if keys[i] is None:
                    i += 1; continue
                k = keys[i]
                j = i
                while j < n_angles and keys[j] == k:
                    j += 1
                wedges.append((k, i, j))
                i = j
            if (len(wedges) >= 2
                    and wedges[0][0] == wedges[-1][0]
                    and wedges[0][1] == 0
                    and wedges[-1][2] == n_angles):
                first = wedges[0]; last = wedges[-1]
                merged = (first[0], last[1] - n_angles, first[2])
                wedges = [merged] + wedges[1:-1]
            wedges = [w for w in wedges if (w[2] - w[1]) >= 4]
            if not wedges:
                r.status = ExtractionStatus.NO_BARS
                r.reason = "no wedges"
                r.elapsed_seconds = time.time() - t0; return r
            by_key: dict = {}
            for (k, a, b) in wedges:
                by_key[k] = by_key.get(k, 0) + (b - a)
            total = sum(by_key.values())
            # Self-rejection: a real pie covers most of the 360° circle.
            # Wedges covering <80% means we found a circular blob in
            # something else (scatter cluster, bar chart artifact).
            coverage = total / n_angles
            if coverage < 0.80:
                r.status = ExtractionStatus.NO_BARS
                r.reason = (f"wedges cover only {round(100*coverage,1)}% "
                              f"of circle; not a pie")
                r.elapsed_seconds = time.time() - t0; return r
            # Self-rejection: verify the detected circle is actually
            # the figure's dominant content by sampling outside it --
            # should be mostly background (white).
            try:
                check_r = int(rad * 1.15)
                outside_bg = 0; outside_total = 0
                for k in range(0, n_angles, 10):
                    theta = (k / n_angles) * 2 * math.pi - math.pi / 2
                    x = int(round(cx + check_r * math.cos(theta)))
                    y = int(round(cy + check_r * math.sin(theta)))
                    if 0 <= x < W and 0 <= y < H:
                        outside_total += 1
                        s = int(hsv[y, x, 1]); v = int(hsv[y, x, 2])
                        if (s < 30 and v > 200) or v < 40:
                            outside_bg += 1
                if outside_total > 0:
                    bg_frac = outside_bg / outside_total
                    if bg_frac < 0.40:
                        r.status = ExtractionStatus.NO_BARS
                        r.reason = (f"only {round(100*bg_frac,1)}% bg "
                                      f"outside circle; not a pie")
                        r.elapsed_seconds = time.time() - t0; return r
            except Exception:
                pass
            pie_slices = []
            for k, span in sorted(by_key.items(), key=lambda kv: -kv[1]):
                pct = round(100.0 * span / total, 2)
                pie_slices.append({
                    "label": f"Slice {len(pie_slices) + 1}",
                    "color_key": list(k) if isinstance(k, tuple) else [k],
                    "angle_degrees": round(360.0 * span / total, 2),
                    "percent": pct,
                })
            r.pie_slices = pie_slices
            r.categories = [s["label"] for s in pie_slices]
            r.values = [s["percent"] for s in pie_slices]
            r.calibration = {"center": [cx, cy], "radius": rad}
            r.status = ExtractionStatus.OK if 2 <= len(pie_slices) <= 12 \
                else ExtractionStatus.PARTIAL
            r.confidence = 0.65 if r.status == ExtractionStatus.OK else 0.4
            r.reason = f"{len(pie_slices)} wedges"
        except Exception as e:
            r.status = ExtractionStatus.ERROR
            r.reason = f"{type(e).__name__}: {e}"
        r.elapsed_seconds = round(time.time() - t0, 3)
        return r
