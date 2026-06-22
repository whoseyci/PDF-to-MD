"""tesseract-based axis detection + linear pixel↔value calibration."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import pytesseract
    from pytesseract import Output
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

_NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")


@dataclass
class Word:
    text: str
    x: int; y: int; w: int; h: int
    conf: float
    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2


@dataclass
class AxisCalibration:
    axis: str
    p_to_v: Tuple[float, float]
    ticks: List[Tuple[int, float]]
    confidence: float
    perp_band_pixel: int
    label_text: Optional[str] = None
    def pixel_to_value(self, p): s, b = self.p_to_v; return s * p + b


def ocr_words(image_path: Path) -> List[Word]:
    if not _HAS_TESS: return []
    try: img = Image.open(image_path).convert("RGB")
    except Exception: return []
    w, h = img.size
    scale = 1
    if max(w, h) < 800:
        scale = max(1, 800 // max(w, h) + 1)
        img = img.resize((w * scale, h * scale), Image.LANCZOS)
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
    except Exception:
        return []
    words = []
    for i in range(len(data["text"])):
        t = (data["text"][i] or "").strip()
        if not t: continue
        try: conf = float(data["conf"][i])
        except (TypeError, ValueError): conf = -1.0
        if conf < 30: continue
        words.append(Word(text=t,
                            x=int(data["left"][i]) // scale,
                            y=int(data["top"][i]) // scale,
                            w=max(1, int(data["width"][i]) // scale),
                            h=max(1, int(data["height"][i]) // scale),
                            conf=conf))
    return words


def _parse_num(s):
    s = s.strip().rstrip("%")
    if not _NUM_RE.match(s): return None
    s2 = s.replace(",", ".") if s.count(",") == 1 and "." not in s \
        else s.replace(",", "")
    try: return float(s2)
    except ValueError: return None


def find_axes(words, image_size):
    W, H = image_size
    numerics = [(w, _parse_num(w.text)) for w in words
                 if _parse_num(w.text) is not None]
    y_axis = _fit_axis(numerics, axis="y", image_size=(W, H))
    x_axis = _fit_axis(numerics, axis="x", image_size=(W, H))
    return x_axis, y_axis


def _fit_axis(numerics, *, axis, image_size):
    if len(numerics) < 2: return None
    W, H = image_size
    if axis == "y":
        coords = [(w.cx, w, v) for (w, v) in numerics]
        band_tol = max(8, W * 0.06)
    else:
        coords = [(w.cy, w, v) for (w, v) in numerics]
        band_tol = max(8, H * 0.04)
    coords.sort(key=lambda t: t[0])
    bands = []; band_centers = []
    for c, w, v in coords:
        placed = False
        for i, bc in enumerate(band_centers):
            if abs(c - bc) <= band_tol:
                bands[i].append((w, v))
                band_centers[i] = (bc * len(bands[i]) + c) / (len(bands[i]) + 1)
                placed = True; break
        if not placed:
            bands.append([(w, v)]); band_centers.append(c)
    best = None; best_score = -1.0
    for band in bands:
        if len(band) < 2: continue
        if axis == "y":
            xs_full = np.array([w.cy for (w, _v) in band], dtype=float)
            perps   = np.array([w.cx for (w, _v) in band], dtype=float)
        else:
            xs_full = np.array([w.cx for (w, _v) in band], dtype=float)
            perps   = np.array([w.cy for (w, _v) in band], dtype=float)
        ys_full = np.array([v for (_w, v) in band], dtype=float)
        keep = np.ones(len(xs_full), dtype=bool)
        slope = intercept = r2 = 0.0
        while keep.sum() >= 2:
            xs, ys = xs_full[keep], ys_full[keep]
            if len(set(xs.tolist())) < 2: break
            slope, intercept = np.polyfit(xs, ys, 1)
            pred = slope * xs + intercept
            resid = np.abs(ys - pred)
            ss_res = float((resid ** 2).sum())
            ss_tot = float(((ys - ys.mean()) ** 2).sum()) or 1e-9
            r2 = 1 - ss_res / ss_tot
            if r2 >= 0.99 or keep.sum() == 2: break
            worst_local = int(np.argmax(resid))
            gi = np.flatnonzero(keep); keep[gi[worst_local]] = False
        if axis == "y" and slope >= 0: continue
        if axis == "x" and slope <= 0: continue
        if r2 < 0.95 or keep.sum() < 2: continue
        used_xs = xs_full[keep]; used_ys = ys_full[keep]
        used_perps = perps[keep]
        perp_center = float(used_perps.mean())
        score = r2 * keep.sum()
        if score > best_score:
            ticks = sorted(zip(used_xs.astype(int).tolist(), used_ys.tolist()),
                            key=lambda t: t[1])
            best = AxisCalibration(axis=axis,
                                     p_to_v=(float(slope), float(intercept)),
                                     ticks=ticks, confidence=float(r2),
                                     perp_band_pixel=int(perp_center))
            best_score = score
    return best


def find_label_text(words, *, near, axis_cal, image_size, skip_first_band=False):
    W, H = image_size
    out = []
    for w in words:
        if _parse_num(w.text) is not None: continue
        if len(w.text) < 2: continue
        if near == "y":
            if w.cx >= axis_cal.perp_band_pixel - 2: continue
            d = axis_cal.perp_band_pixel - w.cx
        else:
            if w.cy <= axis_cal.perp_band_pixel + 2: continue
            d = w.cy - axis_cal.perp_band_pixel
        out.append((d, w))
    if not out: return None
    out.sort(key=lambda t: t[0])
    band_tol = max(30.0, 0.06 * (W if near == "y" else H))
    if not skip_first_band:
        near_dist = out[0][0]
        title_words = [w for (d, w) in out if d <= near_dist + band_tol]
    else:
        near_dist = out[0][0]
        tick_band_end = near_dist + band_tol
        beyond = [(d, w) for (d, w) in out if d > tick_band_end]
        if not beyond: return None
        title_near = beyond[0][0]
        title_words = [w for (d, w) in beyond if d <= title_near + band_tol]
    if not title_words: return None
    title_words.sort(key=lambda w: (w.cy if near == "y" else w.cx))
    seed = title_words[0]
    if near == "y":
        line = [w for w in title_words if abs(w.cx - seed.cx) < 80]
        line.sort(key=lambda w: w.cy)
    else:
        line = [w for w in title_words if abs(w.cy - seed.cy) < 30]
        line.sort(key=lambda w: w.cx)
    text = " ".join(w.text for w in line).strip()
    return text or None
