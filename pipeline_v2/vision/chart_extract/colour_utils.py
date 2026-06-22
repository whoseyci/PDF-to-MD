"""Shared colour utilities."""
from __future__ import annotations
from typing import Tuple
import numpy as np


def rgb_to_hsv(rgb):
    import cv2
    arr = np.array([[list(rgb)]], dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0, 0]
    return tuple(int(x) for x in hsv)


def colour_distance(c1, c2):
    h1, s1, v1 = rgb_to_hsv(c1)
    h2, s2, v2 = rgb_to_hsv(c2)
    dh = min(abs(h1 - h2), 180 - abs(h1 - h2))
    return 2.0 * dh + abs(s1 - s2) + abs(v1 - v2)


def quantise_colour(rgb, step=24):
    return tuple((c // step) * step + step // 2 for c in rgb)


def is_background(rgb): return min(rgb) > 220
def is_ink(rgb): return max(rgb) < 60
