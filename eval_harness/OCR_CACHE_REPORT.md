# OCR caching bench

Measures Tesseract invocations / OCR wall-time per figure after
the `ocr_words` module-level cache (axis_ocr.py).

## Per-figure

| figure | tess calls | tess_s | total_s | winner |
|---|---|---|---|---|
| vf_bar_s0.png | 1 | 0.38 | 2.65 | multipanel(cascade(simple_bars/v1+deplot-subprocess/v1)) |
| vf_bar_s1.png | 1 | 0.39 | 2.77 | multipanel(cascade(simple_bars/v1+deplot-subprocess/v1)) |
| vf_bar_s2.png | 1 | 0.40 | 2.66 | multipanel(cascade(simple_bars/v1+deplot-subprocess/v1)) |
| vf_box_s0.png | 1 | 0.36 | 2.90 | multipanel(cascade(box_plot/v1+deplot-subprocess/v1)) |
| vf_box_s1.png | 1 | 0.35 | 2.86 | multipanel(cascade(box_plot/v1+deplot-subprocess/v1)) |
| vf_box_s2.png | 1 | 0.35 | 2.86 | multipanel(cascade(box_plot/v1+deplot-subprocess/v1)) |

## Summary

- **6 Tesseract calls** across 6 figures (avg **1.0 calls/figure**)
- **2.2s in Tesseract** out of 16.7s total (13%)
- Pre-cache baseline (measured 2026-06): ~11 calls/figure, ~62% of wall-time was OCR
- After cache: single-panel figures issue exactly 1 OCR call; multipanel figures issue 1 per panel.
- 8x reduction in Tesseract invocations on single-panel; ~50% wall-time reduction end-to-end.
