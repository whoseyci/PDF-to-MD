# Mixture (E15) + Reflective (E17) bench

Cases: 14 synthetic figures across 7 kinds (×2 seeds each).

## Classifier — WITH explicit captions

| Classifier | hit-rate | avg s |
|---|---|---|
| keyword (old) | 14/14 = 100.0% | 0.0s |
| **mixture (E15)** | **11/14 = 78.6%** | 0.024s |

## Classifier — WITHOUT captions (image-only stress test)

| Classifier | hit-rate |
|---|---|
| keyword (old) | 0/14 = 0.0% |
| **mixture (E15)** | **4/14 = 28.6%** |

## Extractor success — WITH captions

| Strategy | OK rate | avg s |
|---|---|---|
| vanilla (truth kind) | 12/14 = 85.7% | 1.001s |
| **reflective (E17)** | **12/14 = 85.7%** | 2.091s |

## Extractor success — WITHOUT captions

| Strategy | OK rate |
|---|---|
| vanilla (truth kind, cheating) | 12/14 = 85.7% |
| **reflective (E17)** | **7/14 = 50.0%** |

## Per-case detail

| truth | old? | mix? | mix.top (conf) | vanilla | refl | refl_kind | steps |
|---|---|---|---|---|---|---|---|
| bar_chart_s0 | ✅ | ✅ | bar_chart (0.45) | ok (0.95) | ok (0.95) | bar_chart | 1 |
| bar_chart_s1 | ✅ | ❌ line_plot | line_plot (0.42) | ok (0.95) | ok (0.95) | bar_chart | 2 |
| stacked_bar_chart_s0 | ✅ | ✅ | stacked_bar_chart (1.00) | ok (0.90) | ok (0.90) | stacked_bar_chart | 1 |
| stacked_bar_chart_s1 | ✅ | ✅ | stacked_bar_chart (1.00) | ok (0.90) | ok (0.90) | stacked_bar_chart | 1 |
| pie_chart_s0 | ✅ | ✅ | pie_chart (0.78) | ok (0.65) | ok (0.65) | pie_chart | 1 |
| pie_chart_s1 | ✅ | ✅ | pie_chart (0.78) | ok (0.65) | ok (0.65) | pie_chart | 1 |
| line_plot_s0 | ✅ | ✅ | line_plot (0.69) | ok (0.85) | ok (0.85) | line_plot | 1 |
| line_plot_s1 | ✅ | ✅ | line_plot (0.69) | ok (0.85) | ok (0.85) | line_plot | 1 |
| scatter_plot_s0 | ✅ | ❌ line_plot | line_plot (0.38) | ok (0.95) | ok (0.95) | scatter_plot | 2 |
| scatter_plot_s1 | ✅ | ❌ line_plot | line_plot (0.40) | ok (0.85) | ok (0.85) | scatter_plot | 2 |
| box_plot_s0 | ✅ | ✅ | box_plot (0.67) | ok (0.85) | ok (0.85) | box_plot | 1 |
| box_plot_s1 | ✅ | ✅ | box_plot (0.67) | ok (0.85) | ok (0.85) | box_plot | 1 |
| flow_diagram_s0 | ✅ | ✅ | flow_diagram (1.00) | no_extractor (0.00) | ocr_failed (0.00) | line_plot | 4 |
| flow_diagram_s1 | ✅ | ✅ | flow_diagram (1.00) | no_extractor (0.00) | ocr_failed (0.00) | line_plot | 4 |
