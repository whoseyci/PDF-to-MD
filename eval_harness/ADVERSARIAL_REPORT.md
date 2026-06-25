# Adversarial bench

Cases: **18** = 9 adversarial figures × 2 caption conditions

## Overall: **16/18 = 88.9% correct**

## By variant

| variant | description | correct/n |
|---|---|---|
| bar_styled_as_box | bar with internal horizontal cap (looks like box median line) | 2/2 (100.0%) |
| scatter_with_regression | scatter + best-fit line (looks like line plot) | 2/2 (100.0%) |
| pie_with_legend_box | pie + bordered legend (extra rect could look like axis) | 2/2 (100.0%) |
| box_with_outliers | box plot with many outlier dots (looks like scatter) | 2/2 (100.0%) |
| horizontal_bars | horizontal bar chart | 2/2 (100.0%) |
| chart_with_colorful_legend | line chart with multi-color legend (could look stacked) | 2/2 (100.0%) |
| grouped_bars | multi-group bars (could look like scatter) | 2/2 (100.0%) |
| negative_values | bars with negative values | 2/2 (100.0%) |
| log_scale | bar chart with log-scale y axis | 0/2 (0.0%) |

## Failures

| truth | variant | cap | picked | status | arb |
|---|---|---|---|---|---|
| bar_chart | log_scale | rich | line_plot | no_axis | highest quality (0.3) was line_plot/no_axis |
| bar_chart | log_scale | empty | line_plot | no_axis | highest quality (0.3) was line_plot/no_axis |