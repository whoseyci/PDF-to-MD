# Real corpus extraction bench

Ran `run_smart_extraction` on **10** raster images extracted from **3** PDFs.
Total runtime: **567.5s** (~56.75s per figure).

## Aggregate

* **OK extractions:** 4/10 = **40.0%**
* PARTIAL extractions: 5/10 = 50.0%
* Classified as decorative (skipped): 0/10 = 0.0%
* Other (NO_BARS / NO_AXIS / ERROR): 1/10 = 10.0%

## By winning kind

| kind | n |
|---|---|
| flow_diagram | 5 |
| scatter_plot | 2 |
| box_plot | 2 |
| stacked_bar_chart | 1 |

## By status

| status | n |
|---|---|
| partial | 5 |
| ok | 4 |
| no_bars | 1 |

## Per-paper

| paper | n figs | OK | partial | decorative | other | pct_ok | total s |
|---|---|---|---|---|---|---|---|
| carceles-rodriguez-et-al-2022 | 2 | 2 | 0 | 0 | 0 | 100.0% | 49.0 |
| baden-bohm-2023 | 4 | 1 | 2 | 0 | 1 | 25.0% | 205.9 |
| cast-model-paper-2026 | 4 | 1 | 3 | 0 | 0 | 25.0% | 311.8 |

## Worst-N papers

| paper | n figs | pct_ok | top failure status |
|---|---|---|---|
| baden-bohm-2023 | 4 | 25.0% | partial (2) |
| cast-model-paper-2026 | 4 | 25.0% | partial (3) |
| carceles-rodriguez-et-al-2022 | 2 | 100.0% | ? |