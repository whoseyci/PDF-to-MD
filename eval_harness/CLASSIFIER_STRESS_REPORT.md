# Classifier stress bench

Cases: **81** total = 27 figures × 3 caption conditions

Caption conditions:
* `rich`: descriptive caption ('Figure 1. Bar chart of yield by treatment')
* `minimal`: just 'Figure 1.'
* `empty`: no caption at all (image only)

## Accuracy by caption condition

| Condition | Keyword | Mixture | Hybrid |
|---|---|---|---|
| rich | 21/27 (77.8%) | 24/27 (88.9%) | 24/27 (88.9%) |
| minimal | 0/27 (0.0%) | 10/27 (37.0%) | 10/27 (37.0%) |
| empty | 0/27 (0.0%) | 10/27 (37.0%) | 10/27 (37.0%) |

## Accuracy by truth kind (across all caption conditions)

| Kind | n | Keyword | Mixture | Hybrid |
|---|---|---|---|---|
| bar_chart | 18 | 6/18 (33.3%) | 6/18 (33.3%) | 6/18 (33.3%) |
| box_plot | 9 | 3/9 (33.3%) | 3/9 (33.3%) | 3/9 (33.3%) |
| decorative | 6 | 0/6 (0.0%) | 6/6 (100.0%) | 6/6 (100.0%) |
| equation | 3 | 0/3 (0.0%) | 3/3 (100.0%) | 3/3 (100.0%) |
| flow_diagram | 6 | 2/6 (33.3%) | 2/6 (33.3%) | 2/6 (33.3%) |
| line_plot | 12 | 4/12 (33.3%) | 12/12 (100.0%) | 12/12 (100.0%) |
| pie_chart | 12 | 4/12 (33.3%) | 10/12 (83.3%) | 10/12 (83.3%) |
| scatter_plot | 9 | 0/9 (0.0%) | 0/9 (0.0%) | 0/9 (0.0%) |
| stacked_bar_chart | 6 | 2/6 (33.3%) | 2/6 (33.3%) | 2/6 (33.3%) |

## Confusion matrix — Hybrid (collapsed across caption conditions)

| truth ↓ pred → | bar_chart | box_plot | decorative | equation | flow_diagram | line_plot | pie_chart | scatter_plot | stacked_bar_chart | table_as_image |
|---|---|---|---|---|---|---|---|---|---|---|
| **bar_chart** | 6 | . | . | . | . | 12 | . | . | . | .  |
| **box_plot** | . | 3 | . | . | . | 6 | . | . | . | .  |
| **decorative** | . | . | 6 | . | . | . | . | . | . | .  |
| **equation** | . | . | . | 3 | . | . | . | . | . | .  |
| **flow_diagram** | . | . | . | . | 2 | 4 | . | . | . | .  |
| **line_plot** | . | . | . | . | . | 12 | . | . | . | .  |
| **pie_chart** | . | . | . | . | . | 2 | 10 | . | . | .  |
| **scatter_plot** | . | 3 | . | . | . | 3 | . | . | . | 3  |
| **stacked_bar_chart** | . | . | . | . | . | 4 | . | . | 2 | .  |
| **table_as_image** | . | . | . | . | . | . | . | . | . | .  |

## Specific failures (hybrid)

Total wrong: 37/81

| truth | variant | cap | hybrid_said | mix_conf | ocr_len |
|---|---|---|---|---|---|
| scatter_plot | clean_0 | rich | table_as_image | 0.573 | 2 |
| scatter_plot | clean_1 | rich | box_plot | 0.55 | 0 |
| scatter_plot | dense | rich | line_plot | 0.584 | 3 |
| bar_chart | clean_0 | minimal | line_plot | 0.582 | 6 |
| bar_chart | clean_1 | minimal | line_plot | 0.576 | 13 |
| bar_chart | clean_2 | minimal | line_plot | 0.592 | 47 |
| bar_chart | grayscale | minimal | line_plot | 0.709 | 37 |
| bar_chart | low_dpi | minimal | line_plot | 0.75 | 0 |
| bar_chart | legend | minimal | line_plot | 0.626 | 51 |
| stacked_bar_chart | clean_0 | minimal | line_plot | 0.616 | 33 |
| stacked_bar_chart | clean_1 | minimal | line_plot | 0.613 | 33 |
| pie_chart | donut | minimal | line_plot | 0.573 | 0 |
| scatter_plot | clean_0 | minimal | table_as_image | 0.573 | 2 |
| scatter_plot | clean_1 | minimal | box_plot | 0.55 | 0 |
| scatter_plot | dense | minimal | line_plot | 0.584 | 3 |
| box_plot | clean_0 | minimal | line_plot | 0.652 | 13 |
| box_plot | clean_1 | minimal | line_plot | 0.653 | 16 |
| box_plot | grayscale | minimal | line_plot | 0.671 | 0 |
| flow_diagram | clean_0 | minimal | line_plot | 0.297 | 0 |
| flow_diagram | clean_1 | minimal | line_plot | 0.309 | 0 |
| bar_chart | clean_0 | empty | line_plot | 0.582 | 6 |
| bar_chart | clean_1 | empty | line_plot | 0.576 | 13 |
| bar_chart | clean_2 | empty | line_plot | 0.592 | 47 |
| bar_chart | grayscale | empty | line_plot | 0.709 | 37 |
| bar_chart | low_dpi | empty | line_plot | 0.75 | 0 |
| bar_chart | legend | empty | line_plot | 0.626 | 51 |
| stacked_bar_chart | clean_0 | empty | line_plot | 0.616 | 33 |
| stacked_bar_chart | clean_1 | empty | line_plot | 0.613 | 33 |
| pie_chart | donut | empty | line_plot | 0.573 | 0 |
| scatter_plot | clean_0 | empty | table_as_image | 0.573 | 2 |
| scatter_plot | clean_1 | empty | box_plot | 0.55 | 0 |
| scatter_plot | dense | empty | line_plot | 0.584 | 3 |
| box_plot | clean_0 | empty | line_plot | 0.652 | 13 |
| box_plot | clean_1 | empty | line_plot | 0.653 | 16 |
| box_plot | grayscale | empty | line_plot | 0.671 | 0 |
| flow_diagram | clean_0 | empty | line_plot | 0.297 | 0 |
| flow_diagram | clean_1 | empty | line_plot | 0.309 | 0 |