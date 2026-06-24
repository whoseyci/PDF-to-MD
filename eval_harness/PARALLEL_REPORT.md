# Parallel-extractor bench (vs reflective)

Cases: **34** = 17 figures × 2 caption conditions.
Truth = expected winning extractor kind.

## Aggregate

| Strategy | correct/total | avg s | avg extractors run |
|---|---|---|---|
| Reflective (kind-ladder) | 28/34 = 82.4% | 1.75s | (varies) |
| Parallel + classifier hint | 32/34 = 94.1% | 4.69s | 5.0 |
| **Smart (caption-decisive→reflective, else→parallel)** | **32/34 = 94.1%** | 4.19s | 4.3 |

## By caption condition

| Condition | Reflective | Parallel+hint | Smart |
|---|---|---|---|
| rich | 15/17 (88.2%) | 16/17 (94.1%) | 16/17 (94.1%) |
| empty | 13/17 (76.5%) | 16/17 (94.1%) | 16/17 (94.1%) |

## By truth kind

| Kind | n | Reflective | Parallel+hint | Smart |
|---|---|---|---|---|
| bar_chart | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| box_plot | 4 | 2/4 (50.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| decorative | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| equation | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) |
| flow_diagram | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| line_plot | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| pie_chart | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| scatter_plot | 4 | 0/4 (0.0%) | 2/4 (50.0%) | 2/4 (50.0%) |
| stacked_bar_chart | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |

## Smart failures

| truth | variant | cap | smart picked | status |
|---|---|---|---|---|
| scatter_plot | clean_1 | rich | flow_diagram | partial |
| scatter_plot | clean_1 | empty | flow_diagram | partial |