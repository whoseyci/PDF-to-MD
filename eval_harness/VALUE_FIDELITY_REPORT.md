# Value-fidelity bench

Cases: **15** = 5 kinds × 3 seeds.
Direct = call the right specialist by name.
Smart = production `run_smart_extraction`.

## Per-kind value error

| kind | n | direct OK | smart picked right | avg direct err | avg smart err |
|---|---|---|---|---|---|
| bar | 3 | 3/3 | 3/3 | 0.08 | 0.08 |
| pie | 3 | 3/3 | 3/3 | 0.08 | 0.08 |
| line | 3 | 3/3 | 3/3 | 0.12 | 0.12 |
| box | 3 | 3/3 | 3/3 | 0.99 | 0.99 |
| stacked | 3 | 3/3 | 3/3 | 0.11 | 0.11 |

## Per-case detail

| kind | seed | direct status | direct err | smart picked | smart err |
|---|---|---|---|---|---|
| bar | 0 | ok | 0.09 | bar_chart | 0.09 |
| bar | 1 | ok | 0.09 | bar_chart | 0.09 |
| bar | 2 | ok | 0.05 | bar_chart | 0.05 |
| pie | 0 | ok | 0.06 | pie_chart | 0.06 |
| pie | 1 | ok | 0.11 | pie_chart | 0.11 |
| pie | 2 | ok | 0.08 | pie_chart | 0.08 |
| line | 0 | ok | 0.14 | line_plot | 0.14 |
| line | 1 | ok | 0.09 | line_plot | 0.09 |
| line | 2 | ok | 0.12 | line_plot | 0.12 |
| box | 0 | ok | 1.02 | box_plot | 1.02 |
| box | 1 | ok | 0.95 | box_plot | 0.95 |
| box | 2 | ok | 1.0 | box_plot | 1.0 |
| stacked | 0 | ok | 0.01 | stacked_bar_chart | 0.01 |
| stacked | 1 | ok | 0.02 | stacked_bar_chart | 0.02 |
| stacked | 2 | ok | 0.29 | stacked_bar_chart | 0.29 |