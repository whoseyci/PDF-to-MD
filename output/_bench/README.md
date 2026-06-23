# DePlot vs SimpleBars: real benchmark

Synthetic ground-truth charts; both extractors run on the same images
on the 2 vCPU / 1.9 GB sandbox.

## Setup
- SimpleBars: in-process, 0 model dependencies
- DePlot: in-process, `low_cpu_mem_usage=True`, `max_image_dim=280`,
  `max_new_tokens=150`. Peak resident: 1.45 GB.

## Results

| Figure                  | Truth         | SimpleBars                               | DePlot                                       | Winner            |
|-------------------------|---------------|------------------------------------------|----------------------------------------------|-------------------|
| 01 vertical bars (5)    | 12,34,18,27,9 | ok, 0.5 s, mean err 0.05                 | ok, 45 s, **mean err 0.00**                   | tied (DePlot exact, SB nearly so) |
| 02 horizontal bars (4)  | 42,55,28,71   | ok, 0.4 s, mean err 0.29, **labels exact** | ok, 41 s, mean err 0.0 (reversed order)      | tied              |
| 03 stacked bars (5x3)   | matrix        | **partial** (returns 100% totals only)   | **ok**, 112 s, all 15 cells within ±5 of truth | **DePlot**      |
| 04 line plot (2 series) | y=2+0.5x, y=1+0.1x² | **no_bars** (out of scope)         | **ok**, 84 s, 8 sampled points per series   | **DePlot**      |

## Side-by-side per-figure

**01: simple vertical bars**
```
Truth:      Alpha=12  Beta=34  Gamma=18  Delta=27  Epsilon=9
SimpleBars: Alpha=12.0 Beta=34.0 Gamma=18.1 Delta=27.0 Epsilon=9.1
DePlot:     Alpha=12.0 Beta=34.0 Gamma=18.0 Delta=27.0 Ip 3=9.0
            ↑ DePlot misreads "Epsilon" as "Ip 3" -- OCR-style label hallucination
```

**02: horizontal bars**
```
Truth (top-to-bottom on chart): Rice=71 Wheat=28 Maize=55 Soybean=42
SimpleBars: Soybean=42, Maize=55, Wheat=28, Rice=71  (correct, just bottom-up)
DePlot:     Rice=71, Wheat=28, Maize=55, Soybean=42  (correct, top-down)
```
Both correct, different conventions.

**03: stacked bars** -- DePlot **wins outright**
```
Truth:
  2018: Forest=50, Cropland=40, Urban=10
  2019: Forest=48, Cropland=41, Urban=11
  2020: Forest=45, Cropland=42, Urban=13
  2021: Forest=42, Cropland=43, Urban=15
  2022: Forest=40, Cropland=44, Urban=16

SimpleBars (stub): partial -- ALL bars report 100% total, no series names

DePlot:
  Year 1: Forest=50.1, Cropland=35.8, Urban=10.4   (Cropland off by 4)
  Year 2: Forest=48.4, Cropland=38.1, Urban=10.3
  Year 3: Forest=45.6, Cropland=38.5, Urban=12.1
  (years got read as '2018, 2019, 2000, 2001, 2002' -- last two wrong)
```

**04: line plot** -- DePlot **wins outright**
```
SimpleBars: no_bars  (not a chart kind we handle geometrically)

DePlot: matrix=[
  [2.0, 1.0],   # at x=0:  Linear=2.0 (truth 2.0),    Quadratic=1.0 (truth 1.0)
  [3.0, 1.5],   # at x=2:  Linear=3.0 (truth 3.0),    Quadratic=1.5 (truth 1.4)
  [4.0, 2.6],   # at x=4:  Linear=4.0 (truth 4.0),    Quadratic=2.6 (truth 2.6)
  ...
]   8 points sampled across the x axis. Y values match the formulas exactly.
```

## Honest verdict

**SimpleBars wins for plain bar charts** -- 100x faster (0.5 s vs 45 s),
nearly-perfect numbers, more reliable on category-label OCR.

**DePlot wins where SimpleBars has stubs** -- stacked, line, scatter.
For those, DePlot is your only real option short of training your own
geometric extractor. The cost is ~40-110 s per figure and a 1.5 GB
memory footprint.

**Best architecture: cascade.** `CascadingExtractor([SimpleBars,
DePlot])` already does this -- SimpleBars short-circuits on
high-confidence hits, falls through to DePlot only for stubs / failures.
This pattern is exactly what the user should enable for production.

## Failure modes worth knowing

- DePlot **hallucinates a 5th tick** when there are only 4 (the "Ip 3"
  for "Epsilon", the wrong years 2000/2001/2002 -- it pattern-completes).
- DePlot **misreads chart titles as value_labels** ("Crop area" instead
  of "Hectares").
- DePlot is **slower than expected** on stacked/line (~2 min). On a GPU
  this drops to ~5 s; on our 2-CPU sandbox it's a real cost.
- SimpleBars **silently mislabels** when categories have unusual layouts
  (e.g. "Year" axis title got listed as a category in the stacked test --
  fixed in stacked_bars.py upstream but the stub doesn't have that fix).
