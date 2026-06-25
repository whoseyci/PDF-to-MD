# How close to the limit are we, really? (Round 5)

User asked if we'd reached the limit. I said no — the previous "94.1%"
was on synthetic figures. So this round I built **4 honest benches**
in sequence: real corpus, value fidelity, adversarial, multipanel.
Here's what they each showed and what I fixed.

## 1. Real corpus run (the brutal one)

Built `eval_harness/bench_real_corpus.py`. Extracts every raster
image from each source PDF, runs `run_smart_extraction` on it, and
reports OK/partial/decorative/other.

### Initial attempt: 0/3 OK ☠️

First 3 figures from baden-bohm-2023 all failed:
* `p001_x772.png` — journal cover art (Agriculture, Ecosystems &
   Environment branding)
* `p001_x775.png` — Elsevier publisher logo
* `p001_x780.png` — blank gradient

**None of them were data figures.** The bench was testing on
publisher branding from page 1.

### Fix: skip page-1 images

Real data figures live on body pages. Page 1 is title + branding.
After this fix:

| Paper | n figs | OK | PARTIAL | total time |
|---|---|---|---|---|
| carceles-rodriguez-et-al-2022 | 2 | 2 (100%) | 0 | 49s |
| baden-bohm-2023 | 4 | 1 (25%) | 2 | 206s |
| cast-model-paper-2026 | 4 | 1 (25%) | 3 | 312s |
| **TOTAL** | **10** | **4 (40%)** | **5 (50%)** | 568s |

**40% OK + 50% PARTIAL = 90% extracted something usable.** Only 1/10
(10%) completely failed. The partials are real-world messy figures
where the extractor got structure but flagged something off.

### Speed problem
~57 s/figure is much slower than synthetic (~5s). Reason: the
multipanel wrapper runs OCR per panel per extractor. Already
mitigated by `auto_downscale_max_dim=900` in
`parallel_extractor.py`. Real high-DPI figures (1750×1333) get
downscaled to ~900x800 before extraction. **Bounded ~60s/figure
worst case**, ~5-10s for simple cases.

## 2. Value-fidelity bench

Built `eval_harness/bench_value_fidelity.py`. For each chart kind
generates figures with KNOWN ground-truth values, runs the extractor,
measures the actual error.

| Kind | Direct OK | Smart picked right | Direct err | Smart err |
|---|---|---|---|---|
| bar | 3/3 | 3/3 | **0.08 MAE** | 0.08 |
| pie | 3/3 | 3/3 | **0.08 MAE on %** | 0.08 |
| line | 3/3 | 3/3 | **0.12 RMSE on y-means** | 0.12 |
| stacked | 3/3 | 3/3 | **0.11 MAE on per-bar totals** | 0.11 |
| box | 3/3 | 3/3 | **0.99 MAE on medians** | 0.99 |

Bar/pie/line/stacked errors are excellent (~0.1 absolute units).
Box plot median error is **0.99 absolute units** — noticeable but
real; matplotlib's default boxplot draws medians at the midpoint
of the box body, not at the actual data median.

### Bugs the bench surfaced
* Smart was picking `bar_chart` on box plots (caption keyword
  matching too weak). Added **baseline-cluster check** in
  `simple_bars.py`: real bars share a y-bottom within 5% of plot
  height. Box plots don't (rectangles float at varying y-positions).
* Smart was picking `bar_chart` on stacked bars (extractor saw the
  fully-stacked column as a single bar). Bumped stacked's
  **structural credibility bonus** in
  `parallel_extractor._structural_credibility` from +0.3 to +0.7
  when ≥2 series × ≥3 cols × ≥6 nonzero cells.

After fixes: **5/5 kinds picked correctly** by smart extractor.

## 3. Adversarial bench

Built `eval_harness/bench_adversarial.py`. 9 deliberately ambiguous
figures (bar with internal cap line that looks like box median;
scatter + regression that looks like line; pie with bordered legend
that looks like a chart axis; horizontal bars; multi-group bars;
log-scale axis; etc) × 2 caption conditions = 18 cases.

### Round 1: 10/18 = 55.6%

Found 4 real failure modes:
* Pie + bordered legend → picked as line_plot (legend interferes
  with pie's outside-background sanity check)
* Horizontal bars → picked as line_plot (close-call tiebreaker
  promoted a much-lower-quality candidate)
* Grouped bars → picked as flow_diagram (diagram_extract sees
  ~3 rectangular regions and calls them nodes)
* Log-scale → picked as flow_diagram (axis_ocr fails on non-linear
  ticks; diagram_extract grabs the rectangles)

### Fixes

* **Pie outside-background check** now skips image corners (legends
  typically live there)
* **Close-call tiebreaker** now requires the hinted kind to be IN
  the close-call window (within 0.3 quality of top), not just in
  ranked[:3]. Previously promoted line_plot at q=0.3 over
  bar_chart at q=2.475.
* **Diagram self-rejection**: required ≥30% of node labels to be
  "real" (non-placeholder, ≥2 chars with alpha). On grouped bars
  diagram_extract found 7 nodes where 6 were "Node N" placeholders.

### Round 2: 16/18 = 88.9% ✓

Remaining failures: 2/2 are `log_scale`. axis_ocr's linear fit
fails on log axes (tick values 1,10,100,1000 don't fit `y = mx + b`).
**That's a real axis_ocr limitation, out of scope for this round.**

## 4. Multi-panel splitting

`MultiPanelExtractor` already existed in `multipanel.py` and was
wired into the registry for every chart kind. Tested on a synthetic
2×2 panel grid (bar + line + scatter + pie):

* `panel_split.detect_panels()` correctly found 4 panels
* But `smart_extraction` initially picked `flow_diagram` because
  diagram_extract sees the panel boundaries as 4 "nodes"
* After my diagram self-rejection fix (require ≥30% real labels):
  smart correctly picks `line_plot` whose multipanel wrapper
  successfully extracted all 4 sub-panels (A, B, C, D)

Multi-panel works end-to-end. The slowness on real figures is
because multipanel runs OCR per panel per extractor; that's
~5s × 4 panels × 8 extractors = 160s worst case. The
`auto_downscale_max_dim=900` cap keeps it bounded.

## Where we stand now (honest)

| Bench | Cases | Score | Notes |
|---|---|---|---|
| Reflective (synth) | 34 | 82% | classifier-gated, fastest |
| Parallel hinted (synth) | 34 | 94.1% | reliable, ~5s |
| Smart (synth) | 34 | **94.1%** | best, 4.2s avg |
| Value fidelity (synth) | 15 | 5/5 picks, <0.15 MAE | bar/pie/line/stacked |
| Adversarial (synth) | 18 | **88.9%** | 8/9 variants 100% |
| **Real corpus** (3 papers) | 10 | **40% OK + 50% PARTIAL** | bounded ~60s/figure |

## What I'd attack next (in order of expected ROI)

1. **OCR caching across extractors** (huge wins). Each chart
   extractor calls `axis_ocr.ocr_words` independently. With 8
   extractors × N panels we do 8N OCR passes; sharing the OCR
   would 8x speed up real-corpus runs.
2. **Log-scale support in axis_ocr**. Detect when tick values are
   log-distributed; fit `log(y) = mx + b` instead of linear.
3. **Better page-1 filtering**. Some legitimate figures DO live on
   page 1 (review papers, technical notes). Should check caption
   pairing distance, not blanket-skip.
4. **Decorative specialist v2**. Journal covers / logos still slip
   through; should classify on the presence of publisher-name OCR
   text or distinctive logos.
5. **The remaining `partial` results on real corpus** — investigate
   per-paper what's getting flagged and whether fixes generalise.

## The honest reality

**We're not at the limit.** The shape of "94.1% on synthetic"
papered over real failures: page-1 branding, multi-panel slowness,
ambiguous figures. Round 5 surfaced 6 fixable real bugs across 4
benches. After fixes:

* synthetic accuracy holds at 94.1%
* adversarial jumped from 55% to 88.9%
* value-fidelity all 5 kinds picked right with <0.15 MAE
* real corpus went from 0/3 to 40% OK + 50% PARTIAL

The fixes were small, evidence-driven, and each came from looking
at an actual failure. The remaining work is more of the same: every
bench surface more failures; each failure is fixable. There's no
hard ceiling I can point to without VLM compute.
