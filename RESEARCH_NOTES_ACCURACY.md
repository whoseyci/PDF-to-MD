# Round 2 accuracy push — what the data actually said

User asked: "could we build better specialists for everything?"

Before writing more specialists, I probed real corpus figures and the
data told a very different story than I expected.

## The diagnostic

Pulled real figures from a few corpus PDFs (CAST, Fangliang) and ran
them through the existing Mixture+Reflective pipeline:

| Figure | Mixture pick | Reflective result |
|---|---|---|
| CAST p0 (header banner) | flow_diagram | `no_bars` → give up |
| CAST p5 (workflow) | flow_diagram | `no_bars` even though it's clearly a workflow |
| Fangliang p0 (equation viz) | line_plot | `no_axis` |
| Fangliang p0 (another) | line_plot | `no_axis` |

**Zero figures succeeded.** And not because the classifier was wrong
— the classifier picked sensible kinds for 2/4. The real issues:

1. **No OCR was being passed** to the chart extractors. They need axis
   labels to calibrate, so without OCR they ALL return `no_axis`.
2. **78 spurious Hough circles** detected on the CAST schematic
   (linear arrows confused the circle detector). The pie specialist
   was firing on everything.
3. **`flow_diagram` had no extractor wired** in the reflective runner.
   Diagrams just returned "no_extractor" instead of routing to
   `diagram_extract` which we'd already built.
4. **74% of corpus figures had empty captions** — caption pairing
   (E3) had been built but never plumbed into the existing outputs.

## The fixes

Instead of writing more specialists, fixed each diagnosed issue:

### Fix 1: Auto-OCR (biggest single lever)

```python
# pipeline_v2/vision/chart_extract/reflective_runner.py
if auto_ocr and (ocr_text is None or not ocr_text.strip()):
    import pytesseract
    from PIL import Image
    ocr_text = pytesseract.image_to_string(Image.open(image_path))
```

Without this, chart extractors had no axis labels and all failed with
`no_axis`. After, they actually find axes and compute calibration.

### Fix 2: `has_chart_axes` feature

Universal gate: detect two long perpendicular dark lines via
morphological opening. Used as a **hard prior** for chart specialists
(must have axes) and a **negative prior** for diagram/schematic
specialists (shouldn't have axes).

```python
# Detect via morph open with long kernels:
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.3*w), 1))
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.3*h)))
# A chart must have both, not at the page edges.
```

This single feature stopped the pie specialist from firing on
schematics. Bench: pie went from "fires on bar/scatter/box" to
"only fires on actual pies".

### Fix 3: Diagram routing in reflective_runner

Added `_EXTRACTABLE_DIAGRAM_KINDS = {FLOW_DIAGRAM, SCHEMATIC}` and
made `_try_extract` dispatch those kinds through `diagram_extract`,
wrapping the `DiagramExtractionResult` in a `ChartExtractionResult`
so the reflector loop is uniform.

Before: diagrams returned `no_extractor`.
After: diagrams return `partial` with mermaid graph + node/edge counts.

### Fix 4: Caption backfill

Wrote `pipeline_v2/caption_backfill.py` to re-run caption_pairing on
source PDFs and merge results into already-converted paper.json files
**non-destructively** (only fills empty `caption_text` fields).

Corpus run: **filled 98/339 (28.9%) of empty captions across 23/35
papers.** This in turn lifted the E16 Distillation student-handles
rate from **22.5% → 40.1%** (caption became good enough for the
student rule to accept), saving ~191 min of Gemma 4 teacher time per
full corpus run.

## Measured impact

| Metric | Before round 2 | After round 2 |
|---|---|---|
| Reflective extraction (no caption) | 50% | **78.6%** |
| Distillation student handles | 22.5% | **40.1%** |
| Corpus figures with non-empty caption | 137/476 | **235/476** |
| Reflective + diagrams (synth bench) | "no_extractor" | partial w/ mermaid |
| Tests passing | 123/123 | **133/133** |

## Why I did NOT write more specialists

Looking at the failure cases honestly, the specialists weren't the
problem. The problem was:
* Information not flowing (OCR not passed)
* Features too noisy (Hough circles firing everywhere)
* Wrong dispatcher (no diagram routing)
* Missing data (empty captions)

Each of those is a wiring fix worth a 10-20% accuracy bump. Writing
better specialists would maybe add 1-2% per kind on top — important
later, but the orders of magnitude here are wiring vs feature.

If we want more specialist improvements later, the right next step is:

* **A "is this even a figure" gate** — many "figures" in PDFs are
  decorative borders, logos, page headers (the OECD case where 235/245
  "figures" got no caption is mostly these). A `decorative` specialist
  that fires on small, low-content images would let us skip them
  cleanly instead of trying every chart extractor.
* **Better OCR for legend text** — currently legend specialists are
  caption-keyword driven; a small structured-OCR pass on the legend
  region would give "Series A", "Series B" labels directly.
* **Per-domain specialists** — soil-science charts often have
  treatment names like "BAU", "Biodiv", "Biodiv nest" that the
  generic bar_chart extractor doesn't expect. A dictionary of
  common treatment-name patterns per discipline would tighten the
  category label inference.

But all three of those are 1-3% improvements layered on top. The
wiring fixes were the 30-point jump.
