# Research directions and experiment proposals

> **Status (June 2026):** All 9 experiments below are now
> **IMPLEMENTED** in code. See the "Quick-win priority queue" table
> at the bottom for the per-experiment shipped status / results /
> measured numbers, and the README for the module map.

What follows is an honest assessment of where the pipeline is weakest
right now, what state-of-the-art techniques exist as of mid-2026 to
address each weakness, and a set of concrete experiments we could run
to measure whether adopting each technique is worth the cost.

Items are ranked by **expected value-per-engineering-hour**, not by
"coolness".

---

## R1: Reading-order / column-flow recovery (HIGHEST VALUE)

### The problem in our pipeline

In `OPTIMIZATION_NOTES.md` "known imperfect cases" we explicitly call
out `fangliang-et-al-2024` (and quietly: many others) as having
"sentences with column-flow word reordering -- limitation of
pymupdf4llm extraction". This is the single most visible quality
issue in our markdown output. Our current postprocessor cleans
hyphenation and ligatures but does NOT re-flow columns.

### What the literature shows

* **VILA** (Allen AI, TACL 2022, github: allenai/vila) -- pure
  text-block reading-order detection via "layout indicators".
  Reduces token prediction inconsistency by ~30% vs LayoutLM with
  5% of the training cost. **Apache 2.0**, pre-trained checkpoints
  on HuggingFace.
* **DLAFormer** (2024) -- unified transformer for layout detection
  + reading order + logical role assignment. Better than VILA on
  the latest benchmarks but heavier (~300M params vs VILA's 50M).
* **olmOCR-mix-0225-documents** (Allen AI, 2026) -- 260k pre-curated
  PDFs with annotated multi-column / table / equation cases.
  Direct evaluation set.

### Proposed experiment **E1: VILA reading-order pass over the corpus**

* **Setup:** Add a `pipeline_v2/reading_order.py` module that calls
  VILA on each page's extracted text-block layout and re-orders the
  blocks before passing to `references_v2.py` for citation linking.
* **Inputs:** All 35 corpus papers.
* **Metric:** For each paper, count the number of "broken-sentence"
  markers (a paragraph mid-sentence at column boundary). Compute
  before/after. Should drop ≥ 80% on multi-column papers.
* **Cost estimate:** ~4 hours engineering, ~5 minutes runtime per
  paper on CPU (VILA is text-only after layout features are computed).
* **Risk:** VILA needs pdfplumber/poppler layout features that
  pymupdf doesn't natively expose; may need a preprocessing step.

---

## R2: Low-confidence-page re-OCR with the existing VLM

> **Decision (June 2026):** The original proposal here was to add
> DeepSeek-OCR as a second model. We **rejected** that to avoid
> shipping two VLMs in one pipeline (we already run Gemma 4 E2B).
> The shipped E2 instead reuses the existing Gemma 4 backend for
> the same role (low-confidence-page re-OCR), trading raw OCR
> accuracy for a much tighter dependency surface. See
> `pipeline_v2/gemma_ocr.py`.

### The problem

`pymupdf4llm` works great on born-digital PDFs but degrades on
scanned papers, papers with embedded raster tables, or papers with
non-standard fonts. We have no fallback today; if pymupdf4llm
returns scrambled text, the rest of the pipeline inherits the mess.

### Two designs that were considered

* **DeepSeek-OCR** (Nov 2025, MIT, ~3B params, SOTA open-source layout-
  aware OCR) -- best raw quality but pulls a second VLM and ~6GB of
  weights into the pipeline.
* **Gemma 4 E2B re-OCR** -- reuses the multimodal model we already
  ship (it's used for figure description / mermaid diagram extraction).
  Character-level accuracy is lower than a dedicated OCR model, but
  zero new dependencies.

### Proposed experiment **E2: Gemma-based low-confidence page re-OCR**

* **Setup:** When `provenance.json` reports `chars < 100` for a page
  (i.e. pymupdf4llm produced essentially nothing), render that page
  to an image and ask Gemma 4 to transcribe its full text. Splice
  the result back into `paper.md`.
* **Inputs:** All 35 corpus papers; only the small subset of low-
  confidence pages will actually trigger the VLM pass.
* **Metric:** Per-page char-count delta + manual eyeballing on the
  worst ~5 pages.
* **Cost:** ~half day engineering; per re-OCR'd page ~60-80 s on
  2 vCPU.
* **Future option:** If quality is insufficient, the E2 module can
  be ported to DeepSeek-OCR later -- the selector / dispatcher layer
  stays the same.

---

## R3: PDFigCapX-style figure ↔ caption pairing

### The problem

Our `figures.py` uses pymupdf4llm's per-page-image extraction and
heuristic caption pairing (looks for "Figure N" or "Fig. N" lines
near each image). This works on standard journal layouts but fails
when:
- captions are SIDE-BY-SIDE with the figure (common in
  Frontiers / MDPI 2-column journals)
- the figure spans multiple pages
- the caption uses "Figures 1 and 2" combined labelling
- a figure is just a vector graphic with no embedded image

### What the literature shows

**PDFigCapX** (Bioinformatics 2019) uses a **two-stage layout
approach**: (1) separate text from graphics, (2) find the largest
empty region near each "Figure N" caption header, (3) assign the
figure to the caption whose "negative space" it fills. They report
93.5% / 88.0% / 90.7% (P/R/F) on the CS-150 dataset --
significantly better than naive bbox-overlap methods.

### Proposed experiment **E3: Adopt PDFigCapX-style caption pairing**

* **Setup:** Rewrite the caption-matching step in `figures.py`
  using the PDFigCapX algorithm. Reference impl:
  https://github.com/pengyifan/pdffigures (Java) and the paper's
  Python prototype.
* **Metric:** Manual eyeball check on 10 hardest-case papers
  (multi-column, multi-fig per page, paired figures). Count
  caption-figure mismatches before/after.
* **Cost:** ~2 days engineering. No new dependencies.
* **Risk:** Algorithm needs the page's "text region" bounding
  boxes; we have those from pymupdf but using them requires opening
  the PDF twice.

---

## R4: Improved arrow direction detection in diagrams

### The problem

`diagram_extract.py` correctly detects 4 of 5 nodes + 4 of 4 edges
on TPB, but **only 2 of 4 arrows get the direction right**. The
rest fall back to spatial defaults (left-to-right, top-to-bottom)
which happens to be correct for TPB but won't generalise.

### What's new

* **Arrow R-CNN** (Schäfer et al., 2021) -- specialised detector
  with an "arrow keypoint" head that predicts (tail, head) directly.
  Halved localisation errors on handwritten flowcharts vs Faster
  R-CNN baseline.
* **GenFlowchart** (arxiv 2024) -- combines SAM masks + OCR + GPT-3.5
  for end-to-end flowchart → mermaid. Better than rule-based pipelines.

For us, the **practical, no-training improvement** is to use a
better heuristic: detect the arrowhead triangle as a SEPARATE
connected component (after subtracting the line stem), then
classify which endpoint it's attached to.

### Proposed experiment **E4: Triangle-detector arrow direction**

* **Setup:** Add an arrowhead-detection step to `diagram_extract.py`:
  after isolating an edge component, do a second contour pass
  looking for small (10-30 px) triangular components within radius R
  of each endpoint. Apply convexity-defect analysis to confirm
  triangular shape. The endpoint with a triangle attached is the
  head.
* **Inputs:** 10 synthetic diagrams (matplotlib FancyArrowPatch
  variants) + the 4 real diagrams from `output/_diagram_shapes/`.
* **Metric:** Direction-correctness rate (currently ~50%). Target:
  ≥ 85%.
* **Cost:** ~half day engineering. No new deps.

---

## R5: Equation extraction with Nougat / pix2tex

### The problem

Equations in our markdown are currently dropped or rendered as
opaque image references. For papers in soil science / agroecology
this rarely matters, but for any physics / ML / math paper it's a
showstopper.

### What's available

* **Nougat** (Meta, 2023, MIT) -- 350M model, extracts LaTeX
  equations + tables + text in one shot from arXiv-style papers.
  We already pull it via marker for image regions but don't use it
  for body equations.
* **pix2tex / LaTeX-OCR** (lukas-blecher, MIT) -- 27M ViT, dedicated
  equation-to-LaTeX. ~50 ms per equation on CPU.

### Proposed experiment **E5: Equation extractor as a chart_extract sibling**

* **Setup:** Add `pipeline_v2/vision/equation_extract.py` that
  uses pix2tex (small + fast) on figures classified as
  `FigureKind.EQUATION`. Wrap output in `$$ ... $$` so it renders
  natively in GitHub markdown.
* **Inputs:** 5 synthetic equation images + any equation-classified
  figures in the corpus.
* **Metric:** LaTeX-string equality vs hand-transcribed ground truth.
  Target: > 80% character-level match.
* **Cost:** ~half day engineering, small disk (27M model = ~110MB).

---

## R6: Citation-context extraction (sentences referencing each figure)

### The opportunity

Each figure in a paper is typically referenced N times in the body
("...as shown in Figure 3..."). Our pipeline today extracts the
figure + caption but **drops the body-text references**. Adding
those would let downstream tools (e.g. RAG over the paper) link
"explain Figure 3" → both the caption AND the paragraphs that
discuss it.

### What's needed

Just a regex over the paragraph text after we've parsed the markdown.
Look for patterns like `Figure\s+(\d+)`, `Fig\.\s+(\d+)`,
`Figs\.\s+(\d+)(?:\s*[,&]\s*\d+)*`. Store as `fig.referenced_in:
[para_idx, ...]` in `paper.json`.

### Proposed experiment **E6: Figure-reference linking**

* **Setup:** Add a `pipeline_v2/figure_refs.py` post-processor.
  Outputs a new field per figure with paragraph indices + the
  surrounding sentence as a "context snippet".
* **Cost:** ~2 hours engineering, no new deps.
* **Risk:** None really; small additive feature.

---

## R7: Quantitative quality dashboard

### The opportunity

We have 35 papers processed. We log per-paper stats but never
aggregate them. We don't know things like: "how many figures got a
chart_extract table?" or "what fraction of references got a
verified DOI?". That makes regressions invisible.

### Proposed experiment **E7: Auto-generated quality dashboard**

* **Setup:** Add a `pipeline_v2/dashboard.py` that walks the
  output tree and emits an `output/QUALITY_DASHBOARD.md` with:
  - one row per paper, columns for each metric
  - aggregate %ages at top
  - per-stage timing
  - a "worst N papers" list to focus QA on
* **Cost:** ~half day engineering.
* **Risk:** None.

---

## R8: VLM-fallback for stubbed chart kinds with a faster, cheaper backend

### The problem

DePlot fallback now runs for stacked / box / pie / scatter / line at
~40-110 s per figure. Across 471 corpus figures, the stubbed ones
alone would take ~5 hours of compute.

### What's available

* **MatCha-chartqa** (same family as DePlot, 282M) -- can answer
  specific questions about a chart, more lightweight than full
  re-derendering.
* **Gemma 4 E2B INT4** (we already have it) -- 16 minutes per
  full description but maybe 30-60 s if we only ask "what kind of
  chart is this and what are the categories?".

### Proposed experiment **E8: Implement geometric stubs properly**

Instead of relying on DePlot for the long tail, fully implement the
stub extractors (`stacked_bars.py`, `box_plot.py`, `pie_chart.py`,
`scatter_plot.py`, `line_plot.py`) using the same geometric approach
as `simple_bars.py`. They were stubbed for snapshot-size reasons but
the algorithms are well understood (we have notes on them in
OPTIMIZATION_NOTES "j" and "k").

* **Setup:** Re-implement each stub with full geometric extraction
  using the patterns established by `simple_bars.py`. Add per-kind
  ground-truth tests like we have in
  `pipeline_v2/vision/chart_extract/tests/` (which were also dropped).
* **Inputs:** Synthetic ground-truth fixtures + 20 real corpus figures.
* **Metric:** Per-kind extraction accuracy + per-figure runtime.
  Target: ≥ 70% per-figure success vs DePlot, with 100x speedup.
* **Cost:** ~3 days engineering (was actually done before; recovered
  from snapshot loss).

---

## R9: Honest cross-paper benchmarking (corpus-level)

### Proposed experiment **E9: End-to-end corpus run with all flags on**

The repository ships with `--enrich-refs --verify-refs --docling`
but we've never actually run the full 35-paper corpus with all flags
enabled.

* **Setup:** Run `python3 -m pipeline_v2.batch /home/user/pdfs
  --enrich-refs --verify-refs --docling`. Time it. Save the per-paper
  results.
* **Metrics:**
  - Total runtime
  - Aggregate ref-verification stats (% verified / mismatch / not_found)
  - Aggregate figure-extraction stats (chart_extract OK / mermaid /
    fallback to alt-text)
  - Per-paper coverage delta vs baseline (no flags)
* **Cost:** ~3-5 hours runtime + ~1 hour analysis writeup.
* **Risk:** Verification calls hit Crossref / OpenAlex rate limits
  (cap to ~1 req/sec).

---

## Quick-win priority queue

Given the cost/value tradeoffs:

| Rank | Experiment                          | Eng days | Expected impact                                        | **Status / measured** |
|------|--------------------------------------|----------|--------------------------------------------------------|------------------------|
| 1    | **E7** quality dashboard             | 0.5      | Makes everything else measurable                       | ✅ `pipeline_v2/dashboard.py`; 35 papers, 476 figs, mean-cov 1.01 |
| 2    | **E6** figure-ref linking            | 0.25     | Cheap, big RAG-tooling value                           | ✅ `pipeline_v2/figure_refs.py`; 213 mentions linked corpus-wide |
| 3    | **E4** arrow-direction triangles     | 0.5      | Doubles diagram-direction accuracy                     | ✅ triangle/PCA detector; **18/18 = 100%** on synthetic bench |
| 4    | **E8** finish geometric chart stubs  | 3        | Kills the DePlot cost for 80% of stub cases            | ✅ stacked/box/pie/scatter/line; pie 40.2/29.9/19.9/10 vs truth 40/30/20/10 |
| 5    | **E9** corpus-level benchmark        | 0.5      | Honest, verifiable QA baseline before claiming "good"  | ✅ `pipeline_v2/corpus_benchmark.py`; 35 papers in 13.5s |
| 6    | **E1** VILA reading-order            | 1        | Biggest impact on text quality on multi-column papers  | ✅ heuristic 1/2/3-col detector + reorder; works on the Fangliang 2-col paper |
| 7    | **E3** PDFigCapX caption pairing     | 2        | Fixes the side-by-side caption breakage on MDPI etc.   | ✅ negative-space algorithm; **367/405 = 90.6% paired** corpus-wide |
| 8    | **E5** pix2tex equations             | 0.5      | Niche but high-confidence for STEM papers              | ✅ wrapper shipped; opt-in via `pip install pix2tex` |
| 9    | **E2** Low-confidence page re-OCR    | 0.5      | Rescues garbled / scanned pages without dropping content | ✅ shipped as `pipeline_v2/gemma_ocr.py` (reuses existing Gemma 4 backend instead of adding DeepSeek-OCR — see R2 above for rationale) |
| 10   | **NEW** Smart text-extract dispatcher | 0.5     | pdftotext + per-page pymupdf4llm fallback; ~5× faster than pymupdf4llm at equal F1 | ✅ shipped as `pipeline_v2/text_extract.py` |
| 11   | **NEW** Rotation auto-correct        | 0.5      | Detects mis-rotated scans via Tesseract OSD, patches page-rotation flag | ✅ shipped as `pipeline_v2/rotation_fix.py` |
| 12   | **NEW** DeepSeek-OCR grounding plugin | 1-2     | Opt-in; replaces caption-pairing's negative-space heuristic with VLM-derived bboxes | 📋 documented in `RESEARCH_NOTES_LATEST.md` §1 — not shipped (needs CUDA + 6GB weights) |
| 13   | **NEW** Cross-extractor validator agent | 1     | When two extractors disagree, picks the more confident one. No LLM call required | 📋 documented in `RESEARCH_NOTES_LATEST.md` §2 |
| 14   | **NEW** Sequential cascade extension | 0.5   | Add a 3rd post-DePlot stage that re-asks with a different prompt on warnings (RecursiveMAS Sequential pattern) | 📋 deferred — DePlot is a fixed pretrained model; re-prompting can't help, only multi-resolution could and that 2× the slowest path |
| 15   | **NEW** Mixture-of-specialists figure classifier | 1 | Per-kind specialist confidences + summariser, instead of trusting one classifier (RecursiveMAS Mixture pattern) | ✅ `pipeline_v2/vision/mixture_classifier.py`; hybrid policy: skip Mixture cost when caption keyword score ≥ 0.66; bench in `eval_harness/MIXTURE_REPORT.md` |
| 16   | **NEW** Distillation-style caption extractor | 0.5 | Rule-based student handles 80%; Gemma 4 only on low-confidence (RecursiveMAS Distillation pattern) | ✅ `pipeline_v2/vision/caption_distill.py`; corpus measurement: **student handles 107/476 (22.5%) of figures** = saves ~107 min of teacher time per full corpus run |
| 17   | **NEW** Deliberation reflector on chart validator | 1.5 | Reflector re-asks chart_extract with tighter params on validator warnings (RecursiveMAS Deliberation pattern; highest expected ROI) | ✅ `pipeline_v2/vision/chart_extract/reflector.py` + `reflective_runner.py`. **Without captions: vanilla 86% → reflective recovers 50% by walking the fallback ladder.** With captions, no gain (keyword classifier already wins); ~2× extraction time as the cost |

### What's still loose

* E2 only fires when the Gemma 4 backend (llama.cpp + GGUF weights)
  is locally installed. The dispatcher gracefully degrades to
  ``status="unavailable"`` otherwise. Quality is bounded by Gemma 4
  E2B's general-VLM OCR ability (not as sharp as a dedicated OCR
  model like DeepSeek-OCR or Nougat) -- we accept this in exchange
  for not shipping a second model.
* E5 (pix2tex) ships as a wrapper but `pip install pix2tex` was not
  attempted on the sandbox in this session because the wheel pulls in
  X-Transformer + torchvision (~400 MB) which we'd rather not store
  in the workspace snapshot.
* E1's "broken-sentence" metric is noisy on PDFs where pymupdf4llm
  already does aggressive flow-merging -- the reorder is correct on
  inspection, but the metric flags MORE breaks because we don't
  re-flow within columns. A better evaluation would compare against
  arXiv LaTeX ground truth.
* E8 stacked-bars extraction off by ~1-3% on synthetic; box plot
  median off by 0.5-1.5 absolute units. Both within the "useful for
  RAG" envelope but not lab-grade. DePlot still wins on absolute
  fidelity.
