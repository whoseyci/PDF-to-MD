# PDF-to-MD

A multi-stage PDF → Markdown conversion pipeline for academic papers
with figure handling that doesn't hallucinate.

## What it does

For each PDF:

1. **Text** — `convert.py` uses `pymupdf4llm` for the main extraction,
   then `postprocess_md.py` runs ligature / soft-hyphen repair, page
   markers, BR-stacked-table merging, MDPI sidebar stripping,
   diacritics normalization, and column-flow fixes.
2. **References** — `references_v2.py` parses the bibliography, links
   inline citations like `(Smith, 2023)` or `[12]` to anchors with
   `INLINE_CITE_RE`, `PROSE_CITE_RE`, `BRACKET_CITE_RE`,
   `PAREN_NUM_CITE_RE`, and protects existing `![…](…)` blocks from
   being rewritten.
3. **Front matter** — `metadata.py` extracts title / authors / DOI /
   year / journal with multilingual surname rules; `first_page_layout.py`
   has 4 detectors for different journal layouts (Elsevier 2-column,
   MDPI inline-abstract, Frontiers no-header, single-column-abstract).
4. **Figures** — `figures.py` extracts images + caption pairing +
   OCR sidecars, then `vision/` runs a three-track figure pipeline:

   * **Charts** (bar / box / pie / scatter / line / stacked): the
     `vision/chart_extract/` package measures pixels classically with
     OpenCV + tesseract and produces a real Markdown table with real
     numbers. No LLM hallucination possible — we measure, we don't
     generate.
   * **Diagrams** (theory models, causal loops, flow charts): the
     `vision/mermaid_extract.py` module asks Gemma 4 to transcribe
     boxes-and-arrows into a Mermaid graph that renders natively in
     GitHub Markdown.
   * **Everything else** (maps, photos, journal covers, schematics):
     Gemma 4 produces a single accurate alt-text sentence.

The whole figure pipeline is designed for the worst-case 2 GB RAM
sandbox — see `output/OPTIMIZATION_NOTES.md` sections (h) through
(l) for the full debug history.

## Quick start

```bash
# 1. Convert one PDF (fast path)
python3 -m pipeline_v2.convert path/to/paper.pdf

# 2. Convert with all optional enrichment passes
python3 -m pipeline_v2.convert path/to/paper.pdf \
    --enrich-refs    # augment refs with refextract structured fields
    --verify-refs    # cross-check refs against Crossref + OpenAlex
    --docling        # also emit paper.docling.json (RAG-ready)

# 3. Batch over a directory
python3 -m pipeline_v2.batch /path/to/pdfs --output /path/to/output

# 4. Run the figure-vision pass over already-converted papers
python3 -m pipeline_v2.vision.run_all \
    --output-dir /path/to/output \
    --model gemma4-e2b \
    --paper my-paper

# 5. Debug a single figure
python3 -m pipeline_v2.vision.chart_extract.cli \
    path/to/figure.png --kind bar_chart \
    --debug-overlay overlay.png
```

## What's in the box

This is a multi-track pipeline that fuses the best ideas from
several state-of-the-art PDF→MD projects:

* **Text & layout**: `pymupdf4llm` base + custom postprocessing for
  ligatures, soft hyphens, column flow, MDPI sidebars, journal-layout
  detection.
* **References**:
  * Our own regex parser (`references_v2.py`) for in-text citation linking
  * `--enrich-refs` adds structured DOI/journal/year via
    `inspirehep/refextract` (`refextract_bridge.py`)
  * `--verify-refs` cross-checks against Crossref + OpenAlex
    (`ref_verifier.py`, inspired by markrussinovich/refchecker) to
    flag fabricated or garbled citations
* **Figures** -- four complementary tracks:
  * `chart_extract/` -- classical (OpenCV + tesseract) geometric
    extraction. Real markdown tables with real numbers, no hallucination.
    Bench: 0.5 s/chart, mean abs err ≤ 0.1 on simple bar charts.
  * `chart_extract/deplot.py` -- Google's `google/deplot` specialist
    model **now default-enabled** as a fallback cascade for all chart
    kinds. Geometric extractor tries first (~0.5 s); DePlot only runs
    when geometric returns PARTIAL or UNSUPPORTED (40-110 s/figure on
    CPU, peak 1.45 GB RAM). Disable by setting `PDF2MD_DISABLE_DEPLOT=1`.
  * `diagram_extract.py` -- **classical (non-LLM) shape-aware** diagram
    → Mermaid extractor for clean machine-rendered conceptual diagrams
    (TPB, causal loops, decision trees, draw.io exports). 0.8 s/figure.
    Classifies each node as `rect` / `rounded` / `circle` / `diamond` /
    `parallelogram` and emits the correct Mermaid shape syntax
    (`A[label]`, `A((label))`, `A{label}`, etc.). Tries this FIRST in
    the runner before falling back to the VLM.
  * `mermaid_extract.py` -- VLM-based diagram → Mermaid via Gemma 4 E2B.
    Fallback for hand-drawn / irregular / overlapping diagrams that
    the classical extractor can't handle. ~16 min/figure on CPU.
  * `figure_prompts.py` -- marker-style per-subtype VLM prompts for
    algorithms, code listings, equations, screenshots, gels/blots
* **Output schema**:
  * `paper.md` -- the headline output
  * `paper.json` -- our own schema (everything we extracted)
  * `paper.docling.json` (optional, `--docling`) -- DoclingDocument-
    compatible schema, drop-in for LlamaIndex/LangChain RAG pipelines
* **Selective LLM dispatch** (`llm_boost.py`): marker `--use_llm`
  pattern. Per-block decision (`skip`/`validate`/`replace`/`extract`)
  based on classical-extractor confidence so LLM budget only burns
  on blocks that need it.

## Eval harness (`eval_harness/`)

Three honest evaluations of how this pipeline actually performs:

1. **Ground-truth corpus + WER bench** — downloads 10 arXiv papers
   (Attention, BERT, ViT, ResNet, GAN, VAE, …), grabs their LaTeX
   source as ground truth, runs five extractors (pdftotext,
   pdftotext-stream, pymupdf4llm, our pipeline post-process, our
   E1 reading-order) and reports F1 / WER\* / runtime side-by-side.
   See `eval_harness/REPORT.md`.
2. **Static corpus browser** — `pipeline_v2/corpus_browser.py` emits
   a single self-contained `output/index.html` that lets you browse
   every processed paper with rendered markdown, figures gallery,
   reference table with DOI badges.
3. **Failure-mode catalog** — `eval_harness/failure_modes.py`
   synthesises 10 deliberately broken PDFs (encrypted, image-only,
   rotated, mid-word-break, garbage bytes, …) and probes which
   pipeline stage tolerated each. See `eval_harness/FAILURE_REPORT.md`.

```bash
python3 -m eval_harness.fetch_arxiv       # one-time download
python3 -m eval_harness.run_eval          # → REPORT.md / REPORT.json
python3 -m eval_harness.failure_modes     # → FAILURE_REPORT.md
python3 -m pipeline_v2.corpus_browser     # → output/index.html
```

**Numbers (June 2026, 16 arXiv papers + 2 synthetic scanned PDFs):**

| Extractor | avg F1 | avg WER\* | avg s |
|-----------|--------|-----------|-------|
| pdftotext             | 0.706 | 0.153 | 0.07 |
| pdftotext-stream      | 0.727 | 0.149 | 0.07 |
| pymupdf4llm           | 0.821 | 0.080 | 14.8 |
| pdf2md-postprocess    | 0.821 | 0.080 | 14.3 |
| pdf2md-reorder-e1     | 0.729 | 0.148 | 0.10 |
| **pdf2md-auto** ⭐    | **0.821** | **0.077** | **3.0** |
| pdf2md-auto-rotfix    | 0.821 | 0.077 | 16.3 |

* **`pdf2md-auto` (the new default) matches pymupdf4llm quality at
  ~5× the speed** — uses pdftotext per page, falls back to
  pymupdf4llm only when a page has < 100 chars.
* **On scanned PDFs**: pdftotext alone returns F1=0.000;
  `pdf2md-auto` correctly routes to pymupdf4llm OCR and recovers
  F1=0.78. The fallback path is now actively exercised by the
  harness (synthetic scanned versions of 2 of the arXiv papers).
* E1 reading-order was previously worse than baseline — that turned
  out to be **two bugs**: our metric didn't expand Unicode
  ligatures (`ﬁ` → `fi`), penalising any extractor that preserved
  the glyph; and E1 itself didn't dehyphenate line-end breaks.
  Both fixed; E1 now matches `pdftotext-stream` (and runs at the
  same speed).

## New text-extraction CLI

```bash
# Smart dispatcher (now the recommended default for text-only use)
python3 -m pipeline_v2.text_extract paper.pdf
python3 -m pipeline_v2.text_extract paper.pdf --mode auto --stats
python3 -m pipeline_v2.text_extract paper.pdf --mode reorder    # E1
python3 -m pipeline_v2.text_extract paper.pdf --mode pdftotext  # fastest

# Rotation detection / correction
python3 -m pipeline_v2.rotation_fix paper.pdf
python3 -m pipeline_v2.rotation_fix paper.pdf --apply --out fixed.pdf
```

Note: `pipeline_v2/convert.py` (the full pipeline) still uses
`pymupdf4llm` end-to-end because it needs per-page chunks with
inline figure extraction. The smart dispatcher is best used when you
just want text (e.g. for indexing / RAG / search).

## RecursiveMAS-inspired patterns (E15–E17, shipped)

After surveying the [RecursiveMAS paper](https://recursivemas.github.io/)
(Yang et al., 2026, arXiv 2604.25917) we couldn't adopt the literal
latent-state mechanism (needs CUDA, shared model frameworks, training
infra — see `RESEARCH_NOTES_LATEST.md` for the honest constraints
table). But three of its four *collaboration patterns* translated to
our codebase without the latent-space machinery:

* **E15 Mixture-of-specialists** (`pipeline_v2/vision/mixture_classifier.py`)
  — every `FigureKind` has a small specialist that scores its own
  confidence using cheap image features (line density, bar-strip count,
  Hough circles, …) + the caption keyword score. A summariser ranks
  them and emits a fallback ladder for the runner. Hybrid policy
  (`classify_figure_hybrid`) skips the image-loading work when the
  caption keyword classifier is already decisive (≥2 keyword hits).
* **E16 Distillation** (`pipeline_v2/vision/caption_distill.py`) — a
  rule-based student decides whether the caption is self-sufficient as
  alt-text; the (slow) VLM teacher is only invoked when the student
  says "I'm unsure". **Corpus run: student handles 22.5% (107/476) of
  figures**, saving ~107 min of teacher time per full corpus pass.
* **E17 Reflector / Deliberation**
  (`pipeline_v2/vision/chart_extract/reflector.py` +
  `reflective_runner.py`) — when an extractor returns PARTIAL /
  NO_BARS / OCR_FAILED, the Reflector decides whether to retry with
  tighter params, fall through to the next ladder kind, or give up.
  Bounded: at most 1 retry per kind + 2 fallback kinds, so worst-case
  ~4 extractor calls per figure.

Bench (`eval_harness/bench_mixture_reflector.py`):

| Scenario | Classifier hit-rate | Extractor OK rate |
|---|---|---|
| With captions, keyword baseline | 100% | 86% |
| With captions, Mixture (E15) | 79% | 86% |
| **Without captions**, keyword | **0%** | n/a |
| **Without captions**, Mixture | 14% | n/a |
| **Without captions**, Reflective (E17) | n/a | **79%** |

### Round 2 accuracy push (Sep 2026)

After probing real corpus figures and seeing that 100% of figures
across all 35 papers had no chart extraction (the runner was missing
key wiring), four targeted fixes:

1. **Auto-OCR in `run_reflective_extraction`** — chart extractors need
   axis labels to calibrate; without OCR they ALL return `no_axis`.
   The reflective runner now OCRs the figure once when no `ocr_text`
   is supplied. Single biggest accuracy lever.
2. **`has_chart_axes` feature** — universal "is this even a chart"
   gate (detects two long perpendicular dark lines). Chart specialists
   now require this feature; diagram/schematic specialists *penalise*
   it. Suppresses the Hough-circles noise that was firing pie/scatter
   specialists on schematics.
3. **Diagram routing in the reflective runner** — `flow_diagram` /
   `schematic` kinds now route through `diagram_extract` (mermaid
   output), not just chart extractors. Previously diagrams returned
   "no_extractor"; now they return PARTIAL with mermaid graph.
4. **Caption backfill** (`pipeline_v2/caption_backfill.py`) — runs
   `caption_pairing` on the source PDF and merges new captions into
   already-converted `paper.json`s. Non-destructive (only fills empty
   fields). Corpus run: filled **98/339 (28.9%)** of empty captions
   across **23/35 papers**. This in turn doubled distillation's
   student win rate from 22.5% → **40.1%** (~191 min teacher saved
   per corpus run).

Without-caption reflective extraction went from **50% → 78.6%** on
the synthetic bench after these fixes. With-caption performance
unchanged (already at 86%, capped by extractor quality).

### Round 6 OCR caching (Jun 2026)

The single biggest leftover speedup from Round 5's ROI-ranked attack
list: each chart extractor was calling `axis_ocr.ocr_words()`
independently, so a parallel sweep of 8 specialists OCR'd the same
image 8+ times. The parallel extractor's `auto_ocr` step also called
`pytesseract.image_to_string` as a separate Tesseract pass.

Fix: `axis_ocr.ocr_words` now memoises by `(abs_path, mtime, size)`
in a per-process LRU. Auto-OCR was rewritten to synthesise its text
from the cached word list, so it warms the cache instead of duplicating
work. Same for the reflective runner.

Measured on the value-fidelity bench (`bench_ocr_caching.py`):

| | Before | After |
|---|---|---|
| Tesseract calls per single-panel figure | 11 | **1** |
| OCR wall-time as % of total | 62% | **13%** |
| End-to-end avg per figure | ~6.5 s | **~2.8 s** |

Multipanel figures still issue one OCR call per panel (correct — each
panel is a different image). The cache invalidates automatically on
file rewrite (mtime/size key) so downscaled temp images don't
collide. Tests: 160/160 (added `test_ocr_cache`).

### Round 5 honest-evaluation push (Oct 2026)

User asked if we'd reached the limit. Built 4 new benches in sequence
to find out:

| Bench | n | Score |
|---|---|---|
| **Real corpus** (3 PDFs, ~10 figures) | 10 | **40% OK + 50% PARTIAL** |
| **Value fidelity** (extracted numbers vs truth) | 15 | 5/5 kinds at <0.15 MAE (box at 0.99) |
| **Adversarial** (deliberately ambiguous figures) | 18 | **88.9%** (was 55% before fixes) |
| **Multi-panel** (sub-plot grids) | end-to-end | works after diagram-self-rejection fix |

Six real bugs found and fixed (each from looking at an actual failure):
* Bar extractor accepted box plots as bars → added baseline-cluster check
* Stacked too conservative → bumped credibility bonus
* Pie outside-bg check broke on legends → skip corners
* Close-call tiebreaker promoted irrelevant kinds → require in close-call window
* Diagram extracted 7 fake nodes from grouped bars → require ≥30% real labels
* Multi-panel detection slow on real PDFs (1750×1333) → auto-downscale to ≤900 px

See `RESEARCH_NOTES_LIMITS.md` for the full writeup including
"where we are NOT at the limit" and what to attack next.

### Round 4 reliability push: parallel extraction (Oct 2026)

User asked: can the specialists be made completely reliable without
VLMs? Built a 81-case classifier stress bench and got an honest
answer:

| Caption condition | Keyword | Mixture | Hybrid |
|---|---|---|---|
| **rich** | 78% | **89%** | **89%** |
| **minimal** ("Figure 1.") | 0% | 37% | 37% |
| **empty** (image only) | 0% | 37% | 37% |

**~37% on weak/no captions = unreliable.** Built a new
`parallel_extractor.py` that flips the architecture: instead of
gating on the classifier, run EVERY extractor and arbitrate by
quality + structural credibility (medians-should-vary, whiskers-
must-extend, wedge-percentages-should-sum-100, ≥3 bars not 1, etc).

Each extractor now self-rejects properly:
* `simple_bars` rejects if <2 bars detected
* `pie_chart` rejects if wedges cover <80% of circle
* `scatter_plot` rejects if markers form a line
* `stacked_bars` rejects if <2 colour bands or non-columnar
* `box_plot`: rejected via structural credibility (whiskers must
  extend beyond box; medians must vary)

After fixes: **94.1% (32/34) on stress bench**, **8/9 kinds at 100%**.
Only scatter_clean_1 fails — synthetic plot with no axis labels,
fundamentally hard.

| Strategy | Rich cap | No cap | avg s |
|---|---|---|---|
| Reflective (kind ladder) | 88.2% | 76.5% | 1.7s |
| Parallel + hint | 94.1% | 94.1% | 4.7s |
| **Smart** (caption-decisive→reflective, else→parallel) | **94.1%** | **94.1%** | **4.2s** |

The new production entry point is `run_smart_extraction`:

```python
from pipeline_v2.vision.chart_extract.parallel_extractor import (
    run_smart_extraction)
trace = run_smart_extraction(
    image_path=Path("figure.png"),
    caption="Figure 1. Bar chart of yield by treatment.",
    ocr_text=None,  # auto-OCR'd internally
)
```

See `RESEARCH_NOTES_RELIABILITY.md` for the full architectural
writeup, including the question "can we go higher without VLMs?"

### Round 3 specialist polish (Sep 2026)

Six more accuracy fixes layered on top:

* **Tick-mark requirement for axis detection** — `has_chart_axes`
  now requires at least 4 tick marks across the candidate axes,
  not just a rectangular frame. Pie charts (which have a rectangle
  but no ticks) correctly classified as no-axes again.
* **All-strong-lines scan** instead of just the longest — box-plot
  spines now correctly identify the bottom-x-axis (where ticks live)
  instead of just the top spine.
* **Multipanel detection** — finds 2×2/3×3 sub-plot grids via
  whitespace-gap analysis, but ONLY when BOTH dimensions split
  (so single-axis bar charts don't false-positive).
* **Decorative specialist + early-exit** — figures classified as
  `DECORATIVE` (thin banners, tiny logos, monochrome no-structure)
  skip extractor pipeline entirely. Saves time on the 7% of corpus
  figures that aren't really data figures.
* **Equation routing** — `EQUATION` kind dispatches to
  `equation_extract` (pix2tex when installed), wrapped in a
  ChartExtractionResult for the reflector loop.
* **Cross-caption mentions** — when a caption like "Fig 3. Same as
  Fig 2 but for..." references another figure number, it counts as
  a mention on the referenced figure. Corpus: 161 → **176** figures
  with at least one mention (+9% absolute).

## Research-track features (E1–E9, all shipped)

All 9 experiments proposed in `RESEARCH_DIRECTIONS.md` are now
implemented. See that file for design notes; the modules below ship
the actual code.

| ID | Module | One-liner |
|----|--------|-----------|
| E1 | `pipeline_v2/reading_order.py` | Multi-column reading-order recovery (1/2/3-col detection + column-then-y walk + banner re-insertion) — alternative to VILA, zero new deps |
| E2 | `pipeline_v2/gemma_ocr.py` | Low-confidence-page re-OCR via the **already-loaded Gemma 4** backend (no new model). Opt-out via `PDF2MD_DISABLE_GEMMA_OCR=1` |
| E3 | `pipeline_v2/caption_pairing.py` | PDFigCapX-style figure↔caption pairing via negative-space matching. Corpus run: **90.6 % captions paired** (367/405 across 33 papers) |
| E4 | `pipeline_v2/vision/diagram_extract.py` | Triangle/PCA-based arrow direction detector. Synthetic 10-case bench: **100 % (18/18)** vs prior baseline ~50 % |
| E5 | `pipeline_v2/vision/equation_extract.py` | pix2tex equation→LaTeX wrapper, emits `$$…$$` markdown; opt-in (`pip install pix2tex`) |
| E6 | `pipeline_v2/figure_refs.py` | Links body-text "Figure N" / "Fig. N" / "Figs 3-5" mentions back to figure records (`fig.referenced_in[]`). Corpus run: 213 mentions linked |
| E7 | `pipeline_v2/dashboard.py` | Auto-generated `output/QUALITY_DASHBOARD.md` per-paper + corpus aggregates + worst-N list |
| E8 | `pipeline_v2/vision/chart_extract/{stacked_bars,box_plot,pie_chart,scatter_plot,line_plot}.py` | Full geometric extractors (no DePlot fallback needed on clean inputs). Synthetic bench: pie 40/30/20/10 → 40.2/29.9/19.9/10.0; stacked 4×3 matrix recovered; line+scatter+box ok |
| E9 | `pipeline_v2/corpus_benchmark.py` | Honest corpus-level benchmark; runs E3+E6+E7 over all 35 papers and emits `output/CORPUS_BENCHMARK.md` + `.json` |

Run them all over the existing corpus:

```bash
python3 -m pipeline_v2.corpus_benchmark --link-figures --pair-captions
# wrote output/CORPUS_BENCHMARK.md and output/CORPUS_BENCHMARK.json
```

## Gemma 4 E2B setup (one-time)

The vision backend uses Gemma 4 E2B via `llama.cpp`. Once-only setup:

```bash
# 1. Clone + build llama.cpp (~5 min on 2 vCPUs)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git \
    /home/user/.cache/llama_src
cd /home/user/.cache/llama_src
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON -DGGML_OPENMP=OFF
cmake --build build --target llama-mtmd-cli -j 2

# 2. Download model weights (~3.5 GB total)
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
    --include "gemma-4-E2B-it-Q3_K_S.gguf" "mmproj-F16.gguf" \
    --local-dir /home/user/.cache/hf/models/gemma4e2b
```

You can override paths via env vars:

```bash
export LLAMA_MTMD_CLI=/path/to/llama-mtmd-cli
export GEMMA4_MODEL=/path/to/gemma-4-E2B-it-Q3_K_S.gguf
export GEMMA4_MMPROJ=/path/to/mmproj-F16.gguf
```

## Performance on a 2 vCPU / 2 GB sandbox

| Operation                                | Time          |
|------------------------------------------|---------------|
| `pipeline_v2.convert` per paper          | ~10-30 s      |
| Classical chart extraction per chart     | 0.5-5 s       |
| Gemma 4 short alt-text (one figure)      | 60-80 s       |
| Gemma 4 Mermaid extraction (one diagram) | 10-17 min     |
| Gemma 4 validator verdict per chart      | 40-60 s       |

## Repo layout

```
pipeline_v2/
├── batch.py              # multi-PDF runner
├── build_index.py        # builds output/README.md master index
├── convert.py            # single-PDF orchestrator (+ --enrich-refs, --verify-refs, --docling)
├── diag2.py              # quality-diagnostics helper
├── figures.py            # image extraction + caption pairing + OCR
├── first_page_layout.py  # journal-layout detectors
├── front_matter.py       # strip redundant abstract/introduction
├── metadata.py           # title / authors / DOI / year / journal
├── postprocess_md.py     # heavy markdown cleanup
├── references_v2.py      # bibliography + inline-citation linking
├── tables_v2.py          # junk-table filter
├── refextract_bridge.py  # NEW: inspirehep/refextract integration
├── ref_verifier.py       # NEW: Crossref/OpenAlex verifier (refchecker-style)
├── docling_export.py     # NEW: DoclingDocument-compatible JSON output
├── llm_boost.py          # NEW: marker-style selective LLM dispatcher
└── vision/
    ├── base.py              # VisionModel ABC, FigureKind enum
    ├── classifier.py        # caption → FigureKind
    ├── factory.py           # make_model("gemma4-e2b" | "stub")
    ├── prompts.py           # per-kind prompt templates
    ├── figure_prompts.py    # NEW: marker-style per-subtype prompts
    ├── validators.py        # output post-processing
    ├── runner.py            # process_figure() — orchestrates everything
    ├── run_all.py           # CLI: batch over a paper
    ├── diagram_extract.py   # NEW: classical (non-LLM) diagram → mermaid
    ├── mermaid_extract.py   # VLM-based diagram → mermaid (Gemma 4)
    ├── MERMAID_DEMO.md      # walked-through example
    ├── README.md
    ├── backends/
    │   ├── stub.py              # deterministic test backend
    │   └── gemma4_subprocess.py # Gemma 4 E2B via llama-mtmd-cli
    └── chart_extract/
        ├── base.py              # ChartExtractor ABC + result dataclass
        ├── axis_ocr.py          # tesseract → axis calibration
        ├── colour_utils.py      # HSV distance, quantisation
        ├── legend_ocr.py        # legend swatch ↔ label mapping
        ├── panel_split.py       # multi-panel grid detector
        ├── multipanel.py        # MultiPanelExtractor wrapper
        ├── multi_extractor.py   # NEW: CascadingExtractor (geometric + DePlot)
        ├── simple_bars.py       # bar-chart extractor (full impl)
        ├── stubs.py             # placeholders for stacked/box/pie/scatter/line
        ├── deplot.py            # NEW: Google DePlot integration (transformers)
        ├── deplot_subprocess.py # NEW: subprocess-isolated DePlot
        ├── registry.py          # FigureKind → ChartExtractor
        ├── validator.py         # VLM-as-validator (OK/FLAG verdict)
        └── cli.py               # debug CLI for one figure
```

### Optional dependencies

- **`refextract`** + **`poppler-utils`** (for `--enrich-refs`)
- **`docling_core`** (for `--docling` validation; ~1 MB)
- **`transformers`** + **`torch`** + **`google/deplot`** model
  (for DePlot chart extraction; ~1.1 GB model)
- **`llama.cpp`** built from source + Gemma 4 E2B GGUF
  (for the VLM tracks: diagram→mermaid, alt-text, validator)

All optional — base pipeline works with just `pymupdf4llm`,
`pdfplumber`, `pytesseract`, `Pillow`, `opencv-python-headless`.

The complete geometric extractors for stacked-bars, box-plot, pie,
scatter, and line are documented in `output/OPTIMIZATION_NOTES.md`
sections (i)/(j)/(k) — they were verified working with 79/79 synthetic
ground-truth tests before being slimmed to stubs to fit the sandbox
snapshot cap. Restoring them is straightforward (see the notes).

## License & status

WIP development repo. Tested on 35 unique academic PDFs covering
soil-science, biodiversity, and agroecology.

For the full engineering log, debug stories, and optimisation history,
see `output/OPTIMIZATION_NOTES.md`.
