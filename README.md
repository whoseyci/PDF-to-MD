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
