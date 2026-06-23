# Eval harness

Three honest checks for the pipeline.

## 1. Ground-truth corpus + WER benchmark

`fetch_arxiv.py` downloads 10 well-known arXiv papers (Attention,
BERT, ViT, ResNet, GAN, VAE, ...) — both the rendered PDF and the
LaTeX source. The LaTeX is converted to plain text via pandoc (with
a permissive regex-stripper as fallback) and saved as
`corpus/<arxiv_id>/ground_truth.txt`.

`run_eval.py` then runs each PDF through five extractors and scores
the output against the ground truth:

| Extractor | What it is |
|-----------|------------|
| `pdftotext` | poppler-utils with `-layout` |
| `pdftotext-stream` | poppler-utils stream order |
| `pymupdf4llm` | the library we build on |
| `pdf2md-postprocess` | pymupdf4llm + our `postprocess_md.postprocess_full` |
| `pdf2md-reorder-e1` | our E1 reading-order pass |

Metrics (all on lowercased, whitespace-normalised text):
* **char_ratio** — `len(extracted) / len(gt)`
* **F1** — word-set F1
* **WER\*** — proxy WER on sorted token bags (monotone, lower=better)

```bash
python3 -m eval_harness.fetch_arxiv
python3 -m eval_harness.run_eval
# → eval_harness/REPORT.md + REPORT.json
```

**Results on 10 papers (June 2026):**

| Extractor | avg F1 | avg WER* | avg s |
|-----------|--------|----------|-------|
| pdftotext            | 0.777 | 0.024 | 0.11  |
| pdftotext-stream     | 0.802 | 0.020 | 0.10  |
| pymupdf4llm          | 0.803 | 0.019 | 15.4  |
| pdf2md-postprocess   | 0.803 | 0.019 | 15.4  |
| pdf2md-reorder-e1    | 0.766 | 0.033 | 0.12  |

### What this told us (honest)

* **Our postprocess adds 0 word-set F1 over pymupdf4llm** — the
  postprocessing rearranges, it doesn't add or lose words. Fine,
  expected; structure ≠ fidelity.
* **Our E1 reading-order pass is actively worse** (F1 0.766 vs 0.802
  for pdftotext-stream). Likely cause: column reassignment is
  dropping some text blocks at column boundaries. **This is the kind
  of thing the harness exists to catch.**
* **pdftotext is ~150× faster and only 0.001 F1 behind pymupdf4llm**
  on these PDFs. For pure-text use cases we shouldn't claim to need
  pymupdf4llm.
* WER* values (1.9-3.3%) are all small because the proxy uses a
  sorted-bag walk, not a true edit distance. The values are
  monotone-comparable, not absolute.

### Caveats

* `WER*` is a proxy, not strict Levenshtein. A true Levenshtein on
  50k-word papers is O(N²) and would take hours per pair. The proxy
  is monotone (worse extractions get higher WER*) so cross-extractor
  ranking is reliable, but **don't quote the absolute numbers as
  "word error rate" anywhere external**.
* Ground truth: arXiv LaTeX → pandoc text. Pandoc DROPS math,
  citations, references, and ~all formatting. The "extracted" side
  KEEPS all that. So all `char_ratio` values are > 1.3 because the
  PDF has more raw chars than the stripped-LaTeX baseline. The
  precision metric naturally gets penalised, the recall stays valid.
* Most of these arXiv papers are 2-col ML papers — we haven't tested
  scanned, multilingual, or non-ML domains here.

## 2. Static corpus browser

`pipeline_v2/corpus_browser.py` emits a single self-contained
`output/index.html` (no server, no JS frameworks, no external assets)
that lets you click through every processed paper:

* searchable list of papers
* rendered markdown body
* figures gallery (with body-text reference counts from E6)
* references table with DOI badges
* image zoom-on-click

```bash
python3 -m pipeline_v2.corpus_browser
# → output/index.html (a single ~7 MB file, open it in any browser)
```

## 3. Failure-mode catalog

`failure_modes.py` synthesises 10 deliberately broken PDFs and probes
which pipeline stage tolerated each:

| # | Failure mode |
|---|--------------|
| F01 | encrypted (password-protected) PDF |
| F02 | empty PDF |
| F03 | image-only PDF (no embedded text) |
| F04 | rotated pages (0/90/180°) |
| F05 | giant paragraph, no headings |
| F06 | mid-word page break |
| F07 | mixed-script unicode (English + CJK) |
| F08 | identical duplicated pages |
| F09 | random-bytes garbage with PDF header |
| F10 | single-page table-only |

```bash
python3 -m eval_harness.failure_modes
# → eval_harness/FAILURE_REPORT.md + FAILURE_REPORT.json
```

**Lessons surfaced by this catalog:**

* **F01 encrypted**: pymupdf opens it but every downstream stage
  crashes with `ValueError: document still encrypted`. The pipeline
  needs an explicit `doc.authenticate()` step OR an early-exit when
  `doc.needs_pass` is true.
* **F03 image-only**: pymupdf4llm silently OCRs via Tesseract — good!
  pdftotext returns empty. Our E1 reorder returns empty. So our
  pipeline IS better on this case (because pymupdf4llm does OCR).
* **F04 rotated**: pymupdf4llm only gets 21 chars (rotation confuses
  its text extractor); pdftotext + our reorder get the full 60+ chars.
* **F09 garbage bytes**: every stage cleanly raises an exception with
  a useful message. Good error hygiene.
* **F10 table-only**: pymupdf4llm gets all 41 chars; pdftotext +
  reorder lose chars (the column layout confuses pdftotext).
