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

**Results on 16 papers (June 2026, includes 2 synthetic scanned PDFs
+ 5 non-ML domains):**

| Extractor | avg F1 | avg WER* | avg s |
|-----------|--------|----------|-------|
| pdftotext             | 0.706 | 0.153 | 0.07 |
| pdftotext-stream      | 0.727 | 0.149 | 0.07 |
| pymupdf4llm           | 0.821 | 0.080 | 14.8 |
| pdf2md-postprocess    | 0.821 | 0.080 | 14.3 |
| pdf2md-reorder-e1     | 0.729 | 0.148 | 0.10 |
| **pdf2md-auto** ⭐    | **0.821** | **0.077** | **3.0** |
| pdf2md-auto-rotfix    | 0.821 | 0.077 | 16.3 |

### What this told us (honest, after fixing two harness bugs)

* **The new smart dispatcher (`pdf2md-auto`) ties pymupdf4llm
  quality at ~5× the speed.** It tries pdftotext per page,
  falls back to pymupdf4llm only when pdftotext returns < 100
  chars on a page. Most pages don't need the fallback, so we
  pay near-zero cost.
* **On scanned PDFs (1503.02531_scanned, 1406.2661_scanned):**
  pdftotext alone returns F1=0.000 (no extractable text).
  The dispatcher routes those pages to pymupdf4llm which OCRs
  via Tesseract and recovers F1=0.78. The harness now actively
  tests this critical fallback path.
* **E1 reading-order pass: was actively worse, now tied with
  pdftotext-stream.** Two bugs fixed:
   1. The harness tokenizer didn't expand Unicode ligatures
      (`ﬁ` → `fi`), so any extractor that preserved the glyph
      lost the word inventory comparison. Fix: `expand_ligatures()`
      in `run_eval.py`. This single fix moved pymupdf4llm-side
      F1 from 0.803 to 0.821 on the original 10 papers.
   2. E1 itself didn't dehyphenate line-end breaks
      (`configura-\ntion`). Fix in `pipeline_v2/reading_order.py`.
* **Rotation-fix adds ~13s/paper** (Tesseract OSD on every page).
  On these born-digital arXiv papers it doesn't help (F1 unchanged
  at 0.821), so it's off by default in the auto dispatcher. Turn on
  with `text_extract(pdf, rotation_fix=True)` when you suspect
  rotated scans. Synthetic F04 rotated-PDF test in the failure
  catalog DOES need it.

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
