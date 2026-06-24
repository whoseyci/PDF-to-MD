# Research notes — what's new in document AI (June 2026)

User asked us to look into two specific things:

1. **"Agent language"** for document parsing pipelines
2. **DeepSeek's VLM pointer tracking** for layout identification

This file captures what we found and how (or whether) each fits the
PDF-to-MD pipeline.

---

## 1. DeepSeek-OCR / OCR-2: "grounding" prompts and pointer tracking

### What changed

DeepSeek-OCR (Oct 2025) and the v2 update (Jan 2026) introduced two
features directly relevant to our problem:

* **Visual Causal Flow architecture** -- v2 reads pages "step by step
  according to structure" instead of left-to-right raster scan.
  Result on OmniDocBench v1.5: 91.09% (up 3.73% from v1). [Source: 36kr.com
  summary, Jan 2026.] PaddleOCR-VL leads at 92.86% but is heavier.

* **"Grounding" prompts** -- when called with
  `"<|grounding|>Convert the document to markdown."`, the model emits
  text *with bounding-box pointers* per text region. This is what
  the user called "pointer tracking". Concretely the output looks
  like:

      <|ref|>title<|/ref|><|grounded|>...
      [box: 120,40 → 580,80]
      Attention Is All You Need
      ...

  i.e. each text region carries its origin bbox. You can then re-derive
  the document layout *from the model's own attention pattern*
  rather than running a separate layout detector.

* **"Free OCR" mode** (`Free OCR.`) -- pure text dump, no structure.
* **"Locate" mode** -- `Locate <|ref|>xxxx<|/ref|>` returns the bbox of
  a given text span. Useful for fact-grounding ("show me where this
  appears in the PDF").

### Available APIs

The HF model card (`deepseek-ai/DeepSeek-OCR`) exposes:

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR",
                                          trust_remote_code=True)
    model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR",
                                         trust_remote_code=True,
                                         _attn_implementation="flash_attention_2",
                                         torch_dtype="bfloat16",
                                         use_safetensors=True).cuda()
    res = model.infer(tokenizer=tok, prompt="<image>\n<|grounding|>...",
                       image_file=img, output_path=out)

Concrete constraints we hit:
* Requires CUDA + flash-attention. **Won't run on our 2 vCPU sandbox.**
* ~6 GB weights, fp16. CPU inference would be 30-60 s/page.

### Should we adopt?

**Not yet, but the abstraction is right.** The path forward:

1. Keep our `pipeline_v2/gemma_ocr.py` interface (`reocr_pages`,
   `select_low_confidence_pages`) -- it already gives us a
   "validator on low-confidence pages" hook.
2. Add a `DeepseekOCR` backend class that implements the same
   interface. The user explicitly preferred we not add a second
   model, so it stays an opt-in plugin.
3. Use the grounding-prompt output (with bboxes) to feed our
   `caption_pairing.py` and `chart_extract` modules a more reliable
   layout. **This would replace our negative-space heuristic in E3
   with VLM-confirmed layout.**

The user's instinct here is right: we don't want to default to
DeepSeek (extra dependency), but the grounding output is genuinely
useful when it IS available. Treat it as a "if you have it, you get
better caption pairing + layout" optional path.

---

## 2. Agent-style document extraction (2026 vintage)

### The landscape

Three flavours of "agent" in document parsing:

**a. Single-pass LLM dressed up as agentic** (most common)
   * OCR → GPT-4 prompt → JSON. No planning, no validation.
   * Examples: Airparser, many SaaS tools.
   * **Verdict**: not really agentic. Not interesting for our use case.

**b. Multi-step agent frameworks** (StructSense, Crew.AI-based)
   * Plan → extract → validate → cross-check → repeat.
   * Uses Crew.AI / LangGraph / LlamaIndex for orchestration.
   * Tools: GROBID for PDF parsing, vector DB for ontology lookup,
     LLM for reasoning.
   * **Verdict**: useful for *structured field extraction* (invoices,
     tax forms, clinical narratives) where you have a known target
     schema. Less relevant for our "convert any PDF to clean markdown"
     job.

**c. PaddleOCR-VL + Qianfan-OCR ("Layout-as-Thought")**
   * arXiv 2603.13398 (Mar 2026): Qianfan-OCR introduces
     **Layout-as-Thought** -- the OCR model emits its layout-analysis
     reasoning as part of the output before generating text. This lets
     a single end-to-end model recover bbox-level layout (like
     DeepSeek-OCR grounding) AND avoid pipeline cascading errors.
   * **Verdict**: the most relevant new direction. The "thought" tokens
     would be useful as a layout signal AND as confidence telemetry.

### Should we adopt agent-style?

**Yes, in one specific spot: cross-validation between extractors.**

We already have multiple extractors that produce overlapping outputs:
  * `chart_extract` (geometric) + `chart_extract/deplot` (VLM)
  * `pdftotext` + `pymupdf4llm` (in `text_extract.py`)
  * `caption_pairing` + `figure_refs` (both find figure mentions
    different ways)

The agent move would be: a small validator agent that, when two
extractors disagree, picks the more confident one (or invokes a
third tiebreaker). This is **light** -- no LangChain, no GPT-4 call
per page. Just a `Validator` class that owns the policy.

I'll write it up as **E10 (validator agent)** in
`RESEARCH_DIRECTIONS.md` and we can implement when the eval shows it
would help.

---

## What actually shipped this turn

(Based on the user's direct asks):

* **pdftotext is now the smart-dispatcher default** -- `text_extract.py`
  tries pdftotext per page, falls back to pymupdf4llm (which OCRs)
  on any page with < 100 chars.
* **E1 fixed** -- two real bugs:
  1. Our tokenizer didn't expand Unicode ligatures (`ﬁ` → `fi`),
     making `classiﬁcation` look like two tokens to the F1 metric.
     This penalised any pymupdf-based extractor that preserved the
     ligature glyph. Fix in `eval_harness/run_eval.py::expand_ligatures`.
  2. E1 reorder didn't dehyphenate line-end breaks
     (`configura-\ntion`). Fix in
     `pipeline_v2/reading_order.py::dehyphenate`.
* **Rotation detector** -- `pipeline_v2/rotation_fix.py` uses
  Tesseract OSD to detect mis-rotated pages and patches the page
  rotation flag so extraction reads upright. Integrated into the
  auto-dispatcher (opt-out with `rotation_fix=False`).
* **Eval corpus expanded** -- 16 papers now (was 10): 5 new
  non-ML domains (physics, math, astronomy, economics, biology)
  plus 2 synthetic "scanned" PDFs (rasterised+re-embedded versions
  of existing arXiv papers, to test the OCR-fallback path).

### Numbers (16 papers, June 2026)

| Extractor | avg F1 | avg WER\* | avg s |
|-----------|--------|-----------|-------|
| pdftotext             | 0.706 | 0.153 | 0.07 |
| pdftotext-stream      | 0.727 | 0.149 | 0.07 |
| pymupdf4llm           | 0.821 | 0.080 | 14.8 |
| pdf2md-postprocess    | 0.821 | 0.080 | 14.3 |
| pdf2md-reorder-e1     | 0.729 | 0.148 | 0.10 |
| **pdf2md-auto** ⭐    | **0.821** | **0.077** | **3.0** |
| pdf2md-auto-rotfix    | 0.821 | 0.077 | 16.3 |

* **pdf2md-auto matches pymupdf4llm quality at ~5× the speed**
  (because most pages don't need the fallback, only the scanned ones).
* Scanned PDFs (1503.02531_scanned, 1406.2661_scanned): pdftotext
  alone returns F1=0.0 (no extractable text); the auto-dispatcher
  hits F1=0.78 by routing those pages to pymupdf4llm's OCR fallback.
* Rotation-fix adds ~13s per paper (Tesseract OSD on every page) and
  doesn't help these "born-digital" arXiv papers -- it's purely
  insurance for the genuinely-rotated case. Off by default, on for
  the F04 rotated test.
