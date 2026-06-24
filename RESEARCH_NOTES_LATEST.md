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

### Update (user-pointed)

User clarified what they meant by "agent language": **RecursiveMAS**
(Yang et al., 2026, arXiv 2604.25917, recursivemas.github.io). My
earlier survey conflated two very different things — let me redo it
with the right reference.

### What RecursiveMAS actually is

A multi-agent framework where agents communicate **through latent
hidden states**, not text. Each agent acts like a recursive layer:

1. **InnerLink** — the agent's last-layer hidden state is fed back
   as its next input, generating a chain of "latent thoughts" in
   continuous space (no token decoding).
2. **OuterLink** — those latent thoughts are projected and passed to
   the next agent in the loop.
3. The loop closes: last agent's latent thoughts feed back to the
   first. **Only the final round decodes to text.**

Both Links are small (~13M params, 0.31% of system); base LLMs are
frozen. Trained with an inner–outer loop: per-agent warm-start
(regression to embedding distribution of GT), then full-system
unroll with one cross-entropy on final text.

**Reported numbers** (across 9 benchmarks):
* +8.3% avg accuracy vs strongest baseline
* 1.2× → 2.4× speedup as recursion deepens (r=1 → r=3)
* 34.6% → 75.6% token-usage reduction
* On AIME 2025 math: +18.1% over best baseline

### Why it works (theory)

* **Runtime**: text-mediated MAS pays an O(|V|·d_h) vocab-projection
  cost per agent step; latent recursion replaces that with O(d_h²),
  much cheaper since d_h ≪ |V|.
* **Gradient stability**: text-based recursive SFT suffers from
  vanishing gradients when token predictions are confident
  (entropy ≤ ε); latent recursion keeps gradient norm Ω(1 − ...),
  so credit signals survive across rounds.

### Four collaboration patterns documented:

1. **Sequential** — Planner → Critic → Solver (decompose-judge-refine)
2. **Mixture** — Math + Code + Science specialists → Summarizer
3. **Distillation** — Expert ↔ Learner (smaller model learns)
4. **Deliberation** — Reflector ↔ Tool-Caller (Python + search)

---

### Can we adopt RecursiveMAS literally?

**No, and I should be honest about why.**

| Constraint of RecursiveMAS | Our reality | Verdict |
|----------------------------|-------------|---------|
| Needs CUDA + 16+ GB VRAM (training & inference) | 2 vCPU, 2 GB RAM, no GPU | ❌ |
| Requires shared latent-space access between agents (raw hidden states) | We invoke Gemma 4 via llama.cpp **subprocesses** — the hidden states never leave the child process | ❌ |
| Same model family / framework (HF transformers, identical compute graph) | We mix subprocess-llama.cpp Gemma, in-process Pix2Struct (DePlot), and pure-Python geometric extractors | ❌ |
| ~13M trainable RecursiveLink params, full inner+outer SFT loop | We have no training infrastructure, no GPU, no labelled trajectories | ❌ |
| Domains: math, code, science, medicine (long reasoning chains) | Our domain: PDF → markdown (mostly perceptual; reasoning is shallow) | ⚠ poor fit |

So a literal RecursiveMAS port is off the table. The **honest take**:
its biggest wins are on benchmarks where multi-round reasoning
actually matters (AIME, GPQA, code gen). PDF→markdown is mostly a
*perception* task — most of our pipeline doesn't benefit from
iterative refinement.

### What we CAN adapt: the four collaboration patterns

The RecursiveMAS structural patterns are useful frames even WITHOUT
the latent-state machinery. We already have a (very limited) version
of "Sequential" in `CascadingExtractor` (simple_bars → DePlot), but
no Mixture, Distillation, or Deliberation. Sketching what each
would look like for us:

**E14 — Sequential pattern (extend our CascadingExtractor)**
  Already partially shipped. Could add: a third stage that runs
  *after* DePlot's output is in hand, looking at confidence
  warnings and re-asking with a different prompt. Currently we just
  return DePlot's first answer. ~0.5d.

**E15 — Mixture pattern for figure classification**
  Today `classifier.py` assigns one `FigureKind` per figure. With
  Mixture: have a `BarSpecialist`, `LineSpecialist`, `DiagramSpecialist`
  each return a confidence + structured output; a `Summarizer` picks
  the highest-confidence interpretation. This is more robust than
  trusting a single classifier. ~1d.

**E16 — Distillation pattern for caption extraction**
  Today we run Gemma 4 (slow, large) on every caption that needs
  alt-text. Distillation: a small student (TextRank summariser,
  rule-based extractor) handles 80% of captions; Gemma is only
  invoked on the 20% the student flags low-confidence. Already
  *roughly* how we use Gemma; would benefit from a formal handshake.
  ~0.5d.

**E17 — Deliberation pattern for the chart_extract validator**
  We have `chart_extract/validator.py`. Add a Reflector agent that
  re-reads the validator's complaint and decides whether to
  re-extract with different params (e.g. tighter axis OCR band,
  different bar-detection threshold) vs accept the partial result.
  This is the most "agentic" pattern and probably the highest-ROI.
  ~1.5d.

### What I'd actually build first

If we're being honest: **none of these are urgent**. The eval harness
just showed that `pdf2md-auto` already matches `pymupdf4llm` quality
at 5× the speed on text extraction. The bottleneck is the
figure pipeline (chart extraction takes 40-100 s/figure with
DePlot fallback), and there a Mixture pattern (E15) or Deliberation
(E17) would help more than a literal RecursiveMAS port.

If/when training infrastructure shows up (GPU host, gradient access
between agents), the actual RecursiveMAS would be worth a real port.
For now: the patterns are good design vocabulary, the latent-state
mechanism isn't reachable with our compute stack.

### The cite

```
@misc{recursivemas,
  title   = {Recursive Multi-Agent Systems},
  author  = {Xiyuan Yang and Jiaru Zou and Rui Pan and Ruizhong Qiu
              and Pan Lu and Shizhe Diao and Jindong Jiang and Hanghang Tong
              and Tong Zhang and Markus J. Buehler and Jingrui He and James Zou},
  year    = {2026},
  eprint  = {2604.25917},
}
```

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
