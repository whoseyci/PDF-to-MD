# Gemma 4 E2B (April 2026) on the 2 GB sandbox

Real test run of `google/gemma-4-E2B-it-GGUF` (Q3_K_S) via
`llama-mtmd-cli`, subprocess-isolated, on 1.9 GB RAM with no swap.

## Setup that made it fit

```
Sandbox:  1.9 GB RAM total, no GPU, 2 vCPU
Model:    unsloth/gemma-4-E2B-it-GGUF / gemma-4-E2B-it-Q3_K_S.gguf  (2.4 GB)
mmproj:   mmproj-F16.gguf                                            (940 MB)
Runtime:  llama.cpp built from source, llama-mtmd-cli only target
Flags:    -fit off            # disable auto-fit check (it predicts OOM)
          --jinja             # needed for the embedded chat template
          --image-max-tokens 70   # smallest of the 5 image-token budgets
          -c 768              # tight context
          -t 2                # match vCPU count
          --no-warmup
```

The key trick: **mmap**. The 2.4 GB GGUF + 940 MB mmproj together far
exceed the 1.9 GB RAM, but `llama.cpp` mmaps both files. The kernel
pages weights in / out as they're touched; resident-set during
inference stays around 1.0-1.4 GB. Each subprocess call gets a fresh
mmap and the kernel reclaims its working set when the process exits.

## Performance

| Phase                              | Time           |
|------------------------------------|----------------|
| Model + mmproj load (cold)         | 3-5 s          |
| Image encoding                     | 1.7-1.8 s      |
| Prompt processing                  | 1-3 s          |
| **Decode**                         | **0.55 tok/s** |
| Short answer (~30 tokens)          | ~60-80 s       |
| Full think + answer (~300 tokens)  | 11-12 min      |
| Validator-style "OK / FLAG"        | ~40-60 s       |

Decode is CPU-bound. The 2 vCPUs are the bottleneck, not memory.

## Quality on real corpus figures

Tested on three baden-bohm-2023 figures (the same paper where SmolVLM-256M
hallucinated *"30-year period from 5000 to 20000"* in the previous
session).

### fig-001 (journal cover / title page)

**SmolVLM-256M output** (last session): *"The image shows a scientific
research paper titled 'Biodiv nest.'…"*  (incorrect title, made up content)

**Gemma 4 E2B output:**
> The image is the cover of a book or publication. The title visible
> is "Agriculture Ecosystems & Environment". The overall theme is
> clearly related to agriculture, ecosystems, and environmental
> [topics]

✓ Correctly identifies it as a journal cover, reads the journal title
exactly, infers the theme. **Far more accurate.**

### fig-004 (6-panel stacked horizontal bar chart)

**Classical chart_extract** (last session): 6 panels detected, 5/6 partial
extractions, real numeric values from geometry.

**Gemma 4 E2B output:**
> The image contains several bar charts (A, B, C, D, E, F) and a
> legend. The legend shows color coding corresponding to different
> categories (likely species or habitats). Each chart appears [to
> compare values across...]

✓ Correctly identifies 6 panels labelled A-F + a legend. (The
classical extractor wins on extracting *numbers*; the VLM wins on
*describing the structure*. These are complementary.)

### fig-007 (box plot, 9 boxes, log10 Y-axis)

**SmolVLM-256M output** (last session): *"The chart shows the effect of
food and nesting resources..."* (vague, wrong)

**Gemma 4 E2B output:**
> The figure compares different conditions/treatments (labeled on
> the x-axis) and measures "Colonies per ha" (labeled on the
> y-axis). X-axis: Shows various experimental groups (e.g., HVL BAU,
> HVL Biodi, BAV BAU, ...).

With a fully-budgeted run (400 tokens, ~12 min):

> A bar chart showing log10 Colonies per ha across various scenarios
> (HVL BAU, HVL Biodi, BAV BAU, BAV Biodi, RHS BAU, and RHS Biodi).

✓ Y-axis label exact: "log10 Colonies per ha". ✓ 6 of the 9 X-axis
groups read correctly. ✓ Calls it a "bar chart" when it's actually a
box plot (one mistake). Massively better than the SmolVLM output.

## Trade-off summary

| Aspect                | SmolVLM-256M       | Gemma 4 E2B Q3_K_S    |
|-----------------------|--------------------|-----------------------|
| Disk                  | 0.5 GB             | 3.4 GB                |
| RAM peak (subprocess) | ~950 MB            | ~1.4 GB               |
| Time per figure       | ~46 s              | 60-700 s              |
| Output quality        | Often hallucinates | Accurate observations |
| Reads OCR'able text   | No                 | Yes                   |
| Identifies chart kind | Sometimes wrong    | Usually right         |
| Multi-step reasoning  | No                 | Yes (with thinking)   |

## Recommended usage

Given the 10-12× slower per-figure cost, Gemma 4 should NOT replace
the classical `chart_extract` pipeline for bar/box/scatter/line/pie
figures (it's slower AND less accurate at extracting actual numbers).

Instead, use it for:

1. **Non-chart figures** (maps, photos, schematics, journal covers)
   where classical extraction returns `UNSUPPORTED`. ~3 min/figure;
   maybe 100 of the 471 corpus figures qualify, so ~5 hours total.

2. **Chart-extraction validator**: ~40 s per chart, replaces the
   SmolVLM "OK / FLAG" cross-checker. The whole 471-figure corpus
   validation would be ~5 hours.

3. **Caption-less figures** for which the classifier returns `UNKNOWN`
   — Gemma reads enough OCR-able text to often produce a useful alt.

The backend is registered as `gemma4-e2b` in
`pipeline_v2/vision/factory.py`:

```python
from pipeline_v2.vision.factory import make_model
m = make_model("gemma4-e2b", per_image_timeout=300)
out = m.describe(image_path, prompt, max_new_tokens=200)
```

## Why not Q4_K_M instead of Q3_K_S?

I picked Q3_K_S (2.4 GB) over Q4_K_M (3.0 GB) to keep more of the
working set in RAM. On this sandbox the disk is fast enough (1.1 GB/s
sustained read) that page-faults aren't the bottleneck anyway — CPU
is — so Q4_K_M would also work and probably produce slightly better
quality. Easy switch via `model_path=` kwarg or by re-downloading.

## Known issues

- **Model defaults to "thinking mode"** even though the template's
  `enable_thinking` flag is false. E2B is supposed to skip thinking
  by default per the model card, but in practice it produces a
  `<|channel>thought\n` block first. The backend's parser handles
  this by either returning the answer after `<channel|>` (if the
  model finished) or returning the substantive observations from
  the thought block (with meta-commentary preamble stripped).
- **Decode rate of 0.55 tok/s** is hard-limited by the 2 vCPU
  sandbox. A 4-core machine would roughly double it; a Raspberry Pi
  5 (4 cores, ARM neon) gets 7.6 tok/s per Google's benchmarks.
- **`--chat-template-kwargs`** flag isn't exposed by `llama-mtmd-cli`
  in this build (only by `llama-server`), so we can't pass
  `enable_thinking: false` cleanly. Workaround pending llama.cpp
  upstream.
