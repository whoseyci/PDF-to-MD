# `pipeline_v2.vision` — vision-model harness for figures

A model-agnostic harness that converts the figures already extracted
by `convert.py` into machine-readable artefacts:

- short alt-text sentences for accessibility
- Mermaid diagrams for workflow / schematic figures
- markdown tables for bar charts, pie charts, and table-as-image figures
- LaTeX for equation figures

## Architecture

```
pipeline_v2/vision/
├── base.py          ← VisionModel ABC + FigureKind enum + result dataclass
├── classifier.py    ← caption-text → FigureKind (deterministic, no ML)
├── prompts.py       ← per-kind prompt templates
├── validators.py    ← per-kind output sanitisers
├── runner.py        ← classify → prompt → model → validate → cache
├── factory.py       ← make_model(name, **opts) registry
├── run_all.py       ← CLI: batch over all papers, optional --inject
└── backends/
    ├── stub.py      ← returns canned strings (for tests)
    ├── smolvlm.py   ← HuggingFaceTB/SmolVLM-{256M,500M,2.2B}-Instruct
    └── gemma.py     ← google/gemma-3-{4b,12b}-it (multimodal)
```

## CLI usage

After running `python pipeline_v2/batch.py`, run the vision pass:

```bash
# 1. Smoke-test with the stub model (no ML deps required)
python -m pipeline_v2.vision.run_all --model stub --paper baden-bohm-2023

# 2. Real run with SmolVLM-256M (needs ≥3 GB RAM)
pip install 'transformers==4.50.*' 'torch' --index-url https://download.pytorch.org/whl/cpu
python -m pipeline_v2.vision.run_all --model smolvlm-256m

# 3. Real run with Gemma 3 4B (needs ≥6 GB RAM bf16, or any GPU)
pip install 'transformers>=4.50' torch accelerate
python -m pipeline_v2.vision.run_all --model gemma3-4b --hf-token <yours>

# 4. After the vision pass, rewrite paper.md with the model's outputs:
python -m pipeline_v2.vision.run_all --model smolvlm-256m --inject

# 5. Any HuggingFace SmolVLM/Idefics3-compatible repo on the fly:
python -m pipeline_v2.vision.run_all --model hf:HuggingFaceTB/SmolVLM-Instruct
```

### CLI flags

| flag | default | purpose |
|---|---|---|
| `--output-dir DIR` | `/home/user/output` | where the per-paper subdirs live |
| `--model NAME` | `stub` | backend (see registry) |
| `--paper SLUG` | (all papers) | restrict to one paper (repeatable) |
| `--inject` | off | rewrite paper.md alt-text + insert mermaid/tables |
| `--force` | off | ignore the per-figure `*.vision.json` cache |
| `--dtype TYPE` | `float32` | `float16` / `bfloat16` / `float32` |
| `--per-image-timeout SEC` | `60` | SIGALRM cutoff per image |
| `--max-image-dim PX` | `512` | downscale images before encoding |

## How fail-safe?

Every layer of the pipeline catches and records errors instead of
propagating them:

- model loading failure → cached and reported per call, never raised
- per-image inference exception → wrapped in `try/except`, captured as
  `FigureVisionResult.error`, sidecar still written, next image
  attempted
- inference exceeding `--per-image-timeout` → `SIGALRM` aborts the
  call, recorded as `_TimeoutError`, batch continues
- validator rejecting output → recorded as `output-failed-validation`,
  no fake mermaid/table written, alt-text falls back to caption

A single failing figure can never abort the batch.

## How model-agnostic?

Swap the backend with one CLI flag (`--model`) or programmatically:

```python
from pipeline_v2.vision import make_model, process_figure

model = make_model("smolvlm-256m", dtype="float16")  # or 'gemma3-4b', 'stub', 'hf:<repo>'
result = process_figure(figure_meta, paper_dir, model)
```

To add a new backend (e.g. a remote API):

```python
# pipeline_v2/vision/backends/my_backend.py
from ..base import VisionModel
from pathlib import Path

class MyBackend(VisionModel):
    name = "my-backend"
    def __init__(self, **opts):
        ...
    def describe(self, image_path: Path, prompt: str, *, max_new_tokens: int = 200) -> str:
        # call API / load weights / etc, return raw text
        ...
```

Then register in `factory.py`:

```python
if name == "my-backend":
    from .backends.my_backend import MyBackend
    return MyBackend(**opts)
```

## Caching

Each figure produces a sidecar at `<paper_dir>/figures/<fig_id>.vision.json`:

```json
{
  "figure_id": "fig-001",
  "kind": "bar_chart",
  "classifier_reason": "strong-hint:stacked/grouped bar",
  "model_name": "smolvlm-256m",
  "prompt": "…full prompt sent to the model…",
  "raw_output": "…exact text the model returned…",
  "alt_text": "Stacked bar chart of land-use proportions across…",
  "mermaid": null,
  "markdown_table": "| Crop | Share |\n|---|---|\n| Maize | 35% |\n…",
  "extracted_data": null,
  "error": null,
  "elapsed_seconds": 3.4
}
```

If the sidecar already exists, the runner skips the figure (use
`--force` to override). Bumping the model name changes the cached
output because re-running with a different `--model` produces a
different sidecar contents.

## Memory rules of thumb (CPU inference)

| Model | params | fp32 peak | fp16 peak | works in 2 GB sandbox? |
|---|--:|--:|--:|---|
| `stub` | 0 | <50 MB | n/a | ✓ (anywhere) |
| `smolvlm-256m-subprocess` | 256 M | ~1.4 GB | ~1.0 GB | ✓ **yes** with `--dtype fp16 --max-image-dim 384` |
| `smolvlm-256m` (in-process) | 256 M | ~1.4 GB | ~1.0 GB | borderline — model OOM kills whole batch |
| `smolvlm-500m` | 500 M | ~2.2 GB | ~1.3 GB | ✗ needs **≥4 GB** |
| `smolvlm-2.2b` | 2.2 B | ~9 GB | ~5 GB | ✗ needs **≥6 GB** + GPU recommended |
| `gemma3-4b` | 4 B | ~16 GB | ~8 GB / 4 GB int4 | ✗ needs **≥10 GB** or GPU |

### CRITICAL: disable image splitting

SmolVLM's default processor has `do_image_splitting=True`, which tiles
every image into **17 × 512 × 512 patches** before the vision encoder.
On tight-memory hosts this produces 17× the activation memory and OOMs
generation. Both the `smolvlm` and `smolvlm-subprocess` backends in
this harness pass `do_image_splitting=False` by default, and clamp the
processor's max longest-edge to `--max-image-dim` (default 384).
Override with `disable_image_splitting=False` if you specifically want
the multi-tile mode.

## Subprocess-isolated backend

`smolvlm-256m-subprocess` spawns a fresh Python child per `describe()`
call. The child loads the model, runs inference, writes the result to
a JSON file, and exits. If the child is OOM-killed by the kernel, the
parent harness catches the `signal 9` exit and records a clean
`"smolvlm subprocess exited with signal 9 (likely OOM-kill)"` error in
the figure's sidecar — the batch continues with the next figure.

Trade-off: ~7-15 s startup overhead per figure (re-loads the model
each time). Worth it on hosts where in-process OOMs would kill the
whole run. On hosts with comfortable RAM use the regular
`smolvlm-256m` backend for ~3-5× speedup.

Verified on our 2 GB sandbox: **7 of 8 figures** in
`baden-bohm-2023` produced valid descriptions, total ~6 min for the
paper (~46 s/figure). The 1 failure was correctly caught by the
prompt-echo filter — SmolVLM regurgitated the prompt template instead
of describing the image, and the validator rejected it.
