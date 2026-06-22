"""
Gemma 4 E2B (April 2026) vision backend, subprocess-isolated via llama.cpp.

See OPTIMIZATION_NOTES.md section (l) for full setup notes. Key facts:

* Uses `unsloth/gemma-4-E2B-it-GGUF` Q3_K_S (2.4 GB) + mmproj-F16 (940 MB)
* Runs via llama.cpp's `llama-mtmd-cli` in a per-call subprocess
* Fits in 1.9 GB RAM via mmap (kernel pages weights in/out as touched)
* CPU-bound: ~0.55 tok/s on 2 vCPUs, so ~60-80 s per short answer

The subprocess pattern guarantees the kernel reclaims the model's
working-set the moment each call finishes, so the pipeline never holds
3+ GB of weights resident in its own Python process.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..base import VisionModel


_DEFAULT_CLI = "/home/user/.cache/llama_src/build/bin/llama-mtmd-cli"
_DEFAULT_LIB = "/home/user/.cache/llama_src/build/bin"
_DEFAULT_MODEL = "/home/user/.cache/hf/models/gemma4e2b/gemma-4-E2B-it-Q3_K_S.gguf"
_DEFAULT_MMPROJ = "/home/user/.cache/hf/models/gemma4e2b/mmproj-F16.gguf"


class Gemma4SubprocessModel(VisionModel):
    """Vision model backed by per-call ``llama-mtmd-cli`` subprocess."""

    name = "gemma4-e2b-subprocess"

    def __init__(self,
                 *,
                 cli_path: Optional[str] = None,
                 ld_library_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 mmproj_path: Optional[str] = None,
                 per_image_timeout: float = 300.0,
                 n_threads: int = 2,
                 ctx_size: int = 768,
                 image_max_tokens: int = 70,
                 temperature: float = 0.1,
                 tmpdir: Optional[str] = None,
                 **_):
        self.cli_path = (cli_path or os.environ.get("LLAMA_MTMD_CLI")
                          or _DEFAULT_CLI)
        self.ld_library_path = (ld_library_path
                                  or os.environ.get("LLAMA_LIB_DIR")
                                  or _DEFAULT_LIB)
        self.model_path = (model_path or os.environ.get("GEMMA4_MODEL")
                            or _DEFAULT_MODEL)
        self.mmproj_path = (mmproj_path or os.environ.get("GEMMA4_MMPROJ")
                              or _DEFAULT_MMPROJ)
        self.per_image_timeout = float(per_image_timeout)
        self.n_threads = int(n_threads)
        self.ctx_size = int(ctx_size)
        self.image_max_tokens = int(image_max_tokens)
        self.temperature = float(temperature)
        self.tmpdir = tmpdir or os.environ.get("TMPDIR") or "/home/user/.tmp"
        Path(self.tmpdir).mkdir(parents=True, exist_ok=True)
        for p in (self.cli_path, self.model_path, self.mmproj_path):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Gemma4SubprocessModel: required file missing: {p}")

    def describe(self, image_path: Path, prompt: str, *,
                 max_new_tokens: int = 200) -> str:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (
            self.ld_library_path + os.pathsep +
            env.get("LD_LIBRARY_PATH", "")).strip(os.pathsep)
        env["TMPDIR"] = self.tmpdir

        cmd = [
            self.cli_path,
            "-m", self.model_path,
            "--mmproj", self.mmproj_path,
            "--image", str(image_path),
            "-p", prompt,
            "-c", str(self.ctx_size),
            "-n", str(max_new_tokens),
            "-t", str(self.n_threads),
            "--image-max-tokens", str(self.image_max_tokens),
            "--temp", str(self.temperature),
            "--no-warmup",
            "--jinja",
            "-fit", "off",
        ]

        # Capture partial output on timeout by using Popen + read with
        # a deadline. If the model is mid-decode when we hit the wall
        # clock, we can still salvage whatever was produced so far.
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        try:
            stdout, stderr = proc.communicate(timeout=self.per_image_timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            timed_out = True

        full = (stdout or "") + "\n" + (stderr or "")
        text = _extract_assistant_response(full)

        if timed_out:
            # Return what we have if non-empty, otherwise raise.
            if text:
                return f"[TIMED OUT after {self.per_image_timeout:.0f}s; partial output below]\n{text}"
            raise RuntimeError(
                f"gemma4 subprocess timed out after "
                f"{self.per_image_timeout:.0f} s on {image_path} "
                "with no output captured")

        if not text:
            if proc.returncode < 0 or proc.returncode == 137:
                raise RuntimeError(
                    f"gemma4 subprocess killed by signal "
                    f"{-proc.returncode} (likely OOM); ret={proc.returncode}")
            text = full.strip()[-2000:]
        return text


# --------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------

_NOISE_PREFIXES = (
    "common_", "load:", "llama_", "main:", "WARN", "warn",
    "init_audio:", "encoding mtmd", "mtmd batch", "WARNING",
)
_TIMESTAMP_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+ [IWE] ")

_TEMPLATE_DUMP_LINES = {
    "You are a helpful assistant", "Hi there", "Hello", "How are you?",
    "<|think|>", "<|turn>system", "<|turn>user", "<|turn>model",
}


def _extract_assistant_response(text: str) -> str:
    """Parse mtmd-cli output → the model's actual response.

    The mtmd-cli stream contains, in order: model response (stdout),
    then llama.cpp log lines + a chat-template-example dump (stderr).
    We cut everything from the start of the template dump onwards,
    then filter remaining log lines, then prefer the post-thought
    answer block if the model produced one.
    """
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)

    for marker in ("chat template example:",
                    "<|turn>system\n<|think|>\nYou are a helpful assistant",
                    "<|turn>system\nYou are a helpful assistant"):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
            break

    kept_lines = []
    for line in text.splitlines():
        s = line.rstrip()
        if not s: kept_lines.append(""); continue
        if _TIMESTAMP_RE.match(s): continue
        if any(s.lstrip().startswith(p) for p in _NOISE_PREFIXES): continue
        if s.strip() in _TEMPLATE_DUMP_LINES: continue
        if s.startswith(("<|turn>", "<turn|>")): continue
        if "/lib/" in s or "/build/bin/" in s: continue
        if s.startswith("terminate called"): continue
        if s.startswith("https://github.com/ggml-org"): continue
        kept_lines.append(s)
    body = "\n".join(kept_lines).strip()

    # Prefer the answer portion if the model produced a thought block.
    m = re.search(r"<channel\|>\s*(.*?)(?:<turn\|>|$)", body, re.DOTALL)
    if m:
        answer = m.group(1).strip()
        if answer:
            return answer

    # No closed thought channel: strip preamble and meta-commentary.
    body = re.sub(r"^<\|channel>(thought|answer)\s*", "", body)
    body = re.sub(r"<turn\|>.*$", "", body, flags=re.DOTALL).strip()
    m2 = re.search(
        r"(?:Examine|Analyze|Look at|Inspect|Study)\s+the\s+(?:Image|Chart|Figure)[^\n]*\n",
        body, re.IGNORECASE)
    if m2:
        body = body[m2.end():].strip()
    return body.strip()
