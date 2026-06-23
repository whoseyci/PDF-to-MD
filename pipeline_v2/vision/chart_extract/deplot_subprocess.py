"""Subprocess-isolated wrapper around DeplotExtractor.

Each `extract()` call spawns a fresh Python subprocess that loads
DePlot, runs once, prints a JSON result, exits. Kernel reclaims
the model's working set the moment the subprocess exits, so the
pipeline never holds 1+ GB of weights resident in its own process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .base import ChartExtractor, ChartExtractionResult, ExtractionStatus


_WORKER_SCRIPT = r'''
import json, sys
from pathlib import Path

job_path = sys.argv[1]
result_path = sys.argv[2]

with open(job_path) as f:
    job = json.load(f)

sys.path.insert(0, job["pipeline_root"])
from pipeline_v2.vision.chart_extract.deplot import DeplotExtractor

ext = DeplotExtractor(
    max_new_tokens=job.get("max_new_tokens", 512),
    max_image_dim=job.get("max_image_dim", 640),
)
res = ext.extract(Path(job["image_path"]))
out = res.to_dict()

with open(result_path, "w") as f:
    json.dump(out, f, ensure_ascii=False)
'''


class DeplotSubprocessExtractor(ChartExtractor):
    """Subprocess-isolated DePlot extractor."""

    name = "deplot-subprocess/v1"

    def __init__(self, *,
                 per_image_timeout: float = 300.0,
                 max_new_tokens: int = 512,
                 max_image_dim: int = 640,
                 tmpdir: Optional[str] = None,
                 pipeline_root: str = "/home/user",
                 **_):
        self.per_image_timeout = float(per_image_timeout)
        self.max_new_tokens = int(max_new_tokens)
        self.max_image_dim = int(max_image_dim)
        self.tmpdir = tmpdir or os.environ.get("TMPDIR") or "/home/user/.tmp"
        self.pipeline_root = pipeline_root
        Path(self.tmpdir).mkdir(parents=True, exist_ok=True)

    def extract(self, image_path: Path, *, caption=None, ocr_text=None
                ) -> ChartExtractionResult:
        t0 = time.time()
        result = ChartExtractionResult(extractor=self.name,
                                          status=ExtractionStatus.ERROR)
        if not Path(image_path).exists():
            result.reason = f"image missing: {image_path}"
            result.elapsed_seconds = round(time.time() - t0, 3)
            return result
        work = Path(self.tmpdir) / f"deplot_{os.getpid()}_{int(time.time()*1000)}"
        work.mkdir(parents=True, exist_ok=True)
        worker_py = work / "worker.py"
        worker_py.write_text(_WORKER_SCRIPT)
        job_path = work / "job.json"
        result_path = work / "result.json"
        job_path.write_text(json.dumps({
            "image_path": str(image_path),
            "pipeline_root": self.pipeline_root,
            "max_new_tokens": self.max_new_tokens,
            "max_image_dim": self.max_image_dim,
        }))
        env = dict(os.environ)
        env.setdefault("HF_HOME", "/home/user/.cache/hf")
        env.setdefault("TMPDIR", self.tmpdir)
        try:
            proc = subprocess.run(
                [sys.executable, str(worker_py),
                  str(job_path), str(result_path)],
                env=env, capture_output=True, text=True,
                timeout=self.per_image_timeout, check=False)
        except subprocess.TimeoutExpired:
            result.reason = (f"deplot subprocess timed out after "
                              f"{self.per_image_timeout:.0f}s")
            result.elapsed_seconds = round(time.time() - t0, 3)
            return result
        if not result_path.exists():
            sig = -proc.returncode if proc.returncode < 0 else proc.returncode
            result.reason = (f"deplot subprocess exited rc={proc.returncode} "
                              f"(sig={sig}; likely OOM); "
                              f"stderr: {(proc.stderr or '')[:500]}")
            result.elapsed_seconds = round(time.time() - t0, 3)
            return result
        try:
            d = json.loads(result_path.read_text())
            for k, v in d.items():
                if hasattr(result, k):
                    setattr(result, k, v)
            if isinstance(result.status, str):
                try:
                    result.status = ExtractionStatus(result.status)
                except ValueError:
                    result.status = ExtractionStatus.ERROR
            result.extractor = self.name
            result.elapsed_seconds = round(time.time() - t0, 3)
        except Exception as e:
            result.reason = f"failed to parse result.json: {e}"
            result.elapsed_seconds = round(time.time() - t0, 3)
        finally:
            try:
                for p in (worker_py, job_path, result_path):
                    if p.exists(): p.unlink()
                work.rmdir()
            except Exception:
                pass
        return result
