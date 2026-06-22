"""
CLI entry-point: run the vision pipeline over every figure already
extracted by `convert.py`.

Usage
-----

    python -m pipeline_v2.vision.run_all \\
        --output-dir /home/user/output \\
        --model stub                     # or gemma4-e2b
        [--paper baden-bohm-2023 …]      # restrict to specific papers
        [--inject]                       # also splice results into paper.md
        [--force]                        # ignore cache

The runner emits one ``figures/fig-NNN.vision.json`` sidecar per
figure and (if ``--inject`` is given) updates paper.md alt-texts /
inserts Mermaid blocks where validation succeeded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running with `python -m pipeline_v2.vision.run_all` OR direct script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.factory import make_model
from vision.runner import process_figure
from vision.base import FigureKind


def _papers_to_process(output_dir: Path, only_slugs: list[str] | None):
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        if only_slugs and d.name not in only_slugs:
            continue
        if not (d / "paper.json").exists():
            continue
        yield d


def _inject_into_markdown(paper_md_path: Path, results, fig_id_to_meta: dict):
    """
    Rewrite paper.md so:
      * image alt-text becomes the validated alt sentence (if any)
      * Mermaid diagrams are inserted right after their image as a
        ```mermaid code block
      * Markdown tables are inserted right after their image
    Idempotent: re-running with the same sidecars is a no-op.
    """
    import re
    text = paper_md_path.read_text(encoding="utf-8")
    by_id = {r.figure_id: r for r in results if r.error is None}

    def _img_repl(m):
        old_alt = m.group("alt")
        path = m.group("path")
        # Find the figure id by matching the path
        fig_id = None
        for fid, fm in fig_id_to_meta.items():
            if fm["file"].endswith(path.lstrip("./")):
                fig_id = fid
                break
        if not fig_id or fig_id not in by_id:
            return m.group(0)
        res = by_id[fig_id]
        new_alt = res.alt_text or old_alt
        out = f"![{new_alt}]({path})"
        # Append mermaid / markdown_table on the next paragraph
        extras = []
        if res.mermaid:
            extras.append(f"\n\n```mermaid\n{res.mermaid}\n```")
        if res.markdown_table:
            extras.append("\n\n" + res.markdown_table)
        return out + "".join(extras)

    new_text = re.sub(
        r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)",
        _img_repl,
        text,
    )

    if new_text != text:
        paper_md_path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/home/user/output")
    ap.add_argument("--model", default="stub",
                    help="stub | gemma4-e2b")
    ap.add_argument("--paper", action="append", default=None,
                    help="restrict to a specific paper slug; repeatable")
    ap.add_argument("--inject", action="store_true",
                    help="rewrite paper.md with alt-text/Mermaid/table from sidecars")
    ap.add_argument("--force", action="store_true",
                    help="ignore the per-figure sidecar cache")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--per-image-timeout", type=float, default=60.0)
    ap.add_argument("--max-image-dim", type=int, default=512)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Loading vision model: {args.model}")
    model = make_model(
        args.model,
        dtype=args.dtype,
        per_image_timeout_seconds=args.per_image_timeout,
        max_image_dim=args.max_image_dim,
    )
    print(f"  → backend: {model.name}")

    grand_total = 0
    grand_ok = 0
    grand_failed = 0
    grand_cached = 0

    for paper_dir in _papers_to_process(output_dir, args.paper):
        paper_json = json.loads((paper_dir / "paper.json").read_text(encoding="utf-8"))
        figs = paper_json.get("figures", [])
        if not figs:
            continue
        print(f"\n[{paper_dir.name}] {len(figs)} figures")

        results = []
        for fm in figs:
            sidecar = paper_dir / "figures" / f"{fm['id']}.vision.json"
            cached_before = sidecar.exists() and not args.force
            t0 = time.time()
            res = process_figure(fm, paper_dir, model, force=args.force,
                                 log_prefix=f"  [{paper_dir.name}/{fm['id']}] ")
            elapsed = time.time() - t0
            if cached_before:
                grand_cached += 1
                tag = "cached"
            elif res.error:
                grand_failed += 1
                tag = f"FAIL {res.error[:50]}"
            else:
                grand_ok += 1
                tag = "ok"
            grand_total += 1
            note = []
            if res.kind:
                note.append(f"kind={res.kind.value if hasattr(res.kind, 'value') else res.kind}")
            if res.mermaid:
                note.append("mermaid✓")
            if res.markdown_table:
                note.append("table✓")
            if res.alt_text:
                note.append(f"alt={len(res.alt_text)}c")
            print(f"  {fm['id']:8s} [{elapsed:5.1f}s] {tag:30s} {' '.join(note)}")
            results.append(res)

        if args.inject:
            paper_md = paper_dir / "paper.md"
            if paper_md.exists():
                fig_id_to_meta = {fm["id"]: fm for fm in figs}
                if _inject_into_markdown(paper_md, results, fig_id_to_meta):
                    print(f"  → injected into paper.md")

    print(f"\n=== Vision pass complete ===")
    print(f"  total:  {grand_total}")
    print(f"  ok:     {grand_ok}")
    print(f"  cached: {grand_cached}")
    print(f"  failed: {grand_failed}")


if __name__ == "__main__":
    main()
