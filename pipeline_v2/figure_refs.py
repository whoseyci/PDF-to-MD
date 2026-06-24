"""E6 -- Figure-reference linking.

For each figure in a paper, find the paragraphs in the body text that
mention it ("Figure 3", "Fig. 3", "Figs. 3 and 4", "Figure 3a-c", ...)
and attach back-references on the figure record so downstream
consumers (e.g. RAG indexers) can pair figure → paragraphs that
discuss it.

Inputs:
    paper.json   -- already produced by the pipeline
    paper.md     -- the rendered markdown (used for paragraph splitting)

Outputs (mutates ``paper.json`` in place):
    for each figure: ``referenced_in: [{para_idx, sentence}, ...]``
    plus a top-level ``figure_references_summary`` block with counts

Pure stdlib. Idempotent (safe to re-run).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------------------

# We accept:
#   "Figure 3" / "Figure 3a" / "Figure 3a-c"
#   "Fig. 3"  / "Fig 3"
#   "Figs. 3 and 4" / "Figures 3-5" / "Figs 3, 4, 7"
# We need to capture the LIST of numbers because a single mention can
# reference multiple figures.
_FIG_TOKEN = re.compile(
    r"\b(?:Figure|Figures|Fig\.|Figs\.|Fig|Figs)\s*"
    r"([0-9]+[A-Za-z]?(?:\s*(?:-|–|—|to|,|and|&)\s*[0-9]+[A-Za-z]?)*)",
    re.IGNORECASE,
)

# Split a captured "3, 4 and 5" / "3-5" / "3 to 5" into individual ints.
_NUM_RANGE = re.compile(
    r"([0-9]+)(?:\s*(?:-|–|—|to)\s*([0-9]+))?", re.IGNORECASE)


def _expand_nums(group: str) -> List[int]:
    """Expand a fig-ref capture into a sorted list of unique ints."""
    out: List[int] = []
    # Strip suffix letters before parsing numbers
    cleaned = re.sub(r"([0-9]+)[A-Za-z]+", r"\1", group)
    # Split on , / and / &
    parts = re.split(r"\s*(?:,|and|&)\s*", cleaned, flags=re.IGNORECASE)
    for p in parts:
        m = _NUM_RANGE.match(p.strip())
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        lo, hi = min(a, b), max(a, b)
        if hi - lo > 50:
            # Sanity bound -- never expand bogus huge ranges
            out.append(a)
            continue
        out.extend(range(lo, hi + 1))
    return sorted(set(out))


def find_mentions(text: str) -> List[Tuple[int, int, List[int]]]:
    """Return list of (start, end, [fig_nums]) for each Figure mention."""
    out = []
    for m in _FIG_TOKEN.finditer(text):
        nums = _expand_nums(m.group(1))
        if nums:
            out.append((m.start(), m.end(), nums))
    return out


# ----------------------------------------------------------------------
# Paragraph splitting
# ----------------------------------------------------------------------

# Split markdown into paragraphs, dropping figure/table/heading lines.
# A paragraph is a contiguous block separated by blank lines.

_SKIP_LINE_PREFIX = ("![", "|", "*— Page ", "## ", "# ", "### ",
                       "**Fig", "**Table", "---")


def split_paragraphs(md: str) -> List[str]:
    blocks = re.split(r"\n\s*\n", md)
    paras = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # Drop tables / figures / page markers / pure heading blocks
        first = b.splitlines()[0]
        if any(first.startswith(p) for p in _SKIP_LINE_PREFIX):
            continue
        # Drop blocks where every line is a heading/list bullet
        if all(line.startswith(("#", "*", "-", ">", "|"))
               for line in b.splitlines() if line.strip()):
            continue
        paras.append(b)
    return paras


def extract_sentence(text: str, pos: int) -> str:
    """Return the sentence containing ``pos`` (rough heuristic)."""
    # Find sentence boundaries around pos.
    # Scan back for [.!?] followed by space, or paragraph start.
    start = 0
    for i in range(pos - 1, -1, -1):
        if text[i] in ".!?\n" and (i + 1 < len(text)
                                       and text[i + 1] in " \n"):
            start = i + 1
            break
    end = len(text)
    for j in range(pos, len(text)):
        if text[j] in ".!?\n" and (j + 1 >= len(text)
                                      or text[j + 1] in " \n"):
            end = j + 1
            break
    return text[start:end].strip()


# ----------------------------------------------------------------------
# Top-level: link a paper
# ----------------------------------------------------------------------

def link_figures(paper: Dict[str, Any], md_text: str,
                 max_context_chars: int = 280,
                 rescue_orphans: bool = True) -> Dict[str, Any]:
    """Mutate ``paper`` in place, attaching ``referenced_in`` lists.

    When ``rescue_orphans=True`` (default), body mentions that don't
    match a captioned figure are rescued via fig-id mapping: if the
    extractor named the images sequentially (fig-001..fig-NNN), we
    assume those IDs map to "Figure 1..N" in the body. This handles
    the common case where the caption-pairing step missed some
    figures but the body still references them.

    Orphan-rescue is conservative: it only fires when the figure
    record has NO caption_number (i.e. it's a placeholder), so we
    never overwrite a confident caption-number link.
    """
    figs = paper.get("figures") or []
    if not figs:
        return paper

    # Build {fig_number: figure_record} from caption_number first
    by_num: Dict[int, Dict[str, Any]] = {}
    for f in figs:
        n_raw = f.get("caption_number")
        if n_raw is None:
            continue
        try:
            n = int(re.match(r"\d+", str(n_raw)).group(0))
        except Exception:
            continue
        by_num[n] = f

    # ALSO build a fallback index from fig-NNN ids (orphan rescue).
    # Only fill slots not already taken by a real caption_number.
    by_id_num: Dict[int, Dict[str, Any]] = {}
    if rescue_orphans:
        for f in figs:
            fid = f.get("id", "")
            m = re.match(r"fig[-_]?0*(\d+)", fid, re.IGNORECASE)
            if not m:
                continue
            n = int(m.group(1))
            if n in by_num:
                continue        # already a real captioned figure
            if (f.get("caption_number") or "").strip():
                continue
            by_id_num.setdefault(n, f)

    paras = split_paragraphs(md_text)

    # Reset referenced_in on each figure for idempotency
    for f in figs:
        f["referenced_in"] = []

    total_mentions = 0
    rescued_mentions = 0
    cross_caption_mentions = 0
    for idx, para in enumerate(paras):
        mentions = find_mentions(para)
        for (start, end, nums) in mentions:
            sentence = extract_sentence(para, start)
            if len(sentence) > max_context_chars:
                sentence = sentence[: max_context_chars - 1] + "…"
            for n in nums:
                fig = by_num.get(n)
                rescued = False
                if not fig:
                    fig = by_id_num.get(n)
                    rescued = fig is not None
                if not fig:
                    continue
                entry: Dict[str, Any] = {
                    "para_idx": idx,
                    "sentence": sentence,
                    "matched": para[start:end],
                }
                if rescued:
                    entry["rescued_via_id"] = True
                    rescued_mentions += 1
                fig["referenced_in"].append(entry)
                total_mentions += 1

    # Cross-caption rescue: a caption like "Fig 3. Same as Fig 2 but for..."
    # has a body-text reference to Fig 2 embedded in the caption itself.
    # We attach those as `cross_caption_referenced_in` on the OTHER figure.
    for src_fig in figs:
        src_cap_num = src_fig.get("caption_number")
        src_cap_text = src_fig.get("caption_text") or ""
        if not src_cap_text:
            continue
        try:
            src_n = int(re.match(r"\d+", str(src_cap_num)).group(0)) \
                if src_cap_num is not None else None
        except Exception:
            src_n = None
        # find_mentions returns offsets into src_cap_text
        for (s, e, nums) in find_mentions(src_cap_text):
            for n in nums:
                if n == src_n:
                    continue  # self-reference, skip
                target = by_num.get(n) or by_id_num.get(n)
                if not target:
                    continue
                # Avoid duplicating mentions already linked from body
                already = any(r.get("source_caption_of") == src_fig.get("id")
                              for r in target.get("referenced_in", []))
                if already:
                    continue
                snippet = src_cap_text[max(0, s - 30): min(len(src_cap_text), e + 80)]
                target.setdefault("referenced_in", []).append({
                    "para_idx": None,
                    "source_caption_of": src_fig.get("id"),
                    "sentence": " ".join(snippet.split()),
                    "matched": src_cap_text[s:e],
                })
                cross_caption_mentions += 1
                total_mentions += 1

    paper["figure_references_summary"] = {
        "n_figures": len(figs),
        "n_figures_with_mentions": sum(
            1 for f in figs if f.get("referenced_in")),
        "total_mentions": total_mentions,
        "rescued_mentions": rescued_mentions,
        "cross_caption_mentions": cross_caption_mentions,
        "by_figure": {
            str(f.get("caption_number")
                  or f.get("id") or "?"): len(f.get("referenced_in", []))
            for f in figs
        },
    }
    return paper


def link_paper_dir(paper_dir: Path) -> Dict[str, Any]:
    """Load paper.json + paper.md, link them, write back paper.json."""
    pj = paper_dir / "paper.json"
    md = paper_dir / "paper.md"
    if not pj.exists() or not md.exists():
        return {"slug": paper_dir.name, "skipped": "missing files"}
    paper = json.loads(pj.read_text(encoding="utf-8"))
    md_text = md.read_text(encoding="utf-8")
    link_figures(paper, md_text)
    pj.write_text(json.dumps(paper, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    summary = paper.get("figure_references_summary", {})
    return {"slug": paper_dir.name, **summary}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--paper", type=Path, default=None,
                    help="Run on a single paper dir instead of the whole corpus")
    args = p.parse_args(argv)

    if args.paper:
        targets = [args.paper]
    else:
        targets = [d for d in sorted(args.output_dir.iterdir())
                   if d.is_dir() and not d.name.startswith("_")]

    n_ok = 0
    for t in targets:
        res = link_paper_dir(t)
        if "skipped" in res:
            print(f"SKIP {res['slug']}: {res['skipped']}")
            continue
        n_ok += 1
        print(f"{res['slug']}: {res.get('total_mentions', 0)} mentions across "
              f"{res.get('n_figures_with_mentions', 0)}/{res.get('n_figures', 0)} figs")
    print(f"linked {n_ok} papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
