"""Single-file static HTML browser for a `output/` corpus.

Walks the existing per-paper output and emits an
``output/index.html`` you can open in any browser (no server needed,
no JS frameworks, no external assets):

  * Left sidebar -- searchable list of papers
  * Right pane  -- selected paper's rendered markdown + figures gallery
                    + reference table with verified-DOI badges
  * Click a `[ref-NNN]` style citation to scroll to its bibliography entry
  * Inline figure expand-on-click

Self-contained: CSS, JS, all data baked into the single HTML file.
"""
from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Minimal markdown → HTML
# ----------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HR_RE = re.compile(r"^\s*---+\s*$", re.MULTILINE)


def md_to_html(md: str) -> str:
    """Tiny home-grown markdown renderer (no external deps).
    Handles: headings, paragraphs, bold, italic, links, images, hrules,
    fenced code blocks. Does NOT do tables (we render them as raw <pre>)
    to keep the renderer small."""
    # Escape HTML first
    md = html_lib.escape(md)

    # Restore quotes within links so they render
    # (we only escaped & < > to be conservative)

    # Headings
    def _h(m):
        n = len(m.group(1))
        text = m.group(2).strip()
        return f"<h{n}>{text}</h{n}>"
    md = _HEADING_RE.sub(_h, md)

    # Horizontal rules
    md = _HR_RE.sub("<hr/>", md)

    # Images first, then links
    md = _IMG_RE.sub(
        lambda m: f'<img class="fig" src="{m.group(2)}" alt="{m.group(1)}"/>',
        md,
    )
    md = _LINK_RE.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', md)

    # Bold/italic
    md = _BOLD_RE.sub(r"<strong>\1</strong>", md)
    md = _ITALIC_RE.sub(r"<em>\1</em>", md)

    # Paragraph splitter: blank-line separated
    paragraphs = re.split(r"\n\s*\n", md)
    out = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # If it's already a tag (heading / hr / table), keep raw
        if p.startswith(("<h", "<hr", "<table", "<pre", "<img", "|")):
            out.append(p)
        else:
            # Single line breaks → <br/>
            out.append("<p>" + p.replace("\n", "<br/>") + "</p>")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Paper bundle (one per output/<slug>/)
# ----------------------------------------------------------------------

@dataclass
class PaperBundle:
    slug: str
    title: str
    md_html: str
    figures: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    stats: Dict[str, Any]


def _img_data_url(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        data = p.read_bytes()
    except Exception:
        return None
    if len(data) > 800_000:        # skip embedding huge images
        return None
    ext = p.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg",
             "jpeg": "image/jpeg", "gif": "image/gif",
             "webp": "image/webp"}.get(ext, "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def collect_paper(paper_dir: Path) -> Optional[PaperBundle]:
    md_path = paper_dir / "paper.md"
    if not md_path.exists():
        return None
    try:
        md = md_path.read_text(encoding="utf-8")
    except Exception:
        md = ""
    stats = {}
    try:
        stats = json.loads(
            (paper_dir / "stats.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    paper = {}
    try:
        paper = json.loads(
            (paper_dir / "paper.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    refs = []
    try:
        refs = json.loads(
            (paper_dir / "references.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    # Replace ![…](./figures/…png) with data URLs so the static page
    # is fully self-contained.
    def _embed(m):
        alt, src = m.group(1), m.group(2)
        src_path = (paper_dir / src).resolve()
        url = _img_data_url(src_path)
        if url:
            return f"![{alt}]({url})"
        return m.group(0)
    md = _IMG_RE.sub(_embed, md)

    md_html = md_to_html(md)

    # Build the figures gallery
    figs = []
    for f in paper.get("figures", []):
        fpath = paper_dir / f.get("file", "")
        url = _img_data_url(fpath) if fpath.exists() else None
        figs.append({
            "id": f.get("id"),
            "caption_number": f.get("caption_number"),
            "caption_text": f.get("caption_text") or "",
            "url": url,
            "n_refs": len(f.get("referenced_in", []) or []),
        })

    title = (stats.get("title") or
              paper.get("metadata", {}).get("title") or
              paper_dir.name)
    return PaperBundle(
        slug=paper_dir.name,
        title=title,
        md_html=md_html,
        figures=figs,
        references=refs or [],
        stats=stats,
    )


# ----------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PDF-to-MD corpus browser</title>
<style>
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #222; }
  .wrap { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
  aside { background: #f3f5f8; border-right: 1px solid #d8dde4; overflow: auto; height: 100vh; position: sticky; top: 0; }
  aside h1 { font-size: 14px; padding: 14px 16px 6px; margin: 0; color: #444; }
  #search { width: calc(100% - 32px); margin: 6px 16px 10px; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; }
  .paperlist { list-style: none; padding: 0; margin: 0; }
  .paperlist li { padding: 6px 16px; cursor: pointer; border-left: 3px solid transparent; font-size: 13px; }
  .paperlist li:hover { background: #e8edf3; }
  .paperlist li.active { background: #dfe7f1; border-left-color: #3667a8; font-weight: 600; }
  main { padding: 24px 36px; max-width: 900px; }
  main h1 { font-size: 22px; line-height: 1.3; }
  main h2 { font-size: 18px; margin-top: 28px; border-bottom: 1px solid #eee; padding-bottom: 4px; }
  main img.fig { max-width: 100%; cursor: zoom-in; border: 1px solid #e3e3e3; margin: 12px 0; }
  main img.zoomed { max-width: none; cursor: zoom-out; }
  table.refs { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.refs td { padding: 4px 8px; vertical-align: top; border-bottom: 1px solid #eee; }
  table.refs td.idcol { width: 60px; color: #888; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; margin-left: 6px; }
  .badge.verified { background: #d8efd8; color: #205020; }
  .badge.unverified { background: #f3e2c8; color: #6a4a10; }
  .stats { background: #fafbfc; padding: 8px 14px; border: 1px solid #e1e4e8; border-radius: 6px; font-size: 13px; margin: 12px 0; }
  .figcard { display: inline-block; margin: 8px; vertical-align: top; max-width: 220px; }
  .figcard img { max-width: 100%; border: 1px solid #ddd; }
  .figcard .cap { font-size: 11px; color: #555; padding: 4px; }
  .empty { color: #888; padding: 28px 0; text-align: center; }
  hr { border: 0; border-top: 1px solid #ddd; margin: 24px 0; }
</style>
</head>
<body>
<div class="wrap">
  <aside>
    <h1>PDF-to-MD corpus (N=__N__)</h1>
    <input id="search" placeholder="filter…"/>
    <ul class="paperlist" id="list"></ul>
  </aside>
  <main id="paper"><div class="empty">Select a paper from the list.</div></main>
</div>
<script>
const PAPERS = __PAPERS_JSON__;
const list = document.getElementById('list');
const main = document.getElementById('paper');
const search = document.getElementById('search');
function renderList(q) {
  list.innerHTML = '';
  q = (q||'').toLowerCase();
  PAPERS.forEach((p, i) => {
    if (q && p.title.toLowerCase().indexOf(q) < 0 && p.slug.indexOf(q) < 0) return;
    const li = document.createElement('li');
    li.textContent = p.title;
    li.title = p.slug;
    li.onclick = () => select(i, li);
    list.appendChild(li);
  });
}
function select(i, li) {
  document.querySelectorAll('.paperlist li').forEach(x => x.classList.remove('active'));
  if (li) li.classList.add('active');
  const p = PAPERS[i];
  const stats = p.stats || {};
  let html = '<div class="stats">';
  for (const k of ['n_pages','n_figures','n_references','n_citations_linked','coverage_ratio','confidence']) {
    if (stats[k] != null) html += '<strong>'+k+':</strong> '+stats[k]+' &nbsp; ';
  }
  html += '</div>';
  html += p.md_html;
  if (p.figures && p.figures.length) {
    html += '<h2>Figures (' + p.figures.length + ')</h2>';
    p.figures.forEach(f => {
      if (!f.url) return;
      html += '<div class="figcard">';
      html += '<img src="'+f.url+'"/>';
      html += '<div class="cap"><strong>Fig '+(f.caption_number||'?')+'</strong> ('+f.n_refs+' refs in body)<br/>'
              + (f.caption_text || '').slice(0, 140) + '</div>';
      html += '</div>';
    });
  }
  if (p.references && p.references.length) {
    html += '<h2>References ('+p.references.length+')</h2>';
    html += '<table class="refs">';
    p.references.forEach(r => {
      html += '<tr>';
      html += '<td class="idcol">'+(r.id||'')+'</td>';
      html += '<td>' + (r.text || '');
      if (r.doi) html += ' <span class="badge verified">DOI</span>';
      else html += ' <span class="badge unverified">no DOI</span>';
      html += '</td></tr>';
    });
    html += '</table>';
  }
  main.innerHTML = html;
  // Wire up image zoom
  main.querySelectorAll('img.fig').forEach(img => {
    img.onclick = () => img.classList.toggle('zoomed');
  });
  // Smooth-scroll for #ref-NNN anchors
  main.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const id = a.getAttribute('href').substring(1);
      const el = main.querySelector('[id="'+id+'"]');
      if (el) { e.preventDefault(); el.scrollIntoView({behavior:'smooth'}); }
    });
  });
}
search.oninput = e => renderList(e.target.value);
renderList('');
</script>
</body></html>
"""


def build_corpus_browser(output_dir: Path,
                          out_file: Optional[Path] = None) -> Path:
    out_file = out_file or output_dir / "index.html"
    bundles: List[PaperBundle] = []
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        b = collect_paper(d)
        if b is not None:
            bundles.append(b)
    papers_json = []
    for b in bundles:
        papers_json.append({
            "slug": b.slug,
            "title": b.title,
            "md_html": b.md_html,
            "stats": b.stats,
            "figures": b.figures,
            "references": b.references,
        })
    html = (_HTML_TEMPLATE
            .replace("__N__", str(len(bundles)))
            .replace("__PAPERS_JSON__",
                       json.dumps(papers_json, ensure_ascii=False)))
    out_file.write_text(html, encoding="utf-8")
    return out_file


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    out = build_corpus_browser(args.output_dir, args.out)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
