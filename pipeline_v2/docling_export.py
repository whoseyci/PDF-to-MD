"""
DoclingDocument-compatible export.

We emit JSON with the same structure as `DS4SD/docling`'s
`DoclingDocument` so downstream RAG tooling (LlamaIndex's
`DoclingReader`, LangChain, etc.) can consume our output without
us taking the full docling dependency.

If `docling_core` (lightweight, schemas only) OR the full `docling`
is installed, we additionally round-trip the doc through
`DoclingDocument.model_validate(...)` to guarantee strict compatibility.

The schema is strict about types: bboxes must be real objects (not
null), page_no must be int (not null), and labels come from a fixed
enum. We use safe defaults throughout so the doc is always valid even
when we don't know the exact source coordinates.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# Safe placeholder bbox -- represents "unknown source coordinates" in
# the bottom-left letterbox area, well outside any real chart.
_DEFAULT_BBOX = {"l": 0.0, "t": 0.0, "r": 0.0, "b": 0.0,
                  "coord_origin": "BOTTOMLEFT"}


def _prov(page_no: Optional[int], charspan: tuple = (0, 0)) -> Dict[str, Any]:
    return {
        "page_no": int(page_no) if page_no else 1,
        "bbox": dict(_DEFAULT_BBOX),
        "charspan": list(charspan),
    }


# --------------------------------------------------------------------
# Build the document
# --------------------------------------------------------------------

class _Counters:
    """Counters for self-refs across one document build."""
    def __init__(self):
        self.text = 0
        self.picture = 0
        self.table = 0
        self.group = 0

    def text_ref(self) -> str:
        r = f"#/texts/{self.text}"; self.text += 1; return r

    def picture_ref(self) -> str:
        r = f"#/pictures/{self.picture}"; self.picture += 1; return r

    def group_ref(self) -> str:
        r = f"#/groups/{self.group}"; self.group += 1; return r


def to_docling_document(paper: Dict[str, Any],
                         *,
                         pdf_path: Optional[Path] = None,
                         markdown: Optional[str] = None
                         ) -> Dict[str, Any]:
    """Build a strict-validating DoclingDocument-compatible dict."""
    name = paper.get("title") or (pdf_path.stem if pdf_path else "document")
    origin = _build_origin(pdf_path)

    counters = _Counters()
    texts: List[Dict[str, Any]] = []
    pictures: List[Dict[str, Any]] = []
    body_children: List[Dict[str, str]] = []

    def add_text(label: str, text: str, *,
                  page_no: Optional[int] = None,
                  charspan: tuple = (0, 0)) -> str:
        ref = counters.text_ref()
        texts.append({
            "self_ref": ref,
            "parent": {"$ref": "#/body"},
            "children": [],
            "content_layer": "body",
            "label": label,
            "prov": [_prov(page_no, charspan)],
            "orig": text or "",
            "text": text or "",
        })
        body_children.append({"$ref": ref})
        return ref

    def add_picture(fig: Dict[str, Any]) -> str:
        # Caption is a SEPARATE text item that we reference.
        caption_ref = None
        if fig.get("caption_text"):
            caption_ref = counters.text_ref()
            texts.append({
                "self_ref": caption_ref,
                "parent": {"$ref": "#/body"},
                "children": [],
                "content_layer": "body",
                "label": "caption",
                "prov": [_prov(fig.get("page"),
                                 (0, len(fig["caption_text"])))],
                "orig": fig["caption_text"],
                "text": fig["caption_text"],
            })
        pic_ref = counters.picture_ref()
        annotations = []
        # Bundle our alt-text, mermaid, etc. as "description" annotations
        # since that's a recognized Docling annotation kind.
        if fig.get("alt_text"):
            annotations.append({
                "kind": "description",
                "text": fig["alt_text"],
                "provenance": "pdf-to-md.vision/alt_text",
            })
        if fig.get("markdown_table"):
            annotations.append({
                "kind": "description",
                "text": "Extracted data table:\n\n" + fig["markdown_table"],
                "provenance": "pdf-to-md.chart_extract",
            })
        if fig.get("mermaid"):
            annotations.append({
                "kind": "description",
                "text": fig["mermaid"],
                "provenance": "pdf-to-md.mermaid_extract",
            })
        pic = {
            "self_ref": pic_ref,
            "parent": {"$ref": "#/body"},
            "children": [],
            "content_layer": "body",
            "label": "picture",
            "prov": [_prov(fig.get("page"), (0, 0))],
            "captions": [{"$ref": caption_ref}] if caption_ref else [],
            "references": [],
            "footnotes": [],
            "annotations": annotations,
        }
        pictures.append(pic)
        body_children.append({"$ref": pic_ref})
        return pic_ref

    # Title
    if paper.get("title"):
        add_text("title", paper["title"], page_no=1,
                  charspan=(0, len(paper["title"])))

    # Authors as a regular text block
    if paper.get("authors"):
        au_text = "; ".join(paper["authors"]) if isinstance(paper["authors"], list) \
            else str(paper["authors"])
        if au_text:
            add_text("text", au_text, page_no=1, charspan=(0, len(au_text)))

    # Abstract
    if paper.get("abstract"):
        add_text("section_header", "Abstract", page_no=1,
                  charspan=(0, len("Abstract")))
        add_text("text", paper["abstract"], page_no=1,
                  charspan=(0, len(paper["abstract"])))

    # Sections -- prefer markdown split on headings if given
    if markdown:
        for blk in _split_markdown_blocks(markdown):
            label = blk["label"]
            # Docling needs specific label literals
            if label == "section_header":
                add_text("section_header", blk["text"],
                          page_no=blk.get("page"),
                          charspan=(0, len(blk["text"])))
            else:
                add_text("text", blk["text"],
                          page_no=blk.get("page"),
                          charspan=(0, len(blk["text"])))
    else:
        for sect in (paper.get("sections") or []):
            heading = sect.get("heading") or sect.get("title")
            if heading:
                add_text("section_header", heading,
                          page_no=sect.get("page"),
                          charspan=(0, len(heading)))
            body = sect.get("body") or sect.get("text")
            if body:
                add_text("text", body, page_no=sect.get("page"),
                          charspan=(0, len(body)))

    # Figures
    for fig in (paper.get("figures") or []):
        add_picture(fig)

    # References
    refs = paper.get("references") or []
    if refs:
        add_text("section_header", "References", page_no=1,
                  charspan=(0, len("References")))
        for ref in refs:
            raw = ref.get("raw") or ref.get("text") or ""
            if not raw: continue
            add_text("reference", raw, page_no=None,
                      charspan=(0, len(raw)))

    doc: Dict[str, Any] = {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": name,
        "origin": origin,
        "furniture": {
            "self_ref": "#/furniture",
            "children": [],
            "content_layer": "furniture",
            "name": "_root_",
            "label": "unspecified",
        },
        "body": {
            "self_ref": "#/body",
            "children": body_children,
            "content_layer": "body",
            "name": "_root_",
            "label": "unspecified",
        },
        "groups": [],
        "texts": texts,
        "tables": [],
        "pictures": pictures,
        "key_value_items": [],
        "form_items": [],
        "pages": {},
    }
    return doc


# --------------------------------------------------------------------
# Optional validation pass
# --------------------------------------------------------------------

def validate_with_docling(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Validate via DoclingDocument if `docling_core` is available."""
    DoclingDocument = None
    version_str = None
    try:
        from docling_core.types.doc.document import DoclingDocument  # type: ignore
        import docling_core  # type: ignore
        version_str = getattr(docling_core, "__version__", "?")
    except ImportError:
        try:
            from docling.datamodel.document import DoclingDocument  # type: ignore
            import docling  # type: ignore
            version_str = getattr(docling, "__version__", "?")
        except ImportError:
            return {"ok": None,
                     "error": "neither docling nor docling_core is installed",
                     "docling_version": None}
    try:
        DoclingDocument.model_validate(doc)
        return {"ok": True, "error": None, "docling_version": version_str}
    except Exception as e:
        err = str(e)
        # Truncate huge multi-error messages
        if len(err) > 500:
            err = err[:500] + "... (truncated)"
        return {"ok": False, "error": f"{type(e).__name__}: {err}",
                 "docling_version": version_str}


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _build_origin(pdf_path: Optional[Path]) -> Dict[str, Any]:
    """Origin block. binary_hash MUST be a valid sha256 hexdigest."""
    if pdf_path is None or not pdf_path.exists():
        # Use the sha256 of an empty string when there's no source PDF
        empty_sha = hashlib.sha256(b"").hexdigest()
        return {"mimetype": "application/pdf",
                 "binary_hash": empty_sha,
                 "filename": pdf_path.name if pdf_path else "unknown.pdf"}
    h = hashlib.sha256()
    try:
        with pdf_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
    except OSError:
        pass
    return {"mimetype": "application/pdf",
             "binary_hash": h.hexdigest(),
             "filename": pdf_path.name}


def _split_markdown_blocks(md: str) -> List[Dict[str, Any]]:
    import re
    out: List[Dict[str, Any]] = []
    current_page = None
    page_marker_re = re.compile(
        r"\u2063\u2063\u2063PB(\d+)/(\d+)\u2063\u2063\u2063")
    cur_text: List[str] = []

    def flush(label: str):
        nonlocal cur_text
        chunk = "\n".join(cur_text).strip()
        if chunk:
            out.append({"label": label, "text": chunk, "page": current_page})
        cur_text = []

    for line in md.splitlines():
        m = page_marker_re.search(line)
        if m:
            current_page = int(m.group(1))
            continue
        if re.match(r"^#{1,6}\s+\S", line):
            flush("text")
            out.append({"label": "section_header",
                         "text": line.lstrip("# ").strip(),
                         "page": current_page})
        else:
            cur_text.append(line)
    flush("text")
    return out


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def write_docling_json(paper_json_path: Path, *, out_path: Optional[Path] = None,
                        pdf_path: Optional[Path] = None,
                        markdown_path: Optional[Path] = None,
                        validate: bool = True) -> Path:
    """Read paper.json, emit paper.docling.json, optionally validate."""
    paper = json.loads(paper_json_path.read_text(encoding="utf-8"))
    md = markdown_path.read_text(encoding="utf-8") if markdown_path \
        and markdown_path.exists() else None
    doc = to_docling_document(paper, pdf_path=pdf_path, markdown=md)
    if out_path is None:
        out_path = paper_json_path.with_name(
            paper_json_path.stem + ".docling.json")
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    if validate:
        v = validate_with_docling(doc)
        if v.get("ok") is False:
            log.warning("docling validation failed: %s", v.get("error"))
        elif v.get("ok") is True:
            log.info("docling validation: OK (docling v%s)",
                       v.get("docling_version"))
    return out_path
