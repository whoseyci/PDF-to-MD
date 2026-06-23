"""
Smoke tests for the optional-integration modules.

Run:
    python3 -m tests.test_integrations
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class _T:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errs = []

    def check(self, cond, msg):
        if cond:
            self.passed += 1
        else:
            self.failed += 1
            self.errs.append(msg)

    def report(self):
        print(f"\n{self.passed} passed, {self.failed} failed")
        for e in self.errs:
            print(f"  FAIL: {e}")
        return 0 if self.failed == 0 else 1


# --------------------------------------------------------------------

def test_refextract_bridge(t):
    from pipeline_v2.refextract_bridge import is_available, _norm_prefix
    t.check(callable(is_available), "is_available is callable")
    t.check(_norm_prefix("Foo, B., 2020. Title. Journal 1, 1-10.")[:5] == "foob2",
            f"_norm_prefix produces correct key: {_norm_prefix('Foo, B., 2020. Title. Journal 1, 1-10.')[:5]!r}")


def test_ref_verifier(t):
    from pipeline_v2.ref_verifier import (_title_overlap_score,
                                            _extract_title_from_ref,
                                            _looks_like_journal_citation)
    # Title overlap
    s1 = _title_overlap_score("Deep learning for natural language",
                                "Deep learning of natural language understanding")
    t.check(0.4 < s1 < 1.0,
            f"title overlap reasonable: got {s1:.2f}")

    # Journal-like detection
    t.check(_looks_like_journal_citation("J. Bio. 99, 12345"),
            "detects journal-like citation")
    t.check(not _looks_like_journal_citation(
        "Distributed representations of words and phrases"),
            "doesn't flag a real title as journal-like")

    # Title extraction from a synthetic ref
    ref = {"raw": "Smith J., Jones K., 2020. A study of widget durability "
             "under stress. Mater. Sci. 42 (3), 1234-1245."}
    title = _extract_title_from_ref(ref)
    t.check(title and "widget" in title.lower(),
            f"extracts a plausible title: {title!r}")


def test_docling_export(t):
    from pipeline_v2.docling_export import (to_docling_document,
                                              validate_with_docling)
    paper = {"title": "Test",
              "authors": ["A B"],
              "abstract": "Brief abstract.",
              "sections": [{"heading": "Intro", "body": "Intro body.", "page": 1}],
              "references": [{"id": "1", "raw": "Foo et al. 2020. X. J. 1, 1-10."}],
              "figures": [{"id": "fig-001", "file": "f.png", "page": 2,
                            "caption_text": "Cap.", "alt_text": "alt"}]}
    doc = to_docling_document(paper)
    t.check(doc["schema_name"] == "DoclingDocument",
            f"schema_name set correctly")
    t.check(len(doc["texts"]) >= 5, f"has >= 5 texts (got {len(doc['texts'])})")
    t.check(len(doc["pictures"]) == 1, f"has 1 picture")

    v = validate_with_docling(doc)
    if v["ok"] is None:
        print("  (docling_core not installed; validation skipped)")
    else:
        err = v.get("error") or "ok"
        t.check(v["ok"], f"validates: {err[:200]}")


def test_figure_prompts(t):
    from pipeline_v2.vision.figure_prompts import (prompt_for_caption,
                                                      postprocess_response)
    cases = [
        ("Algorithm 1: Iterative refinement", "algorithm"),
        ("Listing 2: Python implementation of the foo function", "code_listing"),
        ("Equation 3.4: gradient update rule", "equation"),
        ("Figure 1: SEM image of microbial colonies", "microscopy"),
        ("Western blot showing protein expression", "gel_blot"),
        ("Sankey diagram of energy flows", "sankey"),
        ("Decision tree for triage classification", "decision_tree"),
        ("Screenshot of the dashboard", "screenshot"),
        ("Just some random caption", "generic"),
    ]
    for cap, expected in cases:
        p = prompt_for_caption(cap)
        t.check(p.subkind == expected,
                f"caption {cap!r} -> {p.subkind!r}, expected {expected!r}")

    # postprocess strips preamble
    raw = "Here is the code:\n```python\nprint('hi')\n```\nThanks!"
    out = postprocess_response(raw, "code")
    t.check(out.startswith("```python") and out.endswith("```"),
            f"postprocess strips prose: got {out!r}")


def test_llm_boost(t):
    from pipeline_v2.llm_boost import (decide_for_figure, decide_for_reference,
                                          decide_for_paragraph, estimate_corpus_cost)
    d1 = decide_for_figure({"status": "ok", "confidence": 0.9})
    t.check(d1.decision == "validate", f"high-conf OK → validate (got {d1.decision})")

    d2 = decide_for_figure({"status": "unsupported", "confidence": 0.0})
    t.check(d2.decision == "extract", f"unsupported → extract (got {d2.decision})")

    d3 = decide_for_reference({"raw": "abc def ghi",
                                  "verification": {"verdict": "verified"}})
    t.check(d3.decision == "skip", f"verified → skip (got {d3.decision})")

    d4 = decide_for_paragraph("This is perfectly fine text with normal spacing.")
    t.check(d4.decision == "skip", f"clean para → skip (got {d4.decision})")

    d5 = decide_for_paragraph("ﬁ ﬂ ad hoc \u00ad\u00ad junkly te xt with garbʄle ## bro k en")
    t.check(d5.decision in ("validate", "replace"),
            f"messy para → validate/replace (got {d5.decision})")

    summary = estimate_corpus_cost([d1, d2, d3, d4, d5])
    t.check(summary["n_blocks"] == 5, "estimates cost over 5 blocks")
    t.check("total_estimated_seconds" in summary, "summary has total time")


def test_deplot_extractor_lazy(t):
    """Just check it imports and instantiates -- no real model load."""
    try:
        from pipeline_v2.vision.chart_extract.deplot import DeplotExtractor
        ext = DeplotExtractor()
        t.check(ext.name == "deplot/v1", "deplot extractor has correct name")
        t.check(ext._model is None, "deplot model is lazy-loaded")
    except ImportError:
        print("  (transformers not installed; deplot test skipped)")


def test_diagram_extract(t):
    """Classical (non-LLM) diagram → mermaid extractor.

    Generates a synthetic 5-node 4-edge diagram with matplotlib,
    runs the extractor, asserts node count + node labels + edge count.
    """
    import tempfile
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("  (matplotlib not installed; diagram test skipped)")
        return

    from pipeline_v2.vision.diagram_extract import extract_diagram
    with tempfile.TemporaryDirectory() as td:
        img = Path(td) / "diagram.png"
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')

        def box(x, y, w, h, text, color="#dbe9f4"):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                            linewidth=2, edgecolor="black",
                                            facecolor=color))
            ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                    fontsize=12, fontweight="bold")

        def arrow(x1, y1, x2, y2):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", lw=2, color="black"))

        box(0.2, 4.0, 2.8, 0.8, "Attitude")
        box(0.2, 2.6, 2.8, 0.8, "SubNorm")
        box(0.2, 1.2, 2.8, 0.8, "PBC")
        box(4.0, 2.6, 2.2, 0.8, "Intention", color="#f9e1c8")
        box(7.2, 2.6, 2.5, 0.8, "Behaviour", color="#d8f0d3")
        arrow(3.0, 4.4, 4.0, 3.0)
        arrow(3.0, 3.0, 4.0, 3.0)
        arrow(3.0, 1.6, 4.0, 3.0)
        arrow(6.2, 3.0, 7.2, 3.0)
        fig.tight_layout(); fig.savefig(img); plt.close(fig)

        r = extract_diagram(img)
        t.check(r.status == "ok",
                f"diagram extractor returns ok (got {r.status}: {r.reason})")
        t.check(len(r.nodes) == 5,
                f"detects 5 nodes (got {len(r.nodes)})")
        labels = sorted(n.label.lower() for n in r.nodes)
        # At least the 5 distinct words should appear in the extracted labels
        for keyword in ("attitude", "subnorm", "pbc", "intention", "behaviour"):
            t.check(any(keyword in lab for lab in labels),
                    f"detects label containing {keyword!r} (got {labels})")
        t.check(len(r.edges) >= 3,
                f"detects >= 3 edges (got {len(r.edges)})")
        t.check(r.mermaid and "flowchart" in r.mermaid,
                f"emits valid mermaid block")
        print(f"  diagram: {len(r.nodes)} nodes, {len(r.edges)} edges, "
              f"conf={r.confidence}")


def test_cascading_extractor(t):
    """Cascade should short-circuit on a high-confidence hit."""
    from pipeline_v2.vision.chart_extract.multi_extractor import CascadingExtractor
    from pipeline_v2.vision.chart_extract.base import (ChartExtractor,
                                                          ChartExtractionResult,
                                                          ExtractionStatus)

    class FakeOK(ChartExtractor):
        name = "fake-ok"
        def extract(self, img, **kw):
            return ChartExtractionResult(
                extractor=self.name, status=ExtractionStatus.OK,
                confidence=0.95, categories=["A"], values=[1.0])

    class FakeFail(ChartExtractor):
        name = "fake-fail"
        def extract(self, img, **kw):
            return ChartExtractionResult(
                extractor=self.name, status=ExtractionStatus.NO_AXIS,
                confidence=0.0, reason="testing")

    # OK then Fail -> should short-circuit, never run Fail
    cascade = CascadingExtractor([FakeOK(), FakeFail()])
    r = cascade.extract(Path("/dev/null"))
    t.check(r.status.value == "ok",
            f"cascade short-circuits on OK (got {r.status.value})")
    t.check("cascade(" in r.extractor,
            f"cascade name propagated (got {r.extractor!r})")
    cascade_results = (r.extracted_data or {}).get("cascade_results", [])
    t.check(len(cascade_results) == 1,
            f"only first extractor ran ({len(cascade_results)} results)")

    # Fail then OK -> should run both, end with OK
    cascade2 = CascadingExtractor([FakeFail(), FakeOK()])
    r2 = cascade2.extract(Path("/dev/null"))
    t.check(r2.status.value == "ok",
            f"cascade falls through on fail (got {r2.status.value})")
    cascade_results2 = (r2.extracted_data or {}).get("cascade_results", [])
    t.check(len(cascade_results2) == 2,
            f"both extractors ran ({len(cascade_results2)} results)")


def main():
    t = _T()
    print("=== refextract_bridge ==="); test_refextract_bridge(t)
    print("=== ref_verifier ==="); test_ref_verifier(t)
    print("=== docling_export ==="); test_docling_export(t)
    print("=== figure_prompts ==="); test_figure_prompts(t)
    print("=== llm_boost ==="); test_llm_boost(t)
    print("=== deplot extractor ==="); test_deplot_extractor_lazy(t)
    print("=== diagram extract ==="); test_diagram_extract(t)
    print("=== cascading extractor ==="); test_cascading_extractor(t)
    return t.report()


if __name__ == "__main__":
    sys.exit(main())
