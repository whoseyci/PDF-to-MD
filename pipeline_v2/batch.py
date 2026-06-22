"""Batch convert all unique PDFs using the v2 pipeline."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import hashlib
from convert import convert_pdf

PDF_DIR = Path("/home/user/pdfs")
OUT_ROOT = Path("/home/user/output")


def dedupe_by_content():
    """Return list of canonical PDF Paths, one per content-hash group."""
    by_hash = {}
    for p in sorted(PDF_DIR.glob("*.pdf")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        if h not in by_hash:
            by_hash[h] = p
    return list(by_hash.values())


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    pdfs = dedupe_by_content()
    print(f"Converting {len(pdfs)} unique PDFs (by content hash)")
    print(f"Output: {OUT_ROOT}\n")
    
    t0 = time.time()
    results = []
    failures = []
    for i, p in enumerate(pdfs, 1):
        print(f"[{i:>2}/{len(pdfs)}]", end=" ")
        try:
            r = convert_pdf(p, OUT_ROOT)
            results.append(r)
        except Exception as e:
            print(f"   FAIL on {p.name}: {e}")
            import traceback
            traceback.print_exc()
            failures.append({"file": p.name, "error": str(e)})
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Batch done in {elapsed/60:.1f} min ({elapsed/len(pdfs):.1f}s avg)")
    print(f"  Success: {sum(1 for r in results if not r.get('skipped'))}")
    print(f"  Cached:  {sum(1 for r in results if r.get('skipped'))}")
    print(f"  Failed:  {len(failures)}")
    
    # Aggregate
    confs = {"high": 0, "medium": 0, "low": 0}
    n_refs = n_cites = n_figs = 0
    for r in results:
        if r.get("skipped"):
            continue
        s = r.get("stats", {})
        confs[s.get("confidence", "low")] += 1
        n_refs += s.get("n_references", 0)
        n_cites += s.get("n_citations_linked", 0)
        n_figs += s.get("n_figures", 0)
    print(f"  Confidence: {confs}")
    print(f"  Aggregate: {n_refs} refs, {n_cites} citations linked, {n_figs} figures")
    
    (OUT_ROOT / "_batch_report.json").write_text(json.dumps({
        "elapsed_seconds": elapsed,
        "results": results,
        "failures": failures,
        "totals": {"refs": n_refs, "cites": n_cites, "figs": n_figs, "confs": confs},
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
