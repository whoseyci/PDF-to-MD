"""Diagnose what E1+dehyph is doing wrong vs pdftotext-stream."""
import sys, re
sys.path.insert(0, "/home/user")
from pathlib import Path
import subprocess
from pipeline_v2.reading_order import reorder_pdf_text

def tok(t): return set(re.findall(r"[a-z0-9]+", t.lower()))

for paper in ["1810.04805", "1406.2661", "1503.02531"]:
    pdf = Path(f"/home/user/eval_harness/corpus/{paper}/paper.pdf")
    if not pdf.exists(): continue
    gt = tok(Path(f"/home/user/eval_harness/corpus/{paper}/ground_truth.txt").read_text(errors="replace"))
    pt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    pt_words = tok(pt)
    re_words = tok(reorder_pdf_text(pdf))
    extra_pt = pt_words - re_words
    extra_re = re_words - pt_words
    pt_correct = pt_words & gt
    re_correct = re_words & gt
    print(f"\n=== {paper} ===")
    print(f"  pt_correct={len(pt_correct)} re_correct={len(re_correct)} delta={len(pt_correct)-len(re_correct)}")
    print(f"  in pt not in re: {len(extra_pt)} (correct: {len(extra_pt & gt)})")
    sample = sorted(extra_pt & gt)[:30]
    print(f"  sample correct-only-in-pt: {sample}")
    sample = sorted(extra_re - gt)[:30]
    print(f"  sample wrong-only-in-re: {sample}")
