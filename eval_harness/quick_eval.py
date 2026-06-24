"""Quick diagnostic: run a few extractors on a few papers."""
import sys, time
sys.path.insert(0, "/home/user")
from pathlib import Path
from eval_harness.run_eval import (
    extract_pymupdf4llm, extract_pdftotext_simple,
    extract_pipeline_v2_with_reorder, extract_pipeline_v2_auto,
    extract_pipeline_v2_auto_rotfix,
    normalise, precision_recall_words, fast_wer_estimate)


def main():
    for pdf_name in ["1706.03762", "1810.04805", "1406.2661", "1503.02531"]:
        pdf = Path(f"/home/user/eval_harness/corpus/{pdf_name}/paper.pdf")
        if not pdf.exists():
            continue
        gt = normalise(Path(f"/home/user/eval_harness/corpus/{pdf_name}/ground_truth.txt").read_text(errors="replace"))
        print(f"\n== {pdf_name}")
        for name, fn in [
            ("pdftotext-stream", extract_pdftotext_simple),
            ("pymupdf4llm", extract_pymupdf4llm),
            ("reorder+dehyph NEW", extract_pipeline_v2_with_reorder),
            ("auto NEW", extract_pipeline_v2_auto),
        ]:
            t0 = time.time()
            out = normalise(fn(pdf))
            e = time.time() - t0
            pr = precision_recall_words(out, gt)
            wer = fast_wer_estimate(out, gt)
            f1_str = "{:.4f}".format(pr["f1"])
            wer_str = "{:.4f}".format(wer)
            print(f"  {name:25s} F1={f1_str} WER={wer_str} t={e:.1f}s chars={len(out)}")


if __name__ == "__main__":
    main()
