# Failure-mode catalog

Each row is a synthetic PDF designed to break ONE specific assumption in the pipeline. The cells show which stage tolerated the input.

Legend: ✅ ok · ⚠️ partial (parsed but suspicious) · ❌ crashed

| Failure mode | pages | pymupdf | pymupdf4llm | pdftotext | pdf2md_reorder | caption_pair |
|---|---|---|---|---|---|---|
| F01_encrypted | 1 | ✅ | ❌ | ❌ | ❌ | ❌ |
| F02_empty | 1 | ✅ | ⚠️ 0c | ⚠️ 1c | ⚠️ 0c | ✅ 0pr |
| F03_image_only | 1 | ✅ | ✅ 103c | ⚠️ 1c | ⚠️ 0c | ✅ 0pr |
| F04_rotated | 3 | ✅ | ⚠️ 21c | ✅ 66c | ✅ 61c | ✅ 0pr |
| F05_no_headings | 5 | ✅ | ✅ 17975c | ✅ 17610c | ✅ 17998c | ✅ 0pr |
| F06_midword_break | 2 | ✅ | ✅ 92c | ✅ 92c | ✅ 88c | ✅ 0pr |
| F07_weird_unicode | 1 | ✅ | ✅ 47c | ✅ 44c | ✅ 42c | ✅ 0pr |
| F08_duplicate_pages | 4 | ✅ | ✅ 252c | ✅ 228c | ✅ 226c | ✅ 0pr |
| F09_garbage_bytes | ? | ❌ | ❌ | ❌ | ❌ | ❌ |
| F10_table_only | 1 | ✅ | ✅ 41c | ⚠️ 28c | ⚠️ 19c | ✅ 0pr |

## Detailed observations

### F01_encrypted

* **pymupdf_open**: ✅ n_pages=1 encrypted=1 elapsed_s=0.028
* **pymupdf4llm**: ❌ `ValueError: cannot initialize - document still encrypted`
* **pdftotext**: ✅ rc=1 chars=0 elapsed_s=0.028
* **pdf2md_reorder**: ❌ `ValueError: document closed or encrypted`
* **caption_pairing**: ❌ `ValueError: document closed or encrypted`

### F02_empty

* **pymupdf_open**: ✅ n_pages=1 encrypted=0 elapsed_s=0.0
* **pymupdf4llm**: ✅ chars=0 elapsed_s=0.191
* **pdftotext**: ✅ rc=0 chars=1 elapsed_s=0.016
* **pdf2md_reorder**: ✅ chars=0 elapsed_s=0.001
* **caption_pairing**: ✅ n_pairs=0 elapsed_s=0.0

### F03_image_only

* **pymupdf_open**: ✅ n_pages=1 encrypted=0 elapsed_s=0.0
* **pymupdf4llm**: ✅ chars=103 elapsed_s=1.081
* **pdftotext**: ✅ rc=0 chars=1 elapsed_s=0.016
* **pdf2md_reorder**: ✅ chars=0 elapsed_s=0.004
* **caption_pairing**: ✅ n_pairs=0 elapsed_s=0.003

### F04_rotated

* **pymupdf_open**: ✅ n_pages=3 encrypted=0 elapsed_s=0.0
* **pymupdf4llm**: ✅ chars=21 elapsed_s=0.418
* **pdftotext**: ✅ rc=0 chars=66 elapsed_s=0.014
* **pdf2md_reorder**: ✅ chars=61 elapsed_s=0.003
* **caption_pairing**: ✅ n_pairs=0 elapsed_s=0.001

### F05_no_headings

* **pymupdf_open**: ✅ n_pages=5 encrypted=0 elapsed_s=0.0
* **pymupdf4llm**: ✅ chars=17975 elapsed_s=0.983
* **pdftotext**: ✅ rc=0 chars=17610 elapsed_s=0.018
* **pdf2md_reorder**: ✅ chars=17998 elapsed_s=0.01
* **caption_pairing**: ✅ n_pairs=0 elapsed_s=0.006

### F06_midword_break

* **pymupdf_open**: ✅ n_pages=2 encrypted=0 elapsed_s=0.0
* **pymupdf4llm**: ✅ chars=92 elapsed_s=0.248
* **pdftotext**: ✅ rc=0 chars=92 elapsed_s=0.014
* **pdf2md_reorder**: ✅ chars=88 elapsed_s=0.002
* **caption_pairing**: ✅ n_pairs=0 elapsed_s=0.001

### F07_weird_unicode

* **pymupdf_open**: ✅ n_pages=1 encrypted=0 elapsed_s=0.0
* **pymupdf4llm**: ✅ chars=47 elapsed_s=0.13
* **pdftotext**: ✅ rc=0 chars=44 elapsed_s=0.02
* **pdf2md_reorder**: ✅ chars=42 elapsed_s=0.002
* **caption_pairing**: ✅ n_pairs=0 elapsed_s=0.001

### F08_duplicate_pages

* **pymupdf_open**: ✅ n_pages=4 encrypted=0 elapsed_s=0.0
* **pymupdf4llm**: ✅ chars=252 elapsed_s=0.473
* **pdftotext**: ✅ rc=0 chars=228 elapsed_s=0.015
* **pdf2md_reorder**: ✅ chars=226 elapsed_s=0.005
* **caption_pairing**: ✅ n_pairs=0 elapsed_s=0.001

### F09_garbage_bytes

* **pymupdf_open**: ❌ `FileDataError: Failed to open file '/home/user/eval_harness/failure_pdfs/F09_garbage_bytes.pdf'.`
* **pymupdf4llm**: ❌ `FileDataError: Failed to open file '/home/user/eval_harness/failure_pdfs/F09_garbage_bytes.pdf'.`
* **pdftotext**: ✅ rc=1 chars=0 elapsed_s=0.014
* **pdf2md_reorder**: ❌ `FileDataError: Failed to open file '/home/user/eval_harness/failure_pdfs/F09_garbage_bytes.pdf'.`
* **caption_pairing**: ❌ `FileDataError: Failed to open file '/home/user/eval_harness/failure_pdfs/F09_garbage_bytes.pdf'.`

### F10_table_only

* **pymupdf_open**: ✅ n_pages=1 encrypted=0 elapsed_s=0.0
* **pymupdf4llm**: ✅ chars=41 elapsed_s=0.137
* **pdftotext**: ✅ rc=0 chars=28 elapsed_s=0.014
* **pdf2md_reorder**: ✅ chars=19 elapsed_s=0.002
* **caption_pairing**: ✅ n_pairs=0 elapsed_s=0.001

