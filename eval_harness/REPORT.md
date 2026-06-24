# Eval harness report

Papers: 16 arXiv papers with ground-truth LaTeX rendered via pandoc / regex-stripper.

All extractors normalised identically (lowercase, collapse whitespace, strip non-ASCII) before scoring.

Metrics:
  * **char_ratio** -- extracted_chars / gt_chars (closer to 1.0 = comparable length)
  * **F1** -- word-set F1 vs ground truth
  * **WER\*** -- proxy WER on sorted token bags (monotone, lower is better, not a strict edit distance)
  * **t_s** -- seconds to extract

## Aggregate (averaged across papers)

| Extractor | n ok | n err | avg F1 | avg WER\* | avg s |
|---|---|---|---|---|---|
| `pdftotext` | 16 | 0 | 0.706 | 0.153 | 0.074 |
| `pdftotext-stream` | 16 | 0 | 0.727 | 0.149 | 0.073 |
| `pymupdf4llm` | 16 | 0 | 0.821 | 0.08 | 14.769 |
| `pdf2md-postprocess` | 16 | 0 | 0.821 | 0.08 | 14.317 |
| `pdf2md-reorder-e1` | 16 | 0 | 0.729 | 0.148 | 0.099 |
| `pdf2md-auto` | 16 | 0 | 0.821 | 0.077 | 3.054 |
| `pdf2md-auto-rotfix` | 16 | 0 | 0.821 | 0.077 | 16.332 |

## Per-paper detail

### 0710.5491

_gt: 28,910 chars / 4,859 words; pdf: 393,243 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 31,650 | 0.9249 | 0.0233 | 1.095 | 0.08 |
| `pdftotext-stream` | 31,628 | 0.936 | 0.022 | 1.094 | 0.08 |
| `pymupdf4llm` | 32,620 | 0.9227 | 0.035 | 1.128 | 14.5 |
| `pdf2md-postprocess` | 32,664 | 0.9227 | 0.035 | 1.13 | 14.24 |
| `pdf2md-reorder-e1` | 31,518 | 0.9337 | 0.0261 | 1.09 | 0.12 |
| `pdf2md-auto` | 31,628 | 0.936 | 0.022 | 1.094 | 0.28 |
| `pdf2md-auto-rotfix` | 31,628 | 0.936 | 0.022 | 1.094 | 11.39 |

### 0805.4452

_gt: 37,043 chars / 6,468 words; pdf: 121,741 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 37,650 | 0.8797 | 0.1288 | 1.016 | 0.05 |
| `pdftotext-stream` | 37,614 | 0.8898 | 0.1272 | 1.015 | 0.05 |
| `pymupdf4llm` | 39,061 | 0.8814 | 0.1534 | 1.054 | 3.29 |
| `pdf2md-postprocess` | 39,061 | 0.8814 | 0.1534 | 1.054 | 3.3 |
| `pdf2md-reorder-e1` | 37,467 | 0.8819 | 0.1299 | 1.011 | 0.07 |
| `pdf2md-auto` | 37,614 | 0.8898 | 0.1272 | 1.015 | 0.25 |
| `pdf2md-auto-rotfix` | 37,614 | 0.8898 | 0.1272 | 1.015 | 11.88 |

### 1209.3818

_gt: 21,749 chars / 3,770 words; pdf: 87,306 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 23,580 | 0.873 | 0.0326 | 1.084 | 0.03 |
| `pdftotext-stream` | 23,552 | 0.8867 | 0.0313 | 1.083 | 0.03 |
| `pymupdf4llm` | 23,482 | 0.8915 | 0.0451 | 1.08 | 1.89 |
| `pdf2md-postprocess` | 23,482 | 0.8915 | 0.0451 | 1.08 | 1.86 |
| `pdf2md-reorder-e1` | 23,486 | 0.8903 | 0.0289 | 1.08 | 0.03 |
| `pdf2md-auto` | 23,552 | 0.8867 | 0.0313 | 1.083 | 0.07 |
| `pdf2md-auto-rotfix` | 23,552 | 0.8867 | 0.0313 | 1.083 | 3.41 |

### 1312.6114

_gt: 30,868 chars / 4,710 words; pdf: 3,926,758 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 41,327 | 0.7922 | 0.0214 | 1.339 | 0.09 |
| `pdftotext-stream` | 41,227 | 0.8234 | 0.0163 | 1.336 | 0.09 |
| `pymupdf4llm` | 47,068 | 0.8202 | 0.0157 | 1.525 | 7.79 |
| `pdf2md-postprocess` | 47,068 | 0.8202 | 0.0157 | 1.525 | 7.83 |
| `pdf2md-reorder-e1` | 40,904 | 0.8218 | 0.0163 | 1.325 | 0.1 |
| `pdf2md-auto` | 41,227 | 0.8234 | 0.0163 | 1.336 | 0.29 |
| `pdf2md-auto-rotfix` | 41,227 | 0.8234 | 0.0163 | 1.336 | 12.09 |

### 1406.2661

_gt: 17,579 chars / 2,656 words; pdf: 530,482 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 28,997 | 0.7641 | 0.0113 | 1.65 | 0.06 |
| `pdftotext-stream` | 28,957 | 0.7777 | 0.0079 | 1.647 | 0.06 |
| `pymupdf4llm` | 30,713 | 0.7787 | 0.0079 | 1.747 | 4.98 |
| `pdf2md-postprocess` | 30,856 | 0.7787 | 0.0079 | 1.755 | 4.96 |
| `pdf2md-reorder-e1` | 28,875 | 0.7837 | 0.006 | 1.643 | 0.06 |
| `pdf2md-auto` | 28,957 | 0.7777 | 0.0079 | 1.647 | 0.18 |
| `pdf2md-auto-rotfix` | 28,957 | 0.7777 | 0.0079 | 1.647 | 7.45 |

### 1406.2661_scanned

_gt: 17,579 chars / 2,656 words; pdf: 25,250,030 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 0 | 0.0 | 1.0 | 0.0 | 0.01 |
| `pdftotext-stream` | 0 | 0.0 | 1.0 | 0.0 | 0.01 |
| `pymupdf4llm` | 13,005 | 0.7245 | 0.3724 | 0.74 | 20.38 |
| `pdf2md-postprocess` | 13,005 | 0.7245 | 0.3724 | 0.74 | 20.4 |
| `pdf2md-reorder-e1` | 0 | 0.0 | 1.0 | 0.0 | 0.02 |
| `pdf2md-auto` | 13,005 | 0.7245 | 0.3724 | 0.74 | 20.0 |
| `pdf2md-auto-rotfix` | 13,005 | 0.7245 | 0.3724 | 0.74 | 23.84 |

### 1409.0473

_gt: 32,751 chars / 5,284 words; pdf: 444,482 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 50,101 | 0.7375 | 0.0212 | 1.53 | 0.07 |
| `pdftotext-stream` | 49,991 | 0.7542 | 0.018 | 1.526 | 0.07 |
| `pymupdf4llm` | 52,304 | 0.777 | 0.0074 | 1.597 | 5.56 |
| `pdf2md-postprocess` | 52,590 | 0.777 | 0.0074 | 1.606 | 5.78 |
| `pdf2md-reorder-e1` | 49,734 | 0.7801 | 0.0072 | 1.519 | 0.09 |
| `pdf2md-auto` | 49,991 | 0.7542 | 0.018 | 1.526 | 0.27 |
| `pdf2md-auto-rotfix` | 49,991 | 0.7542 | 0.018 | 1.526 | 14.68 |

### 1503.02531

_gt: 29,485 chars / 4,961 words; pdf: 106,630 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 33,763 | 0.8849 | 0.0032 | 1.145 | 0.04 |
| `pdftotext-stream` | 33,723 | 0.9014 | 0.001 | 1.144 | 0.04 |
| `pymupdf4llm` | 33,513 | 0.9076 | 0.0077 | 1.137 | 3.15 |
| `pdf2md-postprocess` | 33,513 | 0.9076 | 0.0077 | 1.137 | 3.1 |
| `pdf2md-reorder-e1` | 33,671 | 0.9019 | 0.0006 | 1.142 | 0.06 |
| `pdf2md-auto` | 33,723 | 0.9014 | 0.001 | 1.144 | 0.23 |
| `pdf2md-auto-rotfix` | 33,723 | 0.9014 | 0.001 | 1.144 | 9.32 |

### 1503.02531_scanned

_gt: 29,485 chars / 4,961 words; pdf: 25,250,030 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 0 | 0.0 | 1.0 | 0.0 | 0.02 |
| `pdftotext-stream` | 0 | 0.0 | 1.0 | 0.0 | 0.01 |
| `pymupdf4llm` | 16,458 | 0.7841 | 0.4642 | 0.558 | 23.94 |
| `pdf2md-postprocess` | 16,458 | 0.7841 | 0.4642 | 0.558 | 23.35 |
| `pdf2md-reorder-e1` | 0 | 0.0 | 1.0 | 0.0 | 0.01 |
| `pdf2md-auto` | 16,458 | 0.7841 | 0.4642 | 0.558 | 23.8 |
| `pdf2md-auto-rotfix` | 16,458 | 0.7841 | 0.4642 | 0.558 | 27.08 |

### 1512.03385

_gt: 40,497 chars / 6,684 words; pdf: 819,383 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 59,312 | 0.7722 | 0.02 | 1.465 | 0.09 |
| `pdftotext-stream` | 58,958 | 0.8282 | 0.0084 | 1.456 | 0.09 |
| `pymupdf4llm` | 62,369 | 0.8206 | 0.0096 | 1.54 | 8.08 |
| `pdf2md-postprocess` | 62,756 | 0.8206 | 0.0096 | 1.55 | 8.1 |
| `pdf2md-reorder-e1` | 58,980 | 0.823 | 0.0087 | 1.456 | 0.08 |
| `pdf2md-auto` | 58,958 | 0.8282 | 0.0084 | 1.456 | 0.25 |
| `pdf2md-auto-rotfix` | 58,958 | 0.8282 | 0.0084 | 1.456 | 10.35 |

### 1706.03762

_gt: 35,354 chars / 5,636 words; pdf: 2,215,244 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 39,529 | 0.9357 | 0.0403 | 1.118 | 0.12 |
| `pdftotext-stream` | 39,497 | 0.9408 | 0.0401 | 1.117 | 0.12 |
| `pymupdf4llm` | 41,690 | 0.9435 | 0.0357 | 1.179 | 6.38 |
| `pdf2md-postprocess` | 42,083 | 0.9435 | 0.0357 | 1.19 | 6.44 |
| `pdf2md-reorder-e1` | 39,453 | 0.943 | 0.0383 | 1.116 | 0.13 |
| `pdf2md-auto` | 39,497 | 0.9408 | 0.0401 | 1.117 | 0.4 |
| `pdf2md-auto-rotfix` | 39,497 | 0.9408 | 0.0401 | 1.117 | 12.35 |

### 1810.04805

_gt: 43,307 chars / 7,005 words; pdf: 775,166 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 64,125 | 0.6498 | 0.0458 | 1.481 | 0.08 |
| `pdftotext-stream` | 63,419 | 0.7252 | 0.0298 | 1.464 | 0.08 |
| `pymupdf4llm` | 66,562 | 0.7158 | 0.0298 | 1.537 | 12.97 |
| `pdf2md-postprocess` | 66,713 | 0.7158 | 0.0298 | 1.54 | 12.87 |
| `pdf2md-reorder-e1` | 63,451 | 0.722 | 0.0301 | 1.465 | 0.1 |
| `pdf2md-auto` | 63,419 | 0.7252 | 0.0298 | 1.464 | 0.31 |
| `pdf2md-auto-rotfix` | 63,419 | 0.7252 | 0.0298 | 1.464 | 14.24 |

### 1907.05047

_gt: 12,408 chars / 1,937 words; pdf: 301,354 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 15,417 | 0.8554 | 0.0284 | 1.243 | 0.03 |
| `pdftotext-stream` | 15,305 | 0.8985 | 0.0145 | 1.233 | 0.03 |
| `pymupdf4llm` | 15,856 | 0.906 | 0.015 | 1.278 | 10.44 |
| `pdf2md-postprocess` | 16,125 | 0.9066 | 0.015 | 1.3 | 10.5 |
| `pdf2md-reorder-e1` | 15,303 | 0.8979 | 0.0145 | 1.233 | 0.02 |
| `pdf2md-auto` | 15,305 | 0.8985 | 0.0145 | 1.233 | 0.07 |
| `pdf2md-auto-rotfix` | 15,305 | 0.8985 | 0.0145 | 1.233 | 3.53 |

### 1909.11942

_gt: 36,658 chars / 5,762 words; pdf: 419,105 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 60,285 | 0.6575 | 0.0666 | 1.645 | 0.07 |
| `pdftotext-stream` | 60,063 | 0.6778 | 0.0633 | 1.638 | 0.07 |
| `pymupdf4llm` | 62,458 | 0.6843 | 0.0679 | 1.704 | 6.8 |
| `pdf2md-postprocess` | 64,225 | 0.6843 | 0.0679 | 1.752 | 6.7 |
| `pdf2md-reorder-e1` | 60,037 | 0.6874 | 0.0587 | 1.638 | 0.1 |
| `pdf2md-auto` | 60,063 | 0.6778 | 0.0633 | 1.638 | 0.33 |
| `pdf2md-auto-rotfix` | 60,063 | 0.6778 | 0.0633 | 1.638 | 14.99 |

### 2005.14165

_gt: 229,438 chars / 35,260 words; pdf: 6,768,044 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 236,602 | 0.8525 | 0.0058 | 1.031 | 0.2 |
| `pdftotext-stream` | 236,540 | 0.853 | 0.0061 | 1.031 | 0.2 |
| `pymupdf4llm` | 249,769 | 0.853 | 0.0059 | 1.089 | 83.58 |
| `pdf2md-postprocess` | 252,915 | 0.853 | 0.0059 | 1.102 | 76.4 |
| `pdf2md-reorder-e1` | 236,319 | 0.8524 | 0.0059 | 1.03 | 0.42 |
| `pdf2md-auto` | 236,540 | 0.853 | 0.0061 | 1.031 | 1.46 |
| `pdf2md-auto-rotfix` | 236,540 | 0.853 | 0.0061 | 1.031 | 63.16 |

### 2010.11929

_gt: 39,715 chars / 6,329 words; pdf: 3,743,814 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 67,209 | 0.7231 | 0.0068 | 1.692 | 0.15 |
| `pdftotext-stream` | 67,051 | 0.7373 | 0.0051 | 1.688 | 0.14 |
| `pymupdf4llm` | 71,478 | 0.7251 | 0.0017 | 1.8 | 22.58 |
| `pdf2md-postprocess` | 71,665 | 0.7251 | 0.0017 | 1.804 | 23.24 |
| `pdf2md-reorder-e1` | 66,945 | 0.7526 | 0.0016 | 1.686 | 0.18 |
| `pdf2md-auto` | 67,051 | 0.7373 | 0.0051 | 1.688 | 0.68 |
| `pdf2md-auto-rotfix` | 67,051 | 0.7373 | 0.0051 | 1.688 | 21.56 |

