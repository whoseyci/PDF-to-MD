# Eval harness report

Papers: 10 arXiv papers with ground-truth LaTeX rendered via pandoc / regex-stripper.

All extractors normalised identically (lowercase, collapse whitespace, strip non-ASCII) before scoring.

Metrics:
  * **char_ratio** -- extracted_chars / gt_chars (closer to 1.0 = comparable length)
  * **F1** -- word-set F1 vs ground truth
  * **WER\*** -- proxy WER on sorted token bags (monotone, lower is better, not a strict edit distance)
  * **t_s** -- seconds to extract

## Aggregate (averaged across papers)

| Extractor | n ok | n err | avg F1 | avg WER\* | avg s |
|---|---|---|---|---|---|
| `pdftotext` | 10 | 0 | 0.777 | 0.024 | 0.106 |
| `pdftotext-stream` | 10 | 0 | 0.802 | 0.02 | 0.096 |
| `pymupdf4llm` | 10 | 0 | 0.803 | 0.019 | 15.382 |
| `pdf2md-postprocess` | 10 | 0 | 0.803 | 0.019 | 15.369 |
| `pdf2md-reorder-e1` | 10 | 0 | 0.766 | 0.033 | 0.122 |

## Per-paper detail

### 1312.6114

_gt: 30,868 chars / 4,710 words; pdf: 3,926,758 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 41,327 | 0.7922 | 0.0214 | 1.339 | 0.09 |
| `pdftotext-stream` | 41,227 | 0.8234 | 0.0163 | 1.336 | 0.09 |
| `pymupdf4llm` | 47,068 | 0.8202 | 0.0157 | 1.525 | 8.39 |
| `pdf2md-postprocess` | 47,068 | 0.8202 | 0.0157 | 1.525 | 7.75 |
| `pdf2md-reorder-e1` | 41,047 | 0.7753 | 0.0299 | 1.33 | 0.1 |

### 1406.2661

_gt: 17,579 chars / 2,656 words; pdf: 530,482 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 28,997 | 0.7641 | 0.0113 | 1.65 | 0.08 |
| `pdftotext-stream` | 28,957 | 0.7777 | 0.0079 | 1.647 | 0.06 |
| `pymupdf4llm` | 30,713 | 0.7787 | 0.0079 | 1.747 | 5.1 |
| `pdf2md-postprocess` | 30,856 | 0.7787 | 0.0079 | 1.755 | 4.92 |
| `pdf2md-reorder-e1` | 28,853 | 0.7413 | 0.0245 | 1.641 | 0.05 |

### 1409.0473

_gt: 32,751 chars / 5,284 words; pdf: 444,482 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 50,101 | 0.7375 | 0.0212 | 1.53 | 0.11 |
| `pdftotext-stream` | 49,991 | 0.7542 | 0.018 | 1.526 | 0.07 |
| `pymupdf4llm` | 52,304 | 0.777 | 0.0074 | 1.597 | 5.38 |
| `pdf2md-postprocess` | 52,590 | 0.777 | 0.0074 | 1.606 | 5.45 |
| `pdf2md-reorder-e1` | 49,779 | 0.7463 | 0.0187 | 1.52 | 0.08 |

### 1503.02531

_gt: 29,485 chars / 4,961 words; pdf: 106,630 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 33,763 | 0.8849 | 0.0032 | 1.145 | 0.04 |
| `pdftotext-stream` | 33,723 | 0.9014 | 0.001 | 1.144 | 0.04 |
| `pymupdf4llm` | 33,513 | 0.9076 | 0.0077 | 1.137 | 2.95 |
| `pdf2md-postprocess` | 33,513 | 0.9076 | 0.0077 | 1.137 | 3.08 |
| `pdf2md-reorder-e1` | 33,661 | 0.8631 | 0.0117 | 1.142 | 0.06 |

### 1512.03385

_gt: 40,497 chars / 6,684 words; pdf: 819,383 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 59,312 | 0.7722 | 0.02 | 1.465 | 0.1 |
| `pdftotext-stream` | 58,958 | 0.8282 | 0.0084 | 1.456 | 0.09 |
| `pymupdf4llm` | 62,369 | 0.8206 | 0.0096 | 1.54 | 8.31 |
| `pdf2md-postprocess` | 62,756 | 0.8206 | 0.0096 | 1.55 | 8.15 |
| `pdf2md-reorder-e1` | 59,138 | 0.752 | 0.034 | 1.46 | 0.08 |

### 1706.03762

_gt: 35,354 chars / 5,636 words; pdf: 2,215,244 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 39,529 | 0.9357 | 0.0403 | 1.118 | 0.12 |
| `pdftotext-stream` | 39,497 | 0.9408 | 0.0401 | 1.117 | 0.12 |
| `pymupdf4llm` | 41,690 | 0.9435 | 0.0357 | 1.179 | 6.36 |
| `pdf2md-postprocess` | 42,083 | 0.9435 | 0.0357 | 1.19 | 6.49 |
| `pdf2md-reorder-e1` | 39,389 | 0.9386 | 0.0383 | 1.114 | 0.12 |

### 1810.04805

_gt: 43,307 chars / 7,005 words; pdf: 775,166 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 64,125 | 0.6498 | 0.0458 | 1.481 | 0.09 |
| `pdftotext-stream` | 63,419 | 0.7252 | 0.0298 | 1.464 | 0.08 |
| `pymupdf4llm` | 66,562 | 0.7158 | 0.0298 | 1.537 | 13.03 |
| `pdf2md-postprocess` | 66,713 | 0.7158 | 0.0298 | 1.54 | 12.77 |
| `pdf2md-reorder-e1` | 63,843 | 0.6351 | 0.0637 | 1.474 | 0.09 |

### 1909.11942

_gt: 36,658 chars / 5,762 words; pdf: 419,105 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 60,285 | 0.6575 | 0.0666 | 1.645 | 0.07 |
| `pdftotext-stream` | 60,063 | 0.6778 | 0.0633 | 1.638 | 0.07 |
| `pymupdf4llm` | 62,458 | 0.6843 | 0.0679 | 1.704 | 6.93 |
| `pdf2md-postprocess` | 64,225 | 0.6843 | 0.0679 | 1.752 | 6.9 |
| `pdf2md-reorder-e1` | 60,074 | 0.6477 | 0.0758 | 1.639 | 0.09 |

### 2005.14165

_gt: 229,438 chars / 35,260 words; pdf: 6,768,044 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 236,602 | 0.8525 | 0.0058 | 1.031 | 0.2 |
| `pdftotext-stream` | 236,540 | 0.853 | 0.0061 | 1.031 | 0.2 |
| `pymupdf4llm` | 249,769 | 0.853 | 0.0059 | 1.089 | 74.89 |
| `pdf2md-postprocess` | 252,915 | 0.853 | 0.0059 | 1.102 | 75.53 |
| `pdf2md-reorder-e1` | 235,714 | 0.8373 | 0.0174 | 1.027 | 0.4 |

### 2010.11929

_gt: 39,715 chars / 6,329 words; pdf: 3,743,814 bytes_

| Extractor | chars | F1 | WER\* | char ratio | t_s |
|---|---|---|---|---|---|
| `pdftotext` | 67,209 | 0.7231 | 0.0068 | 1.692 | 0.16 |
| `pdftotext-stream` | 67,051 | 0.7373 | 0.0051 | 1.688 | 0.14 |
| `pymupdf4llm` | 71,478 | 0.7251 | 0.0017 | 1.8 | 22.48 |
| `pdf2md-postprocess` | 71,665 | 0.7251 | 0.0017 | 1.804 | 22.65 |
| `pdf2md-reorder-e1` | 66,861 | 0.7184 | 0.0183 | 1.684 | 0.15 |

