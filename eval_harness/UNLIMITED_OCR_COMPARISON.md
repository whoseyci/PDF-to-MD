# Unlimited-OCR vs ours -- feature comparison

PDF: ``Angelioudakis et al. (2025).pdf`` (24 pages, MDPI Diversity 2025)

## Score

- **12/12** features that Unlimited-OCR produced cleanly are also in our output
- Pipeline ran in 22 s on 2 vCPU/2 GB sandbox (no GPU, no network)
- Output paper.md is 88,389 chars (13,605 words); pages 1-3 are 10,361 chars
- Output paper.json has 97 refs, 16 figures

## Matched features

- ✅ Coordinates as ``35°30′59.5″ N``
- ✅ Area as ``m²`` (Unicode)
- ✅ ``7×7`` spacing
- ✅ Mean ± SD as ``1.20±0.07``
- ✅ Shannon H′ as ``H′``
- ✅ Latin binomials italicised (``_Festuca arundinacea_``)
- ✅ Citations linked to refs (``[[1](#ref-001)]``)
- ✅ Author list as structured field
- ✅ Figure files extracted to disk
- ✅ Page-break markers
- ✅ MDPI sidebar NOT inlined in intro
- ✅ ``1. Introduction`` only appears once on page 1

## Unlimited-OCR-specific features

**Unlimited-OCR has these; we don't:**

- · ORCID ``<sup>id</sup>`` icon transcribed (vision-only)
- · ``[Non-Text]`` placeholders for journal banners
- · Math notation as LaTeX ``\( ... \)``
- · Subfigure inline placeholders ``![](images/2.jpg)``
