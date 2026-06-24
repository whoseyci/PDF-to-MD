# Distillation bench (E16)

Corpus: 476 figures across 34 papers.

## Verdict

* **Student (rule-based) handles 107/476 = 22.5% of figures** without needing the VLM teacher.
* Estimated time saved (assuming 60s/teacher call): **6,420s = ~107 minutes**
* Captions that are completely empty: 350

## Per-paper

| Paper | n figs | student | teacher needed | empty caption |
|---|---|---|---|---|
| baden-bohm-2023 | 8 | 7 | 0 | 1 |
| carceles-rodriguez-et-al-2022 | 5 | 3 | 0 | 2 |
| cast-model-paper-2026 | 6 | 0 | 0 | 6 |
| cerda-et-al-2022 | 6 | 1 | 2 | 3 |
| chapela-oliva-et-al-2024-viticulture-cap-critical-analysis | 3 | 0 | 0 | 3 |
| cuadros-casanova-et-al-2022 | 11 | 0 | 0 | 11 |
| duran-zuazo-et-al-2020-benefits-organic-olive-rainfed-system | 3 | 0 | 0 | 3 |
| efthimiou-2024 | 10 | 7 | 1 | 2 |
| fangliang-et-al-2024 | 27 | 8 | 1 | 18 |
| frontiers-in-soil-science-2026 | 7 | 0 | 0 | 7 |
| garcia-ruiz-castillo-llanque-et-al-2020 | 4 | 4 | 0 | 0 |
| garcia-ruiz-castillo-llanque-et-al-2023 | 8 | 8 | 0 | 0 |
| garrido-et-al-2026 | 6 | 0 | 3 | 3 |
| gomez-2025 | 11 | 0 | 0 | 11 |
| gonzalez-rosado-et-al-2021 | 4 | 4 | 0 | 0 |
| homet-et-al-2024 | 3 | 0 | 0 | 3 |
| jimenez-et-al-2023-farming-system-soil-management-floristic | 6 | 4 | 1 | 1 |
| leal-filho-et-al-2026 | 5 | 3 | 1 | 1 |
| lopez-bernal-et-al-2023 | 14 | 7 | 1 | 6 |
| lopez-vicente-calvo-seas-alvarez-cerda-2020 | 6 | 6 | 0 | 0 |
| marja-et-al-2022 | 1 | 1 | 0 | 0 |
| mesas-et-al-2022 | 3 | 2 | 1 | 0 |
| oecd-2023 | 245 | 10 | 0 | 235 |
| pareja-sanchez-et-al-2024 | 6 | 5 | 1 | 0 |
| penuela-et-al-2025 | 6 | 5 | 1 | 0 |
| pereira-de-souza-et-al-2023 | 8 | 6 | 0 | 2 |
| repullo-ruiberriz-de-torres-et-al-2021 | 7 | 3 | 4 | 0 |
| rodriguez-lizana-et-al-2020 | 7 | 2 | 2 | 3 |
| rodriguez-ruiz-gomez-munoz-2025 | 11 | 11 | 0 | 0 |
| s11104-012-1395-0 | 3 | 0 | 0 | 3 |
| s11356-016-8339-9 | 5 | 0 | 0 | 5 |
| sanchez-moreno-et-al-2015-tillage-herbicide-decrease-soil-bi | 3 | 0 | 0 | 3 |
| soil-use-and-management-2024-fernandezsoler-cover-crops-impr | 8 | 0 | 0 | 8 |
| torrus-castillo-et-al-2022 | 10 | 0 | 0 | 10 |