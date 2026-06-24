# Distillation bench (E16)

Corpus: 476 figures across 34 papers.

## Verdict

* **Student (rule-based) handles 191/476 = 40.1% of figures** without needing the VLM teacher.
* Estimated time saved (assuming 60s/teacher call): **11,460s = ~191 minutes**
* Captions that are completely empty: 252

## Per-paper

| Paper | n figs | student | teacher needed | empty caption |
|---|---|---|---|---|
| baden-bohm-2023 | 8 | 8 | 0 | 0 |
| carceles-rodriguez-et-al-2022 | 5 | 5 | 0 | 0 |
| cast-model-paper-2026 | 6 | 6 | 0 | 0 |
| cerda-et-al-2022 | 6 | 4 | 2 | 0 |
| chapela-oliva-et-al-2024-viticulture-cap-critical-analysis | 3 | 3 | 0 | 0 |
| cuadros-casanova-et-al-2022 | 11 | 1 | 0 | 10 |
| duran-zuazo-et-al-2020-benefits-organic-olive-rainfed-system | 3 | 0 | 3 | 0 |
| efthimiou-2024 | 10 | 8 | 2 | 0 |
| fangliang-et-al-2024 | 27 | 16 | 1 | 10 |
| frontiers-in-soil-science-2026 | 7 | 5 | 0 | 2 |
| garcia-ruiz-castillo-llanque-et-al-2020 | 4 | 4 | 0 | 0 |
| garcia-ruiz-castillo-llanque-et-al-2023 | 8 | 8 | 0 | 0 |
| garrido-et-al-2026 | 6 | 2 | 4 | 0 |
| gomez-2025 | 11 | 0 | 0 | 11 |
| gonzalez-rosado-et-al-2021 | 4 | 4 | 0 | 0 |
| homet-et-al-2024 | 3 | 3 | 0 | 0 |
| jimenez-et-al-2023-farming-system-soil-management-floristic | 6 | 5 | 1 | 0 |
| leal-filho-et-al-2026 | 5 | 4 | 1 | 0 |
| lopez-bernal-et-al-2023 | 14 | 11 | 2 | 1 |
| lopez-vicente-calvo-seas-alvarez-cerda-2020 | 6 | 6 | 0 | 0 |
| marja-et-al-2022 | 1 | 1 | 0 | 0 |
| mesas-et-al-2022 | 3 | 2 | 1 | 0 |
| oecd-2023 | 245 | 30 | 0 | 215 |
| pareja-sanchez-et-al-2024 | 6 | 5 | 1 | 0 |
| penuela-et-al-2025 | 6 | 5 | 1 | 0 |
| pereira-de-souza-et-al-2023 | 8 | 8 | 0 | 0 |
| repullo-ruiberriz-de-torres-et-al-2021 | 7 | 3 | 4 | 0 |
| rodriguez-lizana-et-al-2020 | 7 | 4 | 2 | 1 |
| rodriguez-ruiz-gomez-munoz-2025 | 11 | 11 | 0 | 0 |
| s11104-012-1395-0 | 3 | 2 | 1 | 0 |
| s11356-016-8339-9 | 5 | 0 | 5 | 0 |
| sanchez-moreno-et-al-2015-tillage-herbicide-decrease-soil-bi | 3 | 2 | 1 | 0 |
| soil-use-and-management-2024-fernandezsoler-cover-crops-impr | 8 | 6 | 0 | 2 |
| torrus-castillo-et-al-2022 | 10 | 9 | 1 | 0 |