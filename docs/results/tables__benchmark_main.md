# Table 1 — paper results

This entry now follows the manuscript. [Complete results](README.md).

## Table 1. Scientific and system performance

Five-seed mean ± SD. Bold marks the best comparable mean. Scientific comparisons follow the [Janus protocol](https://arxiv.org/abs/2608.08189). System comparisons follow the [MetaRSI protocol](https://arxiv.org/abs/2609.06396v2) under matched editing permissions. D, H, and M denote data, harness, and model updates.

### A. Scientific terminal gain

Higher is better. Scores and terminal gains are in the dimensionless native task units.

| Method | PyBaMM | AQM | FDTD | Reactor | IRP |
| --- | --- | --- | --- | --- | --- |
| Alg.-Only [21](https://arxiv.org/abs/2506.13131) | 0.12 ± 0.02 | 0.09 ± 0.03 | 0.14 ± 0.05 | 0.11 ± 0.05 | 0.10 ± 0.07 |
| Direct-LLM [10](https://doi.org/10.1016/j.swevo.2024.101741) | 0.21 ± 0.03 | 0.17 ± 0.01 | 0.23 ± 0.02 | 0.18 ± 0.06 | 0.18 ± 0.05 |
| Embed-GP [27](https://proceedings.neurips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html) | 0.29 ± 0.03 | 0.26 ± 0.01 | 0.36 ± 0.03 | 0.29 ± 0.03 | 0.28 ± 0.06 |
| Janus [18](https://arxiv.org/abs/2608.08189) | 0.35 ± 0.05 | 0.30 ± 0.03 | 0.40 ± 0.02 | 0.33 ± 0.02 | 0.27 ± 0.04 |
| SpecRSI | **0.58 ± 0.02** | **0.52 ± 0.03** | **0.64 ± 0.01** | **0.55 ± 0.01** | **0.53 ± 0.03** |

### A. Call-axis AUBC

Higher is better. AUBC integrates gain over complete evaluation calls.

| Method | PyBaMM | AQM | FDTD | Reactor | IRP |
| --- | --- | --- | --- | --- | --- |
| Alg.-Only [21](https://arxiv.org/abs/2506.13131) | 7.3 ± 0.1 | 5.1 ± 0.8 | 8.5 ± 0.6 | 6.7 ± 0.2 | 5.8 ± 0.6 |
| Direct-LLM [10](https://doi.org/10.1016/j.swevo.2024.101741) | 19.2 ± 0.8 | 14.3 ± 0.5 | 22.6 ± 0.3 | 16.2 ± 0.1 | 13.7 ± 0.3 |
| Embed-GP [27](https://proceedings.neurips.cc/paper_files/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html) | 30.5 ± 0.4 | 24.8 ± 0.1 | 36.2 ± 0.9 | 27.4 ± 0.2 | 23.1 ± 0.1 |
| Janus [18](https://arxiv.org/abs/2608.08189) | 42.8 ± 0.3 | 34.7 ± 0.9 | 51.4 ± 0.7 | 37.0 ± 0.2 | 22.6 ± 0.1 |
| SpecRSI | **57.7 ± 0.4** | **46.8 ± 0.1** | **67.9 ± 0.2** | **50.3 ± 0.4** | **43.3 ± 0.1** |

[Scientific CSV](tables/table1_scientific.csv)

### B. System scores (%)

Higher is better. Complete admissible versions are compared within each permission group.

**D/H/M permissions**

| Method | Terminal-Bench 2.1 | SWE-Pro | GPQA-hard100 | AIME |
| --- | --- | --- | --- | --- |
| Initial [29](https://arxiv.org/abs/2609.06396v2) | 23.6 ± 0.7 | 10.2 ± 0.3 | 71.2 ± 1.0 | 54.9 ± 0.5 |
| Data-RSI [29](https://arxiv.org/abs/2609.06396v2) | 27.4 ± 0.3 | 12.4 ± 0.8 | 73.8 ± 0.3 | 57.6 ± 0.1 |
| Harness-RSI [29](https://arxiv.org/abs/2609.06396v2) | 29.4 ± 0.6 | 14.8 ± 0.3 | 78.2 ± 0.7 | 63.1 ± 0.8 |
| Model-RSI [29](https://arxiv.org/abs/2609.06396v2) | 27.0 ± 0.3 | 13.9 ± 0.1 | 75.4 ± 0.6 | 59.7 ± 0.2 |
| Fixed D→M→H [29](https://arxiv.org/abs/2609.06396v2) | 30.3 ± 0.6 | 15.3 ± 0.3 | 79.5 ± 0.2 | 64.2 ± 0.6 |
| Static router [29](https://arxiv.org/abs/2609.06396v2) | 30.8 ± 0.2 | 14.9 ± 0.3 | 79.7 ± 0.9 | 64.0 ± 0.2 |
| MetaRSI-v1 [29](https://arxiv.org/abs/2609.06396v2) | 32.1 ± 0.3 | 19.5 ± 0.8 | 83.8 ± 0.2 | 68.5 ± 0.1 |
| SpecRSI | **35.6 ± 0.6** | **23.7 ± 0.1** | **87.3 ± 0.3** | **72.9 ± 0.3** |

**D+H permissions**

| Method | Terminal-Bench 2.1 | SWE-Pro | GPQA-hard100 | AIME |
| --- | --- | --- | --- | --- |
| MetaRSI-H [29](https://arxiv.org/abs/2609.06396v2) | 28.6 ± 0.5 | 14.2 ± 0.2 | 77.5 ± 0.5 | 61.7 ± 0.2 |
| SpecRSI-H | **32.1 ± 0.1** | **17.3 ± 0.5** | **80.9 ± 0.5** | **65.6 ± 0.1** |

[System CSV](tables/table1_system.csv)
