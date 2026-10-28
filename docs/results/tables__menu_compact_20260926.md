# Table 2A,B — paper results

This entry now follows the manuscript. [Complete results](README.md).

## Table 2. Resource trade-offs

All entries are five-seed mean ± SD. Lifecycle costs are full-run totals per seed. Parallel execution compares wall-clock latency and accumulated active worker time.

### A. Candidate-plan limit

| Cap | Gain (pp) | Planning cost (k tokens) |
| --- | --- | --- |
| 8 | 5.2 ± 0.35 | 1.8 ± 0.12 |
| 16 | 5.8 ± 0.22 | 2.7 ± 0.15 |
| 32 | 6.3 ± 0.27 | 3.9 ± 0.21 |
| 64 | 6.5 ± 0.16 | 5.6 ± 0.23 |

[Candidate-plan CSV](tables/table2a_candidate_plans.csv)

### B. Lifecycle cost (k tokens)

| Category | MetaRSI-v1 | SpecRSI |
| --- | --- | --- |
| Acquisition | 14.5 ± 0.41 | 14.2 ± 0.35 |
| Training | 38.6 ± 0.96 | 34.8 ± 0.87 |
| Search | 37.2 ± 0.96 | 36.2 ± 0.86 |
| Reuse checks | 5.1 ± 0.26 | 4.8 ± 0.26 |
| Copies | 6.3 ± 0.26 | 6.1 ± 0.25 |
| Failed work | 6.8 ± 0.40 | 6.3 ± 0.32 |
| Planning | 4.2 ± 0.20 | 3.9 ± 0.20 |
| Total | 112.7 ± 3.44 | **106.3 ± 3.09** |

[Lifecycle CSV](tables/table2b_lifecycle.csv)
