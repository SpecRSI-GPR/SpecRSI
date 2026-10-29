# Data accompanying the paper figures

These files contain the numerical records behind the figures. Figure 1 and Figure 3 are mechanism diagrams and do not introduce experimental data. The records and figure assets correspond to the paper.

Each `*_seeds.csv` lists individual seed values. `*_summary.csv` and `*_summary.md` provide editable numerical tables. Means are unweighted across five seed values; SD uses denominator n-1. Seed pairing follows the stated experimental cohorts. Hit rates use pooled counts.

| Figure | Seed or instance records | Summary table | Units and interpretation |
|---|---|---|---|
| 2 | [300 values](figure2_seeds.csv) | [60 groups](figure2_summary.md) | Cost: k tokens to first qualifying outcome or termination; score: %. |
| 4A | [500 values](figure4a_seeds.csv) | [100 groups](figure4a_summary.md) | Terminal-gain/AUBC relative uplift over Janus, in percent. |
| 4B | [300 score values](figure4b_scores_seeds.csv), [100 derived paired differences](figure4b_paired_gain_seeds.csv) | [Scores](figure4b_scores_summary.md), [paired gain](figure4b_paired_gain_summary.md) | Score %, paired difference pp. Differences are derived from the same score values, not additional runs. |
| 5 | [730 values](figure5_seeds.csv) | [146 groups](figure5_summary.md) | Task score % and mean full-task inference cost in k tokens. |
| 6 | [640 values](figure6_seeds.csv), [pooled hits](figure6_pooled_hits.csv) | [128 groups](figure6_summary.md) | Gain pp; cost k tokens/search; hit count/125. |
| 7 | [470 values](figure7_seeds.csv), [20 factorial-control runs](figure7c_runs.csv) | [94 groups](figure7_summary.md) | Gain pp; C total cost k tokens/seed. B covers rounds 2-20 after initialization. |
| S1 | [620 instance records](figure_s1_instances.csv), [40 seed aggregates](figure_s1_seed_aggregate_seeds.csv) | [8 groups](figure_s1_seed_aggregate_summary.md) | PyBaMM gain: dimensionless; SWE-Pro success: %. Cost aggregates: k tokens/seed. |
| S2 | [95 values](figure_s2_seeds.csv) | [19 groups](figure_s2_summary.md) | Gain pp; B cost k tokens/search; E training cost k tokens. |

## Protocols and accounting

- Figure 2, Figure 4, Figure 5, Table 1, and Figure S1 each define their cohort, starting agent, permissions, and budget. Figure 4 scientific uplift and Table 1 scientific ratios summarize their respective comparisons.
- Figure 5 compares cost with the strongest-scoring baseline in each task. SpecRSI improves all eleven task scores by 3.5-6.0 pp with 3.2%-17.6% lower cost than those baselines.
- Figure 6 uses five seed means per condition. A/B contain 25 searches per seed. C applies eight execution-only caps, where 100% is 65 k tokens of direct draft/verification execution; total cost additionally includes planning and initialization. Held-out negative gains are retained. The A/B hit rates are pooled from 125 searches.
- Figure 7B carries the previous complete admissible agent, resets evidence and budgets, and evaluates rounds 2-20 after round-1 initialization. Frozen improvers remain fixed; Updated uses the respective method's update. Figure 7C totals training plus eight independent evaluation searches per seed; gain is the mean across those searches.
- Figure S1 averages outcomes within each seed, then calculates SD across the five seed aggregates. Seed cost sums 12 PyBaMM or 50 SWE-Pro instances. Paired SWE-Pro outcomes are 155 both-pass, 15 SpecRSI-only, 0 HGM-only, and 80 neither-pass.
- Figure S2A/B compare complete-evaluation gain (CQ) with continuation gain (CR). Original configurations have different inputs/stopping rules (25 searches/seed); matched configurations differ only in Scorer target (8 searches/seed). With lambda=mu=0, CR is gain achievable under remaining-budget feasibility, without an explicit cost-penalty subtraction in its label.
- Figure S2C masks Scorer/Policy decision features while retaining complete Solver constraints and actual verification. Conditions retain no evidence features, availability flags only, observed values only, flags plus values, or full evidence. Resource state remains available.
- Figure S2D/E uses held-out search gains. Pair supervision shares the acquired plans with the mean-only condition; ordinary-search data match acquisition expense within 2%. Training cost is absolute k tokens, not an index. It counts training forward input/output and forced-continuation tokens, excludes search tokens, and does not translate backward computation into tokens.

## Supervision and acquisition controls (Figure S2D/E)

| Condition | Held-out gain (pp), mean ± SD | Training cost (k tokens), mean ± SD |
|---|---:|---:|
| Same plans: mean returns | 5.40 ± 0.57 | 237.00 ± 9.53 |
| Same plans: mean returns + return differences | 6.30 ± 0.65 | 284.00 ± 9.71 |
| Ordinary-search data: same acquisition cost | 5.10 ± 0.57 | 253.00 ± 9.45 |

Pair supervision increases mean gain by 16.7% with 19.8% higher training cost. Data-acquisition expense is matched within 2%; total training cost is measured separately.

## Search strategy and improver update (Figure 7C)

| Search | Improver | Gain (pp), mean ± SD | Total cost (k tokens/seed), mean ± SD |
|---|---|---:|---:|
| Greedy | Frozen | 3.70 ± 1.30 | 59.60 ± 6.76 |
| Greedy | GVO | 4.50 ± 1.38 | 60.80 ± 6.75 |
| VATS | Frozen | 4.40 ± 1.56 | 53.10 ± 6.56 |
| VATS | GVO | 5.70 ± 2.16 | 48.00 ± 6.21 |

VATS with GVO gains 2.00 pp over Greedy with a frozen improver while reducing total cost by 19.5%.

[Manifest and checksums](manifest.json) list the CSV files in this package. Figures have a separate [manifest](../figures/manifest.json). Public links and manifest entries are repository-relative.
