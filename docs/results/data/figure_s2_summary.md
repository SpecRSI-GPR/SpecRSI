# Figure S2: scoring, evidence, and supervision controls

Mean and SD are computed from the accompanying seed data (SD denominator n-1).

| panel | cohort | condition | metric | unit | n | Mean | SD |
|---|---|---|---|---|---|---|---|
| A | Original | CQ | gain | pp | 5 | 4.19 | 0.944987 |
| A | Original | CR | gain | pp | 5 | 5.34 | 0.709225 |
| A | Matched input | CQ | gain | pp | 5 | 4.26 | 1.51756 |
| A | Matched input | CR | gain | pp | 5 | 5.58 | 1.41845 |
| B | Original | CQ | cost | k tokens/search | 5 | 47.86 | 1.32966 |
| B | Original | CR | cost | k tokens/search | 5 | 44.58 | 1.46185 |
| B | Matched input | CQ | cost | k tokens/search | 5 | 47.86 | 4.18127 |
| B | Matched input | CR | cost | k tokens/search | 5 | 44.34 | 3.62533 |
| C | Evidence | Evidence features hidden | gain | pp | 5 | 3.2 | 0.524404 |
| C | Evidence | Available/missing flags only | gain | pp | 5 | 4.7 | 0.65192 |
| C | Evidence | Observed values only | gain | pp | 5 | 3.8 | 0.62849 |
| C | Evidence | Observed values + availability flags | gain | pp | 5 | 5.8 | 0.764853 |
| C | Evidence | Full evidence | gain | pp | 5 | 6.3 | 0.764853 |
| D | Supervision | Same plans: mean returns | gain | pp | 5 | 5.4 | 0.570088 |
| D | Supervision | Same plans: mean returns + return differences | gain | pp | 5 | 6.3 | 0.65192 |
| D | Supervision | Ordinary-search data: same acquisition cost | gain | pp | 5 | 5.1 | 0.570088 |
| E | Supervision | Same plans: mean returns | training_cost | k tokens | 5 | 237 | 9.53494 |
| E | Supervision | Same plans: mean returns + return differences | training_cost | k tokens | 5 | 284 | 9.70953 |
| E | Supervision | Ordinary-search data: same acquisition cost | training_cost | k tokens | 5 | 253 | 9.44669 |
