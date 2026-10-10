# Reproducibility specifications

The versioned data pools, exact member lists, scientific scenarios, model snapshots, and open-model training profiles below specify the inputs and execution conditions used by the corresponding configurations. Separate development, selection, and audit roles make candidate construction and frozen-candidate evaluation explicit.

## Data versions and members

[Dataset specifications](reproducibility-datasets.json) pin upstream revisions, download URLs, file sizes, SHA-256 fingerprints, normalization, and split rules. [Exact split members](reproducibility-splits.json) contain all 139,865 member IDs across 14 data pools and 42 splits, preserving source order, case, and leading zeroes. Each split includes its normalized JSONL fingerprint and the SHA-256 of its ordered ID list.

| Data pool | Development | Selection | Audit |
|---|---:|---:|---:|
| AIME25 | 13 | 11 | 6 |
| Game24 | 254 | 574 | 534 |
| HotpotQA | 19,622 | 39,351 | 38,879 |
| HoVer | 4,421 | 8,898 | 8,852 |
| IFBench | 48 | 110 | 142 |
| LiveBench Math | 91 | 140 | 137 |
| MATH | 3,750 | 3,749 | 5,000 |
| MBPP | 186 | 375 | 413 |
| PlanBench | 479 | 993 | 798 |
| Polyglot | 50 | 80 | 95 |
| PUPA | 171 | 386 | 344 |
| SWE-bench Lite | 60 | 141 | 122 |
| SWE-bench Verified | 103 | 190 | 207 |
| AIME 2024–2026 extension | 15 | 15 | 60 |

The default split uses seed `20260916` and SHA-256 of grouped or normalized inputs, with development/selection/audit thresholds 0.2/0.6/1.0. MATH retains the original 5,000-test audit pool; PlanBench groups variants by instance and removes one-shot demonstrations; the AIME extension develops and selects on 2024 and audits on 2025–2026. Frozen selection IDs and input hashes determine audit-overlap exclusion. HoVer additionally pins the 5,233,330-document corpus and BM25 index settings.

## Scientific scenarios

[Scenario specifications](reproducibility-science.json) provide all 168 registrations under `specrsi-prospective-science-v1`: 56 cases with development, selection, and audit configurations. The domain counts are PyBaMM 12, AQM 11, FDTD 12, Reactor 10, and IRP 11. Each registration includes complete simulator parameters, seeds, numerical tolerances, evaluator identity, source fingerprints, dependency versions, and runtime conventions. Dependencies include PyBaMM 25.12.2, ns-3 3.44.post0, NumPy 1.26.4, and SciPy 1.17.1.

## Training and deployment

[Training specifications](reproducibility-training.json) provide 16 effective open-model training stages, including seeds, stage epochs, actual optimizer-update counts, semantic budgets, collection sizes, and separate Scorer/Policy configurations. The frozen Qwen3-8B backbone uses BF16 and Drafter LoRA rank 16, alpha 32, dropout 0; relational heads have width 256. SFT uses AdamW at `2e-5`, weight decay `1e-4`, and component batch size 1.

| GVO profile | Optimizer and learning rate | Collection/update schedule |
|---|---|---|
| Relational AdamW with component batches | AdamW `2e-5` | 1–2 stage epochs; profile-specific component updates |
| Relational AdamW with frozen collections | AdamW `2e-5` | 3 epochs, 3 updates; 8 contexts × 8 tasks, group size 3 |
| Relational SGD with frozen collections | Actor `1e-3`, value `5e-4` | 3 epochs, 3 updates; 8 contexts × 4 tasks, group size 3 |

Frozen-collection profiles accumulate gradients over each collection before an optimizer step. SGD uses zero momentum and weight decay, PPO clipping 0.2, and KL weight 0.02 against the frozen SFT actor. These schedules keep training batches distinct from the groups of complete searches used for credit assignment.

[Model specifications](reproducibility-models.json) pin Qwen3-8B (`b968826d…`), Kimi-K2.6 (`7eb5002f…`), and Qwen3.5-35B-A3B-FP8 (`9d1823d2…`) to full immutable revisions and file-level fingerprints. Deployment profiles specify precision, GPU counts, context limits, and serving batch limits. Kimi uses tensor/pipeline parallelism 1/7; Qwen3.5 uses one GPU and context length 32,768.

[Environment specifications](reproducibility-environment.json) detail the eight-H20 host and separate training and inference software stacks. Open-model training uses one H20, Python 3.11.6, PyTorch 2.6.0+cu124, Transformers 4.57.6, and PEFT 0.18.1; serving uses vLLM 0.19.1 and PyTorch 2.10.0+cu128.
