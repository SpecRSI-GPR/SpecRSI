# Reproducibility specifications

This reference configuration fixes data membership, the native Qwen3.5-35B-A3B snapshot, scientific scenarios, and execution parameters. Shared starting conditions, verification contracts, and resource accounting preserve a consistent comparison while allowing SpecRSI to adapt its allocation of edits and checks.

## Data versions and exact splits

[Dataset specifications](reproducibility-datasets.json) fix upstream revisions, source URLs, preprocessing, file sizes, and SHA-256 fingerprints. [Exact split members](reproducibility-splits.json) provide all 139,865 ordered IDs in 14 data pools and 42 splits, together with normalized JSONL and ordered-member fingerprints.

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

The default split hashes grouped or whitespace-normalized, case-folded inputs with seed `20260916`; thresholds 0.2 and 0.6 define development, selection, and audit. MATH retains the original 5,000 test problems for audit, hashes the training pool into development/selection, and removes `geometry_train:433` because its input matches a test problem. PlanBench removes one-shot demonstrations and groups equivalent instance variants. The AIME extension divides the 2024 problems into 15 development and 15 selection members and reserves the 2025–2026 problems for its audit pool.

[Subset specifications](reproducibility-subsets.json) additionally provide campaign members, original source roles, sampling seeds, source offsets, hashes, overlap groups, and exclusions. This includes the complete SWE selection/development/audit tracks. Frozen selection IDs and normalized input hashes define audit-overlap exclusion. The AIME extension's holdout is relative to its own 2024 development/selection pool; its 2025 members also appear in the separate AIME25 pool. HoVer pins the 5,233,330-document corpus to revision `ef6a5e72a98b47cef31574a400fea8fe149559a3`, with BM25S 0.2.14, PyStemmer 3.0.0, `k1=0.9`, `b=0.4`, English stemming, and retrieval top-k 7/7/10.

## Scientific scenarios

[Scenario specifications](reproducibility-science.json) define 56 cases and all 168 development/selection/audit registrations under `specrsi-prospective-science-v1`. Each registration includes simulator parameters, seeds, tolerances, hard constraints, source fingerprints, dependency versions, and numerical execution conditions.

| Domain | Cases | Core configuration |
|---|---:|---|
| PyBaMM | 12 | SPM; OKane2022; lumped thermal dynamics; 15/25/35°C; 30 aging cycles; IDAKLU `rtol=1e-6`, `atol=1e-7`; three charging stages |
| AQM | 11 | ns-3 TCP NewReno dumbbell; four flows; 8 s simulations; long/burst/mixed traffic × two seeds; 100 Mb/s access and 20 Mb/s bottleneck |
| FDTD | 12 | Two-dimensional TM Yee solver; 480×320 grid; 0.025 μm cells; Courant 0.65; 10,000 steps; 1.31/1.55 μm wavelengths with straight-waveguide normalization |
| Reactor | 10 | Six-state reaction/thermal model; ten 360 s control segments; three cooling scenarios; Radau `rtol=1e-8`, `atol=1e-9`, maximum step 10 s |
| IRP | 11 | Eight stores; 36 episodes of 180 days; four-day shelf life; Poisson demand; three-day forecasts; all purchase/holding/waste/shortage/routing costs included |

The registered scientific protocol trains on development cases 0–1, searches development cases 4–5, and evaluates frozen candidates on selection cases 6–7. It specifies five independent training seeds (`2026101001`–`2026101005`) and three nested search labels (0–2). Matched methods share the derived search seed, complete root evidence, numeric candidate space, and verification constraints. Each method receives a 3,000,000 ms cost budget, 1,200 s search wall-time limit, 24 primitive attempts, two native verification workers, and at most four candidates including the root. The full action/menu limits and method-specific settings are in [evaluation specifications](reproducibility-evaluation.json).

Complete, admissible outcomes determine acceptance. PyBaMM uses mean predicted lifetime/2,000; AQM combines goodput, p95 delay, loss, and Jain fairness; FDTD combines desired and cross-channel transmissions; Reactor uses mean `(n_C−n_D)/85`; IRP uses mean `1−cost/(12×demand)` minus `0.15×SD` with `ddof=0`. The aggregate native utility and the bounded per-item representation `0.5+atan(native_score)/π` remain separately named. This keeps completion, feasibility, and resource accounting explicit while comparing methods under shared conditions.

## Evaluation and implementation versions

[Evaluation specifications](reproducibility-evaluation.json) fix task metrics, initial instructions, the registered science protocol, and separately identified callable baseline templates. The official model tokenizer and chat template use the same immutable Qwen3.5-35B-A3B BF16 revision; scientific drafting disables thinking and reserves space for the complete edit and EOS.

| Task | Evaluation |
|---|---|
| HotpotQA | Answer F1; EM, supporting-fact, and joint metrics retained separately |
| HoVer | Coverage of the gold supporting-document set by actual retrieval |
| IFBench | Official instruction checks; prompt/instruction × strict/loose; default prompt-loose |
| PUPA | `(quality + 1 − leakage)/2`; both answer orders used for quality; six model calls |
| AIME | Extracted integer in 0–999, exact match |
| LiveBench Math | Pinned official task-specific scorers; named local-judge fallback |
| MATH | Native numerical/symbolic equivalence in an isolated sandbox |
| Game24 | Exact rational arithmetic; each input used exactly once |
| MBPP | Complete original `test_list` |
| PlanBench | PDDL transitions, preconditions, and goal satisfaction |
| Polyglot | Original language-specific compilation and tests |
| SWE-bench | Official per-instance resolved result with pinned repository commit and container |

The IFBench revision is `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`; LiveBench harness revision is `0edef1710338da97d78e071e3c3ac06426a937a8`; NLTK resource revision is `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`. [SWE instance identities](reproducibility-swe-images.json) map all 730 instances in the deduplicated Lite/Verified union to their base commits and OCI digests. Baseline upstream commits and local implementation fingerprints are included in the evaluation JSON. Reference tests are read-only and network-isolated; sandbox defaults are 180 s and 8,192 MiB, with the MATH runtime using 20 s and 2,048 MiB. Every model/judge and verification call is charged to its corresponding resource ledger.

## Model and learned roles

The reference configuration pins `Qwen/Qwen3.5-35B-A3B` to revision `59d61f3ce65a6d9863b86d2e96597125219dc754`, using the official unquantized BF16 text backbone and matching tokenizer/chat template. The backbone, router, and experts stay frozen. Drafter LoRA uses rank 16, alpha 32, and dropout 0, covering full-attention Q/K/V/O and Gated DeltaNet input/output/gate projections. Scorer and Policy use separate width-256 branches with four attention heads. Trainable adapters and branches are stored in FP32.

[Model specifications](reproducibility-models.json) include the publisher configuration, generation protocol, all 26 snapshot files and SHA-256 fingerprints (71,927,083,212 bytes), and the 190 adapted projection paths. Other evaluated backbones use provider APIs; these local training settings apply to the reference implementation.

## Training, seeds, and checkpoints

| Parameter | Reference setting |
|---|---|
| SFT | One epoch; AdamW learning rate `2e-5`, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`; component minibatch 1; gradient-norm cap 1; no scheduler; warmup collection group 2 |
| GVO | Two frozen collection/update rounds; three independent complete searches/group; at most one joint SGD step per complete collection; actor/Scorer rates `1e-3` / `5e-4`; zero momentum and weight decay; no scheduler or gradient clipping |
| Objective | PPO clip 0.2; KL coefficient 0.02 against the fixed post-SFT actor; pair coefficient 1 |
| Collection | Exploration 0.1; temperature 1; two charged forced continuations per arm at state index 0, at most two feasible arms, used for Scorer/pair supervision |
| Deployment | Exploration 0; maximize Policy preference over feasible work |
| Role inputs | Sequence cap 2,048 tokens; gradient checkpointing; four CPU threads/process |
| Frozen semantic encoding | Total 8,192-token allocation across nodes: 512 global, node minimum 32 / maximum 1,024; head-tail truncation and redistributable node budget |
| Initialization | Five seeds per scientific domain: `2026101001`–`2026101005` |
| Nested search | Labels 0, 1, 2; paired seed = integer from the first eight hex digits of SHA-256 of compact JSON `[training_seed,nested_search_seed]` |
| GVO member seeds | `Random(seed + 1000000 * round_index).sample(range(2**63 - 1), group_size)`; forced-continuation stream XOR `0x534349` |

[Training specifications](reproducibility-training.json) define collection, credit assignment, and update semantics. [Complete training configurations](reproducibility-training-configs.json) provide 54 callable configurations, the 25 domain/seed trial configurations with their overrides, and source fingerprints. Separately configured task-answer GRPO uses its own objective and epoch settings. Configuration entries specify protocols; [checkpoint identities](reproducibility-checkpoints.json) identify saved states and their optimizer iteration counts.

Checkpoints restore the Drafter adapter, Scorer/Policy branches, tokenizer, model/training configuration, optimizer/step state, and fixed SFT actor reference against the pinned BF16 base. The five formal warmup checkpoints have 27/14/24/22/19 saved optimizer steps for PyBaMM/AQM/FDTD/Reactor/IRP respectively. SFT step counts depend on effective supervised batches; each complete GVO collection adds at most one joint update.

| Immutable registration | SHA-256 |
|---|---|
| Model lock | `922b07c49ea65f7deabd0844e6fd2a431da11ae220f9e67c181b3b17ab633d9f` |
| Run registration | `290004662729d2536f1ed4dde8f305b3cc0c68de2f099a20349f663ef97c02dc` |
| Source/configuration snapshot | `45ad9238d16b0b8b64004ddea8ee8ef01104e147709fe04147c11afad565f282` |

## Hardware and software

The eight-H20 host has 97,871 MiB per GPU, two AMD EPYC 9K84 96-core CPUs, and NVIDIA driver 575.57.08. Reference execution assigns one H20 to each scientific domain and a separate H20 to model serving, without cross-device gradient synchronization. BF16 serving uses text-only/eager execution, a 32,768-token context, one concurrent sequence, at most 2,048 batched tokens, and Triton Gated DeltaNet prefill.

| Environment | Pinned versions |
|---|---|
| Training | Python 3.11.6; PyTorch `2.6.0+cu124`; Transformers `5.17.0`; PEFT `0.19.1`; Accelerate `1.14.0`; Safetensors `0.8.0`; FLA Core `0.3.2`; causal-conv1d `1.5.3.post1` |
| Serving | PyTorch `2.10.0`; vLLM `0.19.1`; Transformers `4.57.6` |
| Scientific evaluators | PyBaMM `25.12.2`; PyBaMM solvers `0.3.3`; CasADi `3.6.7`; ns-3 `3.44.post0`; NumPy `1.26.4`; SciPy `1.17.1`; Numba `0.66.0`; Pyomo `6.9.5`; HiGHS `1.12.0` |

Use the separate pinned training and serving environments in [environment specifications](reproducibility-environment.json), which also provide package manifests and their fingerprints.
