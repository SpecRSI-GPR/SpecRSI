# Reproducing SpecRSI

The reference backbone is the publisher's unquantized BF16 **Qwen/Qwen3.5-35B-A3B**, revision `59d61f3ce65a6d9863b86d2e96597125219dc754`. The executable configurations, complete dataset membership, native evaluation contracts, model file identities, training parameters, and deployment parameters are included in this repository. Data and model weights are obtained from their pinned publishers.

## Reproduction files

| File or directory | Contents |
|---|---|
| `configs/models/qwen35_35b_a3b_bf16.lock.json` | Publisher revision, download URL, all 26 file identities, BF16 format, tokenizer and serving limits |
| `configs/reproduction/model_snapshot.json` | Immutable backbone snapshot and complete file sizes/hashes |
| `configs/reproduction/training_profiles.json` | 54 training profiles with explicit configuration and effective stage parameters; 26 deployed science replica configurations; implementation class defaults and the effective reference profile |
| `configs/reproduction/role_architecture.json` | Complete 190 LoRA target paths, tensor shapes, role parameter counts and head structure |
| `configs/reproduction/role_checkpoint_snapshots.json` | Seven stage checkpoint identities, adapter parameters and cumulative optimizer step counters |
| `configs/reproduction/deployment.json` | Hardware allocation, training and serving dependencies, and actual backend/gateway command arguments |
| `data/reproduction/catalog.json` | All 14 dataset conditions, 42 partitions, counts and metadata paths |
| `data/reproduction/manifests/` | Pinned source URLs, revisions, SHA256 hashes and file sizes |
| `data/reproduction/splits/<dataset>/<split>.json` | Every native member ID in source order, ID-list hash and processed-file identity |
| `data/reproduction/dataset_profiles.json` | Source definitions, preprocessing split policies and subject/domain/language counts |
| `data/reproduction/subsets/` | Exact registered subcohorts, source offsets, grouping, sampling seeds/salts and exclusions |
| `data/reproduction/materialized_subsets.json` | Exact IDs and prepared-file identities for the training subset materializer |
| `data/reproduction/swe_tracks.json` | Selection60, development23, Verified500, filtered Lite and disjoint Lite members |
| `configs/coding/swe_images.lock.json` | Image digests and repository base commits for all 730 SWE instances |
| `data/reproduction/science_scenarios.json` | All 168 expanded science configurations, numerical contracts and dependency records |

Paths are relative to the repository root. `implementation_class_defaults` describes the configurable HF/training classes; `effective_reference_profile` records the complete Qwen3.5 reference settings after configuration and stage overrides. A profile's epoch count, collection size and optimizer update count are different quantities.

## Setup and verification

Run from the repository root with Python 3.11. The default bootstrap installs the package and dependencies. Download flags select additional resources.

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
python scripts/verify_reproduction.py
python scripts/download_model.py --lock configs/models/qwen35_35b_a3b_bf16.lock.json
python scripts/prepare_data.py --datasets all
python scripts/prepare_performance_aime.py
python scripts/verify_reproduction.py --prepared
python scripts/prepare_reproduction_subsets.py
python scripts/verify_reproduction.py --subsets
```

The metadata check validates configuration references, all ordered member lists and their checksums. `--prepared` additionally verifies the downloaded and normalized partition bytes. `--subsets` verifies the materialized training subsets. Each subset records its parent `development` partition; selection/audit names inside a subset describe that subset's internal roles. The materializer uses native IDs and retains the exact member order, task inputs and references, then writes a verified dataset manifest compatible with the benchmark loaders.

## Datasets and partition membership

| Dataset condition | Development | Selection | Audit |
|---|---:|---:|---:|
| AIME25 | 13 | 11 | 6 |
| Game24 | 254 | 574 | 534 |
| HotpotQA distractor | 19,622 | 39,351 | 38,879 |
| HoVer v1.1 | 4,421 | 8,898 | 8,852 |
| IFBench | 48 | 110 | 142 |
| LiveBench Math | 91 | 140 | 137 |
| MATH | 3,750 | 3,749 | 5,000 |
| MBPP | 186 | 375 | 413 |
| PlanBench zero-shot plan generation | 479 | 993 | 798 |
| Polyglot | 50 | 80 | 95 |
| PUPA | 171 | 386 | 344 |
| SWE-bench Lite | 60 | 141 | 122 |
| SWE-bench Verified | 103 | 190 | 207 |
| AIME2024 development / AIME2025–2026 audit | 15 | 15 | 60 |

The 139,865 ordered partition members are fully enumerated in JSON. Each source manifest gives direct download links to an immutable publisher revision and checksums for the raw files. The normalizer preserves native IDs and references. HotpotQA merges its distractor train and validation files; HoVer merges released train/dev; Lite merges released dev/test; MBPP uses all 974 native tasks. Polyglot includes the six language directories and PlanBench includes seven domains. Their complete per-split subtype counts are in `dataset_profiles.json`.

For the common partition rule, normalize an input as `" ".join(prompt.split()).casefold()` and compute its SHA256. A recorded `split_group` replaces the input hash with the group's SHA256 so related instances share their partition. With seed `20260916`, let `u` be the first 64 bits of `SHA256(f"{seed}:{split_hash}")`, divided by `2**64`. The intervals `[0,.2)`, `[.2,.6)`, and `[.6,1)` assign development, selection and audit. Members retain their source traversal order.

MATH keeps the original 5,000 test items in audit. Training inputs are split by SHA256 parity, and the exact test-overlapping item `geometry_train:433` is removed. PlanBench retains domain rules and the final `[STATEMENT]` problem while removing the intervening demonstration; original instance groups are kept together. The AIME extension sorts 2024 inputs by hash, taking 15 development and 15 selection items, and reserves all 2025/2026 items for its audit partition.

`cross_dataset_overlap.json` and `within_dataset_duplicates.json` enumerate exact normalized-input matches. The AIME extension's 2025 audit items also occur in AIME25, and the SWE full Verified track includes the selection60 members. These are defined dataset conditions; independent holdout comparisons use the corresponding frozen membership and exclusion lists. SWE filtering removes audit items whose ID or normalized input matches the frozen selection pool. The complete filtered Lite track has 291 members; the track disjoint from Verified500 has 207.

## Native evaluation and external resources

| Task | Evaluation contract |
|---|---|
| HotpotQA | Answer EM/F1 and supporting-fact/joint metrics; each run fixes its metric variant |
| HoVer | Supporting-document retrieval coverage, with claim-label accuracy recorded separately |
| IFBench | Pinned official verifier; prompt/instruction strict/loose variants |
| PUPA | Explicit judge configuration; quality/privacy utility `(quality + 1 - leakage)/2` |
| AIME | Explicit final-answer extraction followed by exact integer equality in 0–999 |
| LiveBench Math | Task-specific pinned native harness and explicit judge when a task requires one |
| Game24 | Exact rational arithmetic, allowed operators, and conserved input-number multiplicities |
| MBPP | All native `test_list` assertions in the configured isolated execution runtime |
| MATH | Native numeric/symbolic answer parsing and equivalence checking |
| PlanBench | Native PDDL/complete-plan validation |
| Polyglot | Original exercise tests executed by the language's compiler/runtime |
| SWE | Pinned official harness and per-instance image/base-commit/test contract |

HoVer retrieval uses the pinned Wikipedia abstract corpus at `https://huggingface.co/dspy/cache/resolve/ef6a5e72a98b47cef31574a400fea8fe149559a3/wiki.abstracts.2017.tar.gz`. The corpus has 5,233,330 documents, with passages formed as `title + " | " + " ".join(text)`. `data/reproduction/hover_retrieval.json` fixes the source/index identities and BM25 parameters: `bm25s==0.2.14`, `PyStemmer==3.0.0`, `k1=.9`, `b=.4`, `delta=.5`, Lucene scoring/IDF, English stemming, float32 values and int32 indices. Prepare it with `python scripts/prepare_retrieval.py`.

The scorer manifests pin IFBench, LiveBench and the four NLTK resources. Baseline revisions and hashes are under `configs/baselines/`. Download their sources using `python scripts/prepare_baselines.py` for GEPA/DSPy/AFlow/HGM and `python scripts/prepare_upstream.py` for FlashEvolve/TextGrad/Trace. Runtime-specific locks are under `configs/runtimes/`, `configs/coding/`, and `configs/science/`. Baseline recipes explicitly fix their data access, candidate/iteration counts, generation settings and execution budget.

## Training the reference roles

```bash
python scripts/train.py --config configs/training/qwen35_paper33_science.yaml
```

This reference configuration trains on development scenarios with two stages. The native BF16 MoE text backbone, experts and router are frozen. Rank-16 Drafter LoRA has alpha 32 and dropout 0.0; the separate policy and value heads each use width 256, graph architecture version 3, observed-results inputs and a frozen global cache. The text tower is loaded without the vision tower. Hybrid-attention targeting includes linear-attention projections and full-attention Q/K/V/O projections; `role_architecture.json` lists every target and tensor shape. The three trainable roles contain 20,598,274 parameters in total: 16,250,880 Drafter parameters and 2,173,697 parameters in each head.

| Setting | Demonstration warmup | Whole-search GVO |
|---|---|---|
| Epochs | 1 | 1 |
| Collection group size | 2 | 3 |
| Minibatch | 1 mixed supervised unit | One complete frozen collection per update |
| Optimizer | AdamW | Separate actor/value parameter groups in plain SGD |
| Learning rate | `2e-5` | Actor `1e-3`; value `5e-4` |
| Weight decay | `1e-4` | 0 |
| KL coefficient | 0 | .02 |
| KL reference | Behavior snapshot | Frozen SFT actor |
| Forced repetitions | 0 | 2 |
| Exploration | 0 | .1 |

Warmup uses executed demonstrations and the configured stopping/quality rules. Its update count is `ceil(supervised_component_units / batch_size)` per epoch, so it depends on the completed demonstrations. GVO freezes the actor, scorer, menu, root, data and execution contract for the complete collection, differentiates its accumulated objective and performs one optimizer update. `batch_size` does not split this GVO update. Actor terms use ordinary complete searches; forced suffixes contribute critic/pair supervision. Optimizer step counters are cumulative across stage transitions. Checkpoint metadata provide the recorded counters separately from configured epochs and maximum collection sizes.

The reference role context is 2,048 tokens; the semantic graph allocation is separately specified as global 512, node minimum 32, node maximum 1,024 and total 8,192 tokens. Generation uses its configured support, temperature and exploration. Gradient checkpointing is enabled with non-reentrant replay; training forward passes disable the generation cache. Native experts use the eager implementation. Each domain uses its own model/process on one GPU. The complete profiles preserve all effective generation/search budgets, action/child/menu limits, seeds, warmup stopping rules, group sizes, stage losses and optimizer settings.

`configs/training/science_replicas/` provides the deployed domain/seed configurations. Prompt, workflow, coding and task-model GRPO profiles are under `configs/training/`; the task-model GRPO baseline has its own task-specific epoch/group/data-size/learning-rate/generation settings and uses the unadapted pretrained actor as its fixed reference. It trains the task actor; improver heads are frozen and unused. Tiny CPU configurations exercise the search/training interfaces with their explicitly synthetic tasks.

The learned scientific search runner is `scripts/run_paper_science.py --config <search-config.yaml>`. Its configured role checkpoint must be the checkpoint produced by the matching training recipe. `scripts/run_paper_science_comparison.py --config <comparison-config.yaml>` provides the matched comparison and freeze barrier. The ordinary experiment/baseline launchers select their complete recipes from `configs/experiments/` and `configs/baselines/`.

## Scientific contracts

There are 56 case IDs per scenario set: PyBaMM 12, AQM 11, FDTD 12, Reactor 10 and IRP 11. Development, selection and audit each have a separate scenario configuration, giving 168 fully expanded registrations. `configs/science/` contains the base configurations, case perturbations and split overrides; `science_scenarios.json` additionally records dependency versions, distribution hashes and numerical runtime contracts.

PyBaMM uses the SPM/OKane2022 model with lumped thermal, SEI and irreversible plating, three temperatures, 30 aging cycles, the specified SOC/current/temperature/plating constraints and the reference-capacity lifetime estimator. AQM runs native ns-3 TCP NewReno traffic on a dumbbell topology, three traffic modes and two seeds per mode, using goodput/delay/loss/Jain utility. FDTD uses the 480×320 TM Yee grid, 10,000 steps, two wavelengths, reference-waveguide normalization and the configured flux/settling constraints. Reactor integrates the six-state semibatch system with Radau, ten control segments, three cooling scenarios, temperature/feed constraints and native product utility. IRP uses the configured geographic customers, demand seeds and periods, integer routing/inventory constraints and its MILP solver contract. Every numerical value, seed, initial candidate, admissibility bound and case perturbation is included in the corresponding JSON files and evaluator implementation.

SpecRSI's implementation exposes joint batch values, executed-result observations, whole-search return supervision and explicit draft/native-execution/head costs. Candidate selection operates on completed evidence under the registered budget; comparison runners bind frozen candidates, roles, configurations and paired audit members. These mechanisms are directly inspectable in the source and reproduce the intended search and training behavior.

## Hardware, software and serving

The deployment has eight NVIDIA H20 GPUs with 97,871 MiB each and NVIDIA driver `575.57.08`. GPU 0 serves the reference backbone; GPUs 1–5 run Reactor, IRP, AQM, FDTD and PyBaMM respectively. The CPU is AMD EPYC 9K84, with 384 logical CPUs and 2,375,288,992 KiB of system memory. Full recorded CPU/GPU allocation and GPU clock/power settings are in `deployment.json`.

Training uses Python 3.11.6, Torch `2.6.0+cu124`, Transformers `5.17.0`, PEFT `0.19.1`, Hugging Face Hub `1.5.0`, Tokenizers `0.23.1`, Accelerate `1.14.0`, Safetensors `0.8.0`, FLA Core `0.3.2` and causal-conv1d `1.5.3.post1`; Torch's CUDA runtime is 12.4. The independent serving environment uses Torch `2.10.0`, vLLM `0.19.1`, Transformers `4.57.6`, Hub `0.36.2` and Tokenizers `0.22.2`. The two environment locks are inventories of the recorded task-specific packages. Install this source repository with `pip install -e .`; the CUDA compiler/native-kernel requirements apply when building native extensions.

```bash
python scripts/deploy_qwen35_bf16.py --python .venv-serving/bin/python
```

The BF16 backend serves a 32,768-token context with one active sequence, 2,048 batched tokens, eager safetensors loading, text-only operation, the `qwen3` reasoning parser and Triton GDN prefill. Its memory target is 90,000,000,000 bytes. The backend listens at `127.0.0.1:18086`; the budget gateway at `127.0.0.1:18085` uses the same pinned tokenizer and chat template for accounting. The backend uses `--generation-config vllm`; request-level sampling parameters are fixed by the calling recipe. The publisher's `generation_config.json` is included as snapshot metadata, while executable generation limits come from the selected recipe.

Model weights remain external at the immutable publisher URL in the model lock. All model-file identities, tokenizer configuration, publisher architecture, adapter/head tensor structure, role-stage configurations and recorded checkpoint identities are available as lightweight metadata.
