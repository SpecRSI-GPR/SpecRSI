"""Configure the algorithm and provenance of manuscript ``SpecRSI (33).pdf``.

The paper's three trainable roles are theta (Drafter), phi (the single action
value Scorer/critic) and psi (independent Policy logits); nu contains registered
menu-generator settings. Eq. (4) freezes all of them during each collection
group. A reference backbone name alone does not establish this execution or
learning contract. Section 4.1 identifies Qwen3.5-35B-A3B, while Appendix F.5
specifies a rank-16 Drafter adapter and separate width-256, four-head relation
branches. Model construction and trained tensor validation live in the learning
backend; the structural preflight here is not certification of those hyperparams
or of trained model performance.

``vats_pdf`` selects the Algorithms 1--2 execution semantics. ``legacy_qv``
remains an explicitly separate historical controller, even with the same PDF
path or text-model family. PDF provenance, runtime algorithm selection, model
configuration and a trained checkpoint are distinct facts in a run record.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
import re
from pathlib import Path

PDF_PATH = "ft_local/SpecRSI (33).pdf"
PDF_SHA256 = "c69c4dd5c470029474ba68c6560284e81145cc26889ad612181e268a96eb4236"
HISTORICAL_PDF_SHA256 = "2a4de176166df8b0e61a50eca8616199436018718a5e99280bcc3da10092f8fc"
PDF_ALGORITHM = "vats_pdf"
LEGACY_ALGORITHM = "legacy_qv"
REFERENCE_BACKBONE_REPO = "Qwen/Qwen3.5-35B-A3B"
REFERENCE_BACKBONE_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
# Publisher config.json identity registered in the BF16 model lock. Git blob
# hashing includes the byte length/header; it is not sha1(file_bytes).
REFERENCE_BACKBONE_CONFIG_BLOB_SHA1 = "2fd92bfc9ee58bd0274ad207d4120c3e7458dde8"


def validate_reference_backbone(config: dict) -> dict | None:
    """Validate an explicitly requested manuscript-(33) training reference.

    Set top-level ``require_reference_backbone: true`` to enforce the pinned
    official BF16 Qwen3.5-35B-A3B snapshot before loading model weights. Explicit
    demo/legacy configurations without that flag retain their existing behavior.
    No GPU allocation or network request occurs here. A returned receipt proves
    configuration/metadata checks, not all shard hashes or model performance;
    the registered downloader checks weight shards and the HF loader checks the
    unquantized model and separate trainable relation branches.

    Section 4.1 names the reference backbone; Appendix F.5 specifies rank 16,
    width 256 and four attention heads. Architecture version 3 is this codebase's
    explicit semantic-node implementation choice, not a numbered paper formula.
    Four relation-attention heads are fixed by the model implementation at width
    256, rather than by a separate YAML attention-head hyperparameter.
    The explicit hybrid-attention LoRA profile covers both full-attention and
    Gated DeltaNet projections. Eager MoE experts preserve the registered frozen
    expert execution path; neither setting changes the likelihood objective.
    """
    required = config.get("require_reference_backbone", False)
    if type(required) is not bool:
        raise ValueError("require_reference_backbone must be an explicit boolean")
    if not required:
        return None
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("Reference backbone requires an explicit model mapping")
    expected = {"backend": "hf", "revision": REFERENCE_BACKBONE_REVISION,
                "dtype": "bfloat16", "architecture_version": 3,
                "head_hidden_dim": 256, "lora_rank": 16,
                "language_model_only": True, "require_unquantized_backbone": True,
                "lora_target_profile": "qwen35_hybrid_attention",
                "experts_implementation": "eager"}
    for field, value in expected.items():
        if field not in model:
            raise ValueError(f"Reference backbone requires explicit model.{field}={value!r}")
        if model[field] != value or type(model[field]) is not type(value):
            raise ValueError(f"Reference backbone requires model.{field}={value!r}, got {model[field]!r}")
    for field, minimum in (("feature_dim", 8), ("option_dim", 12)):
        if type(model.get(field)) is not int or model[field] < minimum:
            raise ValueError(f"Reference backbone requires explicit model.{field} >= {minimum}")
    name = model.get("model_path")
    if not isinstance(name, str) or not name:
        raise ValueError("Reference backbone requires an explicit local model.model_path")
    path = Path(name)
    canonical_directory = path.name == "Qwen3.5-35B-A3B"
    pinned_hf_snapshot = (path.name == REFERENCE_BACKBONE_REVISION and path.parent.name == "snapshots"
                          and path.parent.parent.name == "models--Qwen--Qwen3.5-35B-A3B")
    if not (canonical_directory or pinned_hf_snapshot):
        raise ValueError("Reference backbone model_path must identify official Qwen3.5-35B-A3B BF16, not an 8B or FP8 model")
    metadata_path = path / "config.json"
    if not metadata_path.is_file():
        raise ValueError(f"Reference backbone is missing local publisher metadata: {metadata_path}")
    raw = metadata_path.read_bytes()
    try:
        metadata = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Reference backbone config.json is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("Reference backbone config.json must contain a mapping")
    text_config = metadata.get("text_config", {})
    if not isinstance(text_config, dict):
        raise ValueError("Reference backbone requires publisher text_config metadata")
    if any(section.get("quantization_config") is not None for section in (metadata, text_config)):
        raise ValueError("Reference backbone must be publisher BF16 with no quantization_config; FP8 dequantization is not accepted")
    if metadata.get("model_type") != "qwen3_5_moe" or text_config.get("dtype") != "bfloat16":
        raise ValueError("Reference backbone metadata is not the publisher Qwen3.5 MoE BF16 configuration")
    blob_hash = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob_hash != REFERENCE_BACKBONE_CONFIG_BLOB_SHA1:
        raise ValueError("Reference backbone config.json differs from the pinned publisher Git blob SHA-1")
    return {"repo_id": REFERENCE_BACKBONE_REPO, "revision": REFERENCE_BACKBONE_REVISION,
            "model_path": str(path.resolve()), "dtype": "bfloat16",
            "publisher_config_git_blob_sha1": blob_hash,
            "publisher_config_sha256": hashlib.sha256(raw).hexdigest(),
            "architecture_version": 3, "lora_rank": 16, "head_hidden_dim": 256,
            "relation_attention_heads": 4, "language_model_only": True,
            "require_unquantized_backbone": True,
            "lora_target_profile": "qwen35_hybrid_attention",
            "experts_implementation": "eager",
            "validation_scope": "configured reference and publisher metadata; weight integrity is checked by downloader and loader"}


def algorithm_name(settings: dict) -> str:
    algorithm = settings.get("algorithm", LEGACY_ALGORITHM)
    if algorithm not in {PDF_ALGORITHM, LEGACY_ALGORITHM}:
        raise ValueError(f"Unknown search algorithm: {algorithm}")
    if algorithm == PDF_ALGORITHM:
        if settings.get("eta", 0) != 0 or settings.get("margin", 0) != 0:
            raise ValueError("The supplied PDF accepts the best complete admissible agent; eta/margin must be zero")
        if settings.get("reuse", "all_U") != "all_U" or not settings.get("all_u", True):
            raise ValueError("The PDF reference profile requires direct checks (all_U)")
        if settings.get("controller", "learned") not in {"learned", "heuristic_bootstrap"}:
            raise ValueError("PDF VATS needs separate scorer and policy; legacy rule controllers are separate ablations")
        if settings.get("batch_mode", "joint") != "joint":
            raise ValueError("The PDF reference profile scores whole plans; additive scoring is a separate ablation")
    return algorithm


def vats_config(base_config, settings: dict, *, learned_query_cap: int = 0):
    """Combine execution caps with completion-aware menu settings.

    Eq. (10) selects the best complete admissible version without an extra
    adoption margin or forced final completion. Eq. (5) constructs work options
    under the remaining budget; VATSConfig validates their resource limits.

    The legacy SearchConfig supplies execution caps and common limits; its
    historical margin/reserve/reuse defaults are replaced by the PDF contract.
    Nested ``vats`` settings may override only VATS-specific fields, so cannot
    reintroduce those legacy contracts. A supplied scorer/policy cap overrides
    ``learned_query_cap``; otherwise that known conservative cap initializes the
    query charge and the policy charge settled before the final policy forward
    (Algorithm 2, Eq. (16)). These are registered search expenses, not measured
    evidence that a physical model call consumed that number of tokens.
    """
    from .core.vats import VATSConfig
    from .core.search import SearchConfig

    values = {field.name: getattr(base_config, field.name) for field in fields(base_config)}
    values.update(all_u=True, max_sources=0, check_cap=0, lookup_cost=0,
                  reserve_completion=False, margin=0.)
    options = dict(settings.get("vats", {}))
    allowed = {field.name for field in fields(VATSConfig)} - {field.name for field in fields(SearchConfig)}
    if options.keys() - allowed:
        raise ValueError(f"Nested vats options cannot override shared search contracts: {sorted(options.keys() - allowed)}")
    if "scorer_query_cap" not in options:
        options["scorer_query_cap"] = learned_query_cap
    if "policy_call_cost" not in options and learned_query_cap:
        options["policy_call_cost"] = learned_query_cap
    values.update(options)
    return VATSConfig(**values)


def paper_provenance(config: dict) -> dict:
    """Validate configured input constraints before starting execution."""
    path = Path(config.get("paper_path", PDF_PATH))
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if config.get("paper_sha256") and actual != config["paper_sha256"]:
        raise ValueError("Paper PDF differs from the registered SHA-256")
    if config.get("require_current_paper", False) and actual != PDF_SHA256:
        raise ValueError("The current reference requires the exact SpecRSI (33).pdf SHA-256")
    return {"path": str(path), "sha256": actual,
            "matches_implementation_reference": actual == PDF_SHA256,
            "implementation_reference_sha256": PDF_SHA256,
            "manuscript_version": ("33" if actual == PDF_SHA256 else
                                   "historical_20260926" if actual == HISTORICAL_PDF_SHA256 else "unrecognized"),
            "algorithm": algorithm_name(config.get("search", config.get("collection", {})))}


def require_role_checkpoint(search: dict) -> Path | None:
    """Fail before spending task-model calls if learned weights are unavailable."""
    if search.get("controller") != "learned":
        return None
    path = Path(search.get("role_checkpoint", ""))
    manifest = path / "model_config.json"
    if not manifest.is_file():
        raise ValueError("Missing trained role checkpoint; train the PDF roles before learned evaluation")
    data = json.loads(manifest.read_text())
    if data.get("backend") != "hf":
        raise ValueError("Local task experiments require an HF role checkpoint")
    if algorithm_name(search) == PDF_ALGORITHM:
        if (type(data.get("architecture_version")) is not int or data["architecture_version"] not in {2, 3}
                or data.get("option_dim", 0) < 12
                or data.get("feature_dim", 0) < 8):
            raise ValueError("PDF VATS requires architecture_version=2 or explicit version 3, option_dim >= 12 and feature_dim >= 8; legacy roles cannot be reused")
        if not (path / "heads.pt").is_file() or not (path / "adapters").is_dir():
            raise ValueError("PDF role checkpoint is incomplete: trained heads and adapters are required")
        if data["architecture_version"] == 3:
            snapshot = data.get("frozen_text_snapshot")
            if not isinstance(snapshot, str) or not re.fullmatch(r"frozen_node_text_v1:sha256:[0-9a-f]{64}", snapshot):
                raise ValueError("Version 3 requires a valid frozen_text_snapshot identity")
            import torch
            try:
                heads = torch.load(path / "heads.pt", map_location="cpu", weights_only=True)
            except Exception as error:
                raise ValueError("Version 3 semantic heads cannot be read") from error
            for role in ("value", "policy"):
                projection = heads.get(f"{role}_head.node_text_projection.weight") if isinstance(heads, dict) else None
                if not isinstance(projection, torch.Tensor) or projection.ndim != 2 or min(projection.shape) < 1:
                    raise ValueError("Version 3 requires separate value and policy node text projections")
            # The actual HF loader also checks the exact full head parameter set
            # and recomputes the encoder identity from its frozen model weights.
    return path


def vats_cost_accounting(cost_unit: str, policy_call_cost: int, architecture_version: int = 2) -> dict:
    """Separate metered work from the known price charged before policy choice."""
    accounting = {"budget_unit": cost_unit,
            "policy_charge": "fixed_prepaid_conservative_quota",
            "policy_charge_per_menu": policy_call_cost,
            "scorer_charge": "observed_prompt_tokens" if cost_unit == "tokens" else "measured_wall_ms",
            "solver_charge": "configured_logical_overhead",
            "physical_bill_equals_budget_spend": False,
            "note": "Policy quotas and solver overhead are charged to the search budget; physical inference records are reported separately."}
    if architecture_version == 3:
        accounting.update(
            frozen_node_physical_charge="actual_forward_cache_misses_only",
            scorer_charge=("fixed_record_context_plus_unique_node_input_quota"
                           if cost_unit == "tokens" else "measured_wall_ms_including_node_preparation"),
            cache_independent_scorer_token_quota=cost_unit == "tokens",
            note=("Policy, scorer token quotas and solver overhead are search-budget charges; "
                  "physical records contain only actual forwards and node-encoding cache misses. "
                  "Wall time remains actual cache-dependent elapsed time, not a deterministic group quota."))
    return accounting


def objective_coefficients(settings: dict) -> tuple[float, float]:
    """Validate Eq.16's registered expense/time weights before external work."""
    values = []
    for name in ('cost_lambda', 'time_mu'):
        value = settings.get(name, 0.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite nonnegative number')
        values.append(float(value))
    return tuple(values)


def realized_search_return(result, settings: dict) -> float:
    """Eq. (13), (16): R_g = F(S_end) - F(S_0^-) - lambda*C_g - mu*L_g.

    C_g = sum_t(o_t + kappa_t) includes initial preparation, later preparation,
    failed work, and unfinished work. L_g = sum_t(ell_t^m + ell_t^w) measures
    nonoverlapping intervals on one search clock, rather than adding concurrent
    worker durations. Unlike the paid-state suffix G_t in Eq. (11)/(17), the
    whole-search return includes the initial preparation charge. Acquiring the
    complete initial root and evaluating the frozen successor on heldout tasks
    lie outside this search objective and remain in the physical expense ledger.

    No discount, intermediate forecast reward, adoption bonus or optimizer KL
    penalty is added here. In particular Eq. (21)'s optional reference-policy KL
    is an actor regularizer, not part of the environment return R_g used by the
    leave-one-out group advantage in Eq. (13). Accumulate the phase clock with
    the same compensated sum used by SearchResult.objective_elapsed_seconds so
    reporting and the objective cannot disagree through a different summation
    order when short phases accompany much longer evaluation blocks.
    """
    expense_weight, time_weight = objective_coefficients(settings)
    elapsed = math.fsum(transition.elapsed_seconds for transition in result.transitions)
    return result.final_f - result.initial_f - expense_weight * result.total_cost - time_weight * elapsed
