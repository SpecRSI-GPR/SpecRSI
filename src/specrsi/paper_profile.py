"""Configure verification-aware search, role checkpoints, and objective costs."""
from __future__ import annotations

from dataclasses import fields
import json
import math
import re
from pathlib import Path

PDF_ALGORITHM = "vats_pdf"
LEGACY_ALGORITHM = "legacy_qv"


def algorithm_name(settings: dict) -> str:
    algorithm = settings.get("algorithm", LEGACY_ALGORITHM)
    if algorithm not in {PDF_ALGORITHM, LEGACY_ALGORITHM}:
        raise ValueError(f"Unknown search algorithm: {algorithm}")
    if algorithm == PDF_ALGORITHM:
        if settings.get("eta", 0) != 0 or settings.get("margin", 0) != 0:
            raise ValueError("VATS accepts the best complete admissible agent; eta/margin must be zero")
        if settings.get("reuse", "all_U") != "all_U" or not settings.get("all_u", True):
            raise ValueError("VATS requires direct checks (all_U)")
        if settings.get("controller", "learned") not in {"learned", "heuristic_bootstrap"}:
            raise ValueError("VATS requires separate scorer and policy callbacks")
        if settings.get("batch_mode", "joint") != "joint":
            raise ValueError("VATS requires whole-plan scoring")
    return algorithm


def vats_config(base_config, settings: dict, *, learned_query_cap: int = 0):
    """Combine execution caps with completion-aware menu settings.

    Eq. (10) selects the best complete admissible version without an extra
    adoption margin or forced final completion. Eq. (5) constructs work options
    under the remaining budget; VATSConfig validates their resource limits.
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


def algorithm_metadata(config: dict) -> dict:
    """Return the validated search algorithm selected by the caller."""
    return {"algorithm": algorithm_name(config.get("search", config.get("collection", {})))}


def require_role_checkpoint(search: dict) -> Path | None:
    """Fail before spending task-model calls if learned weights are unavailable."""
    if search.get("controller") != "learned":
        return None
    path = Path(search.get("role_checkpoint", ""))
    manifest = path / "model_config.json"
    if not manifest.is_file():
        raise ValueError("Missing trained role checkpoint; train the roles before learned evaluation")
    data = json.loads(manifest.read_text())
    if data.get("backend") != "hf":
        raise ValueError("Local task experiments require an HF role checkpoint")
    if algorithm_name(search) == PDF_ALGORITHM:
        if (type(data.get("architecture_version")) is not int or data["architecture_version"] not in {2, 3}
                or data.get("option_dim", 0) < 12
                or data.get("feature_dim", 0) < 8):
            raise ValueError("VATS requires architecture_version=2 or 3, option_dim >= 12 and feature_dim >= 8")
        if not (path / "heads.pt").is_file() or not (path / "adapters").is_dir():
            raise ValueError("Role checkpoint is incomplete: trained heads and adapters are required")
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
    """
    expense_weight, time_weight = objective_coefficients(settings)
    elapsed = sum(transition.elapsed_seconds for transition in result.transitions)
    return result.final_f - result.initial_f - expense_weight * result.total_cost - time_weight * elapsed
