"""Auditable SFT -> refreshed GVO orchestration using actual stage checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from .trainer import merge_learning_signals


def stage_configs(config, output, *, epochs=None):
    """Resolve stage overrides before any paid collection or training starts."""
    output = Path(output).resolve()
    if config.get("collection", {}).get("algorithm") != "vats_pdf":
        raise ValueError("The two-stage entry point requires the explicit vats_pdf profile")
    stages = []
    for stage in ("warmup", "gvo"):
        resolved = copy.deepcopy(config)
        overrides = copy.deepcopy(resolved.pop(stage, {}))
        resolved.pop("warmup" if stage == "gvo" else "gvo", None)
        for key, value in overrides.items():
            if key in {"collection", "training"}:
                resolved.setdefault(key, {}).update(value)
            else:
                resolved[key] = value
        resolved["stage"] = stage
        resolved["output"] = str(output / stage)
        resolved["lineage_registry"] = str(output / "lineage_splits.json")
        resolved["epochs"] = epochs if epochs is not None else overrides.get("epochs", 1 if stage == "warmup" else config.get("epochs", 1))
        if not isinstance(resolved["epochs"], int) or resolved["epochs"] < 1:
            raise ValueError("Both stages require a positive integer number of epochs")
        if stage == "warmup":
            required_losses = ("plan", "value", "edit") if resolved.get("model", {}).get("backend") == "hf" else ("plan", "value")
            if any(resolved.get("training", {}).get(f"{key}_weight", 1.) <= 0 for key in required_losses):
                raise ValueError("A complete PDF warm-up requires positive loss weights for every initialized role")
            if not resolved.get("data"):
                resolved["collection"].setdefault("demonstration_policy", "complete_then_draft")
                from .demonstrations import demonstration_bridge_kwargs
                demonstration_bridge_kwargs(resolved["collection"])
                resolved["collection"]["forced_repetitions"] = 0
                resolved["collection"]["exploration"] = 0.
            # Offline demonstration files and an online demonstration collector
            # are distinct sources; do not recollect by accident.
            elif "collector" not in overrides:
                resolved.pop("collector", None)
        else:
            resolved["collection"].pop("demonstration_policy", None)
            resolved["collection"].pop("demonstration_stopping", None)
            resolved.pop("data", None)
            resolved["resume"] = str(output / "warmup" / "checkpoint")
            if not resolved.get("collector"):
                raise ValueError("GVO after SFT must freshly collect with the updated roles")
        # Reject impossible menus before loading an HF backbone or acquiring
        # even the common root's paid task outcomes.
        from specrsi.core.search import SearchConfig
        from specrsi.paper_profile import vats_config
        common = {key: resolved["collection"][key] for key in ("max_actions", "max_candidates", "max_children",
            "exploration", "temperature", "execution_cap", "draft_cap") if key in resolved["collection"]}
        from .frozen_text import semantic_input_cap
        from .demonstrations import DemonstrationQuality
        DemonstrationQuality.from_settings(resolved["collection"].get("demonstration_quality"))
        vats_config(SearchConfig(planning_cost=0, **common), resolved["collection"],
            learned_query_cap=semantic_input_cap({"max_length": 1, **resolved.get("model", {})}))
        stages.append(resolved)
    return stages


def run_two_stage(config, output, *, root, epochs=None):
    """Initialize roles with SFT, then execute complete collection/update rounds.

    Warm-up minimizes Eq. (18)'s separate empirical edit/plan negative log
    likelihoods and Eq. (11)'s observed value loss, producing Theta_0. The
    resumed GVO stage implements the two-level Eq. (4) sequence:
        (c_{k+1}^{(g)},D_k^{(g)}) = Search_{Theta_k}(c_k), g=1,...,G,
        D_k = union_{g=1}^{G} D_k^{(g)},
        Theta_{k+1} = GVO(Theta_k,D_k),
        Theta_k=(theta_k,psi_k,phi_k,nu),  bar_phi_k=phi_k.
    The launched training loop collects every group member before invoking
    the outer optimizer. The SFT checkpoint supplies actual initial weights;
    an epoch's updated roles are used only by subsequent fresh collection.
    Drafter theta, Policy psi and Scorer phi are trainable, while the menu
    settings nu and the verification contract stay fixed across these steps.

    Accepted task agents c_{k+1}^{(g)} are outputs of separate inner searches.
    Carrying/resetting their task root for another round is independent of
    updating the improver Theta. A completed stage therefore checks the
    recorded update counts and role coverage as well as publishing weights;
    it does not reinterpret an inner-search leaf as a separate GVO sample.
    """
    output, root = Path(output).resolve(), Path(root).resolve()
    stages = stage_configs(config, output, epochs=epochs)
    output.mkdir(parents=True, exist_ok=False)
    (output / "registration.json").write_text(json.dumps({
        "pipeline": "pdf_sft_then_refreshed_gvo_v1", "stages": stages,
        "demonstrations": "Executed development edits and legal teacher plans",
        "selection_rule": "final GVO checkpoint; no audit-based checkpoint selection"}, indent=2))
    started = time.monotonic()
    evidence = []
    try:
        for resolved in stages:
            stage = resolved["stage"]
            path = output / f"{stage}.yaml"
            path.write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False))
            subprocess.run([sys.executable, str(root / "scripts/train.py"), "--config", str(path)],
                cwd=root, check=True)
            metrics_path = output / stage / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("source_changed_during_run") or not (output / stage / "checkpoint").is_dir():
                raise RuntimeError("A stage did not publish a validated frozen checkpoint")
            initial_steps = metrics.get('initial_optimizer_steps')
            updates = metrics.get('optimizer_updates')
            final_steps = metrics.get('history', [{}])[-1].get('steps') if metrics.get('history') else None
            v3 = resolved.get("model", {}).get("architecture_version", 2) == 3
            minimum_updates = 0 if v3 and stage == "gvo" else 1
            if (type(initial_steps) is not int or type(updates) is not int or updates < minimum_updates
                    or final_steps != initial_steps + updates):
                raise RuntimeError('Each SFT/GVO stage must perform actual new optimizer updates')
            signal = merge_learning_signals(row.get("learning_signal") for row in metrics.get("history", []))
            if v3 and not signal["diagnostics_complete"]:
                raise RuntimeError("v3 training must report observed role updates, including legal zero-advantage groups")
            if v3 and signal["optimizer_steps"] != updates:
                raise RuntimeError("Observed v3 role evidence disagrees with this stage's optimizer update count")
            training_evidence = Path(metrics["checkpoint"]) / "training_evidence.json"
            if v3 and not training_evidence.is_file():
                raise RuntimeError("v3 checkpoint is missing its explicit training evidence manifest")
            if stage == "warmup":
                totals = {key: sum(row.get("training_examples", {}).get(key, 0) for row in metrics["history"])
                    for key in ("plan", "edit", "value")}
                required = ("plan", "value", "edit") if resolved.get("model", {}).get("backend") == "hf" else ("plan", "value")
                if any(totals[key] == 0 for key in required):
                    raise RuntimeError(f"Incomplete SFT role coverage: {totals}; collect actual verified edits/plans/returns before GVO")
                if "edit" in required and not sum(row.get("trainable_edit_tokens", 0) for row in metrics["history"]):
                    raise RuntimeError("SFT must include observed non-forced draft tokens, not only fixed grammar punctuation")
                warmup_source_hashes = metrics["source_hashes"]
            elif metrics["source_hashes"] != warmup_source_hashes:
                raise RuntimeError("Source changed between SFT and GVO; refusing to publish the two-stage checkpoint")
            evidence.append({"stage": stage, "metrics": str(metrics_path),
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                "checkpoint": metrics["checkpoint"], "learning_signal": signal,
                "training_evidence": str(training_evidence) if training_evidence.exists() else None,
                "training_evidence_sha256": hashlib.sha256(training_evidence.read_bytes()).hexdigest()
                    if training_evidence.exists() else None})
        (output / "checkpoint").symlink_to("gvo/checkpoint", target_is_directory=True)
        summary = {"pipeline": "pdf_sft_then_refreshed_gvo_v1", "complete": True,
            "elapsed_seconds": time.monotonic() - started, "stages": evidence,
            "checkpoint": str(output / "checkpoint"), "sota_established": False,
            "gvo_learning_signal": evidence[-1]["learning_signal"],
            "all_roles_updated_in_gvo": evidence[-1]["learning_signal"]["all_actor_roles_updated"]
                and evidence[-1]["learning_signal"]["critic_updated"],
            "completion_scope": "Both stages executed and checkpoint saved; complete does not imply every role received a nonzero update or that performance improved."}
        (output / "metrics.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"complete": False, "completed_stages": evidence,
            "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.monotonic() - started}, indent=2))
        raise
