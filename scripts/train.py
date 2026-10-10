#!/usr/bin/env python3
"""Execute supervised warm-up and complete-search outer learning rounds.

Eq. (4) separates the two clocks used by this entry point:

    (c_{k+1}^{(g)}, D_k^{(g)}) = Search_{Theta_k}(c_k),  g=1,...,G;
    D_k = union_{g=1}^{G} D_k^{(g)};
    Theta_{k+1} = GVO(Theta_k, D_k).

Here k is an outer learning round (``epoch`` below); g identifies an independent
complete inner search; Theta_k=(theta_k,psi_k,phi_k,nu) contains the Drafter,
Policy, Scorer, and fixed menu rules. A collector returns ``groups`` of complete
searches, plus observed scorer-only continuations and pair labels. Each group
has one common root, verification contract, starting budget, worker count, and
parameter snapshot. Several root groups may be collected in a single epoch;
their leave-one-out baselines are computed within groups, never across roots.

Collection runs before ``trainer.gvo_epoch`` and cannot interleave parameter
updates with an unfinished search. The returned successor c_{k+1}^{(g)} and its
acquired evidence remain immutable when the improver parameters change. The
next epoch calls the collector again with the updated model. Carrying or
resetting the task root is a separate collector setting, rather than an implied
consequence of overwriting theta, psi, or phi.

Eq. (18) first initializes the actor roles with executed edits and legal plans;
Eq. (11) initializes the Scorer from observed returns. The GVO stage then uses
Eq. (13) whole-search returns and advantages, Eq. (20) actor likelihood ratios,
Eqs. (11)--(12) scorer losses, and the Eq. (23) parameter update. The evaluator
contract and menu-generator rules stay fixed across both stages.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import importlib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml

from specrsi.learning import CheckerAudit, EditExample, SpecRSITrainer, TinyModelConfig, TinyRoleModel, TrainingConfig, split_by_lineage
from specrsi.learning.records import PairwiseValueExample, _decision, load_jsonl, save_jsonl, trajectory_from_dict


def source_hashes():
    paths = set((ROOT / "src/specrsi").rglob("*.py"))
    paths.add(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def parse_records(records):
    data = {"audits": [], "plans": [], "trajectories": [], "edits": [], "pairs": []}
    for raw in records:
        raw = dict(raw)
        kind = raw.pop("record_type")
        if kind == "audit":
            raw["features"] = tuple(raw.get("features", ()))
            data["audits"].append(CheckerAudit(**raw))
        elif kind == "plan":
            data["plans"].append(_decision(raw["decision"]))
        elif kind == "trajectory":
            data["trajectories"].append(trajectory_from_dict(raw))
        elif kind == "edit":
            raw["input_ids"], raw["labels"] = tuple(raw["input_ids"]), tuple(raw["labels"])
            raw["token_decisions"] = tuple(_decision(d) for d in raw.get("token_decisions", ()))
            data["edits"].append(EditExample(**raw))
        elif kind == "pair":
            raw["left"], raw["right"] = _decision(raw["left"]), _decision(raw["right"])
            for key in ("left_returns", "right_returns", "left_seeds", "right_seeds"):
                raw[key] = tuple(raw[key])
            data["pairs"].append(PairwiseValueExample(**raw))
        else:
            raise ValueError(f"Unknown record_type: {kind}")
    return data


def create_model(config, resume):
    raw = dict(config.get("model", {}))
    backend = raw.pop("backend", "tiny")
    if resume:
        saved = json.loads((Path(resume) / "model_config.json").read_text())
        if "architecture_version" in raw and raw["architecture_version"] != saved.get("architecture_version"):
            raise ValueError("Explicit model architecture_version differs from the resumed checkpoint; train a new version instead")
        saved_backend = saved["backend"]
        if "backend" in config.get("model", {}) and backend != saved_backend:
            raise ValueError("Explicit model backend differs from the resumed checkpoint")
        from specrsi.learning.frozen_text import SEMANTIC_BUDGET_FIELDS, semantic_budget
        from specrsi.learning.hf import HFModelConfig
        for field in ("max_length", "graph_schema", "role_observation_version", "frozen_global_cache", *SEMANTIC_BUDGET_FIELDS):
            if field in raw and raw[field] != saved.get(field, HFModelConfig.__dataclass_fields__[field].default):
                raise ValueError(f"Explicit model {field} differs from the resumed checkpoint; register and train a new observation profile")
        # Validate the saved profile before allocating the backbone. Undeclared
        # resume settings inherit the checkpoint; explicit settings must match.
        semantic_budget(saved)
        if saved_backend == "tiny":
            return TinyRoleModel.load(resume, config.get("device", "cpu"))
        from specrsi.learning.hf import HFRoleModel
        return HFRoleModel.load(resume, raw.get("device"))
    if backend == "tiny":
        return TinyRoleModel(TinyModelConfig(**raw)).to(config.get("device", "cpu"))
    if backend == "hf":
        from specrsi.learning.hf import HFModelConfig, HFRoleModel
        return HFRoleModel(HFModelConfig(**raw))
    raise ValueError("model.backend must be tiny or hf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["demo", "warmup", "gvo", "two_stage"])
    parser.add_argument("--data", help="Verified training records in caller-provided JSONL")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--collector", help="module:function(model, round_index, config) -> isolated complete-search groups")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    from specrsi.paper_profile import algorithm_name, algorithm_metadata
    algorithm = algorithm_name(config.get("collection", {}))
    paper = algorithm_metadata(config) if algorithm == "vats_pdf" else None
    if algorithm == "vats_pdf":
        training = config.get("training", {})
        if training.get("estimator", "gvo") != "gvo" or training.get("reward_mode", "complete_search") != "complete_search":
            parser.error("PDF GVO requires leave-one-out complete-search returns; GRPO/item rewards are separate ablations")
    resume = args.resume or config.get('resume')
    stage = args.stage or config.get("stage", "warmup")
    output = Path(args.output or config.get("output", f"artifacts/training/{stage}"))
    if stage == "two_stage":
        if args.data or args.collector or args.resume:
            parser.error("Configure two-stage data/collector/resume per stage in the YAML")
        from specrsi.learning.pipeline import run_two_stage
        run_two_stage(config, output, root=ROOT, epochs=args.epochs)
        return
    if algorithm == "vats_pdf" and output.exists():
        parser.error("PDF training requires a new --output directory; retain previous checkpoints and traces")
    epochs = args.epochs if args.epochs is not None else config.get("epochs", 1)
    if epochs < 1:
        parser.error("--epochs must be positive")
    data_path = args.data or config.get("data")
    collector_path = args.collector or config.get("collector")
    if stage != "demo" and not data_path and not collector_path:
        parser.error("A caller-provided --data JSONL or --collector is required.")
    initial_source_hashes = source_hashes()
    torch.manual_seed(config.get("training", {}).get("seed", 0))
    torch.set_num_threads(config.get("cpu_threads", 2))
    model = create_model(config, resume)
    if algorithm == "vats_pdf" and (model.config.option_dim < 12 or model.config.feature_dim < 8):
        parser.error("PDF VATS role checkpoints require option_dim >= 12 and feature_dim >= 8")
    trainer = SpecRSITrainer(model, TrainingConfig(**config.get("training", {})))
    if resume:
        trainer.restore_optimizer(resume,
            gvo_update_schedule_override=config.get("training", {}).get("gvo_update_schedule"),
            optimizer_rule_override=config.get("training", {}).get("optimizer_rule"),
            kl_reference_override=config.get("training", {}).get("kl_reference"))
    if stage == "gvo" and trainer.config.kl_reference == "frozen_sft_actor" and trainer.actor_reference is None:
        if not resume:
            parser.error("Frozen SFT actor reference requires an actual warmup checkpoint")
        source_evidence = Path(resume) / "training_evidence.json"
        recorded = json.loads(source_evidence.read_text()) if source_evidence.is_file() else {}
        if not recorded.get("history") or recorded["history"][-1].get("stage") != "warmup":
            parser.error("Missing SFT reference: only the actual end-of-warmup actor may initialize it")
        trainer.bind_actor_reference(provenance={"kind": "sft_checkpoint", "checkpoint": str(Path(resume).resolve()),
            "training_evidence_sha256": hashlib.sha256(source_evidence.read_bytes()).hexdigest(),
            "optimizer_steps": trainer.steps})
    initial_optimizer_steps = trainer.steps
    output.mkdir(parents=True, exist_ok=True)
    config["output"] = str(output)
    config["stage"] = stage
    (output / "resolved_config.json").write_text(json.dumps({**config, "training": asdict(trainer.config), "stage": stage, "data": data_path,
        "epochs": epochs, "output": str(output), "synthetic_demo": stage == "demo", "algorithm_profile": paper}, indent=2))
    history = []
    started = time.monotonic()
    if stage == "demo":
        from specrsi.learning.demo import collect_demo_groups, demo_audits
        audits = demo_audits(model.config.feature_dim)
        train, validation, audit = split_by_lineage(audits, seed=trainer.config.seed)
        for epoch in range(epochs):
            metrics = trainer.warmup_epoch(audits=train)
            history.append({"epoch": epoch, **metrics})
        for epoch in range(epochs):
            groups = collect_demo_groups(model, groups=config.get("groups", 4), group_size=config.get("group_size", 4),
                                         seed=trainer.config.seed, round_index=epoch)
            save_jsonl(output / f"demo_trajectories_round_{epoch}.jsonl", [t for g in groups for t in g])
            metrics = trainer.gvo_epoch(groups, audits=train)
            history.append({"epoch": epoch, **metrics})
        evaluation = trainer.evaluate(audits=validation)
    else:
        records = load_jsonl(data_path) if data_path else []
        from specrsi.learning.splits import LineageSplitRegistry
        registry = LineageSplitRegistry(config.get("lineage_registry", output / "lineage_splits.json"),
            validation_fraction=config.get("validation_fraction", .2), seed=trainer.config.seed)
        train, validation, untouched = registry.split_records(records)
        data, heldout = parse_records(train), parse_records(validation)
        (output / "split_manifest.json").write_text(json.dumps({"train_lineages": sorted({x["lineage_id"] for x in train}),
            "validation_lineages": sorted({x["lineage_id"] for x in validation}),
            "untouched_audit_lineages": sorted({x["lineage_id"] for x in untouched})}, indent=2))
        for epoch in range(epochs):
            if stage == "warmup":
                if collector_path:
                    module, function = collector_path.split(":", 1)
                    collector = getattr(importlib.import_module(module), function)
                    model.eval()
                    try:
                        with torch.no_grad():
                            collection = collector(model, epoch, config)
                        from specrsi.learning.demonstrations import warmup_records
                        observed = warmup_records(collection)
                    except Exception as exc:
                        (output / f"collection_failure_round_{epoch}.json").write_text(json.dumps({
                            "valid_training_batch": False, "stage": "warmup_collection", "round": epoch,
                            "error_type": type(exc).__name__, "error": str(exc),
                            "elapsed_seconds": time.monotonic()-started}, indent=2))
                        raise
                    save_jsonl(output / f"demonstrations_round_{epoch}.jsonl", observed)
                    (output / f"collection_metadata_round_{epoch}.json").write_text(json.dumps(collection.get("metadata", {}), indent=2))
                    collected_train, collected_validation, collected_audit = registry.split_records(observed)
                    (output / f"split_manifest_round_{epoch}.json").write_text(json.dumps({
                        "train_lineages": sorted({x["lineage_id"] for x in collected_train}),
                        "validation_lineages": sorted({x["lineage_id"] for x in collected_validation}),
                        "untouched_audit_lineages": sorted({x["lineage_id"] for x in collected_audit})}, indent=2))
                    epoch_data = parse_records(collected_train)
                    for key in epoch_data:
                        epoch_data[key] += data[key]
                    metrics = trainer.warmup_epoch(**epoch_data)
                    heldout = parse_records(collected_validation)
                else:
                    metrics = trainer.warmup_epoch(**data)
            else:
                if collector_path:
                    module, function = collector_path.split(":", 1)
                    collector = getattr(importlib.import_module(module), function)
                    model.eval()
                    # Eq. (4), inner calls: collect every D_k^{(g)} under the
                    # SAME Theta_k before differentiating anything. no_grad()
                    # disables graph construction during physical acquisition;
                    # model.eval() disables stochastic neural layers so saved
                    # behavior probabilities can be replayed exactly. The
                    # collector freezes theta, psi, phi, and nu explicitly and
                    # supplies independent search states and random streams.
                    # A terminated search may leave many versions incomplete;
                    # its selected output must still satisfy Eq. (10), and its
                    # entire acquired/failed work must be logged for Eq. (13).
                    try:
                        with torch.no_grad():
                            collection = collector(model, epoch, config)
                    except Exception as exc:
                        (output / f"collection_failure_round_{epoch}.json").write_text(json.dumps({
                            "valid_training_batch": False, "stage": "collection", "round": epoch,
                            "error_type": type(exc).__name__, "error": str(exc),
                            "elapsed_seconds": time.monotonic()-started}, indent=2))
                        raise
                    if isinstance(collection, dict):
                        groups = collection["groups"]
                        collected_audits = collection.get("audits", [])
                        forced_trajectories = collection.get("forced_trajectories", [])
                        collected_pairs = collection.get("pairs", [])
                        (output / f"collection_metadata_round_{epoch}.json").write_text(json.dumps(collection.get("metadata", {}), indent=2))
                    else:
                        groups, collected_audits = collection, []
                        forced_trajectories, collected_pairs = [], []
                    # Eq. (4), D_k = union_g D_k^{(g)}: persist the full search
                    # trajectories, retaining group IDs through serialization.
                    # Flattening this output file does not turn individual
                    # decisions or tree leaves into independent group members.
                    # Forced suffixes are stored separately because Algorithm 3
                    # adds them to D_Q / D_pair, but never to D_actor.
                    save_jsonl(output / f"trajectories_round_{epoch}.jsonl", [t for g in groups for t in g])
                    save_jsonl(output / f"audits_round_{epoch}.jsonl", collected_audits)
                    save_jsonl(output / f"forced_trajectories_round_{epoch}.jsonl", forced_trajectories)
                    save_jsonl(output / f"pairs_round_{epoch}.jsonl", [
                        {"record_type": "pair", **asdict(p)} for p in collected_pairs])
                    filtered, validation_trajectories = registry.split_collection({"groups": groups,
                        "audits": collected_audits, "forced_trajectories": forced_trajectories, "pairs": collected_pairs})
                    groups, collected_audits = filtered["groups"], filtered["audits"]
                    forced_trajectories, collected_pairs = filtered["forced_trajectories"], filtered["pairs"]
                    heldout["trajectories"] = validation_trajectories
                else:
                    by_group = defaultdict(list)
                    forced_trajectories = []
                    for trajectory in data["trajectories"]:
                        if trajectory.forced:
                            forced_trajectories.append(trajectory)
                        else:
                            by_group[trajectory.group_id].append(trajectory)
                    groups = list(by_group.values())
                    collected_audits = []
                    collected_pairs = []
                # Eq. (4), outer assignment: Theta_{k+1}=GVO(Theta_k,D_k).
                # All collection is now complete and its recorded probabilities
                # and returns are constants. With pdf_gradient_v1, Eq. (23)
                # applies disjoint actor and Scorer gradients at this same
                # parameter origin. Only after this call can the next loop
                # iteration collect trajectories under the refreshed Theta.
                metrics = trainer.gvo_epoch(groups, audits=data["audits"] + collected_audits,
                    forced_trajectories=forced_trajectories, pairs=data["pairs"] + collected_pairs)
            history.append({"epoch": epoch, **metrics})
            print(json.dumps(history[-1]), flush=True)
        evaluation = trainer.evaluate(audits=heldout["audits"], trajectories=heldout["trajectories"])
    final_source_hashes = source_hashes()
    source_changed = initial_source_hashes != final_source_hashes
    if algorithm == "vats_pdf" and source_changed:
        (output / "training_failure.json").write_text(json.dumps({
            "valid_training_run": False, "source_changed_during_run": True,
            "initial_source_hashes": initial_source_hashes, "final_source_hashes": final_source_hashes}, indent=2))
        raise RuntimeError("Source changed during PDF role training; refusing to publish a frozen checkpoint")
    trainer.save(output / "checkpoint")
    summary = {"synthetic_demo": stage == "demo", "elapsed_seconds": time.monotonic() - started,
               "algorithm": algorithm, "algorithm_profile": paper, "source_hashes": initial_source_hashes,
               "source_changed_during_run": source_changed,
               "initial_optimizer_steps": initial_optimizer_steps,
               "optimizer_updates": trainer.steps - initial_optimizer_steps,
               "data_provenance": config.get("data_provenance", "synthetic_demo" if stage == "demo" else "user_supplied_records_or_collector"),
               "history": history, "validation": evaluation, "checkpoint": str(output / "checkpoint")}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
