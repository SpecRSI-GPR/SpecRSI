"""Pinned upstream GEPA optimization with metered model calls and native scores.

The task adapter exposes development feedback to reflection, uses selection
scores to freeze a candidate, and evaluates that candidate on the disjoint audit
split. Optimizer settings and native metric-call budgets are registered for each
run; reuse identities remain specific to the declared task adapter.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

from specrsi.benchmarks.base import BenchmarkExample
from specrsi.benchmarks.data import validate_disjoint_splits
from specrsi.experiments import score_answer
from specrsi.models import LocalModelClient


def optimization_reference(example: BenchmarkExample):
    """Native coding graders own sealed tests/gold; optimizers see public inputs."""
    if example.dataset in {'swe_verified','swe_lite'}:
        from specrsi.benchmarks.swe_native import public_problem
        return {'native_feedback_only':True,**public_problem(example.reference)}
    if example.dataset=='polyglot':
        return {'native_feedback_only':True,'solution_files':example.reference.get('files',{}).get('solution',[])}
    return example.reference


def baseline_source_identity(name: str) -> dict:
    root = Path(__file__).resolve().parents[3]
    lock = root / "configs/baselines" / f"{name}.lock.json"
    if not lock.is_file():
        raise ValueError(f"Missing pinned baseline source manifest: {lock}")
    record = json.loads(lock.read_text())
    return {"baseline_revision": record["commit"], "baseline_repository": record["repository"],
            "baseline_archive_sha256": record["archive_sha256"]}


class MeteredLocalModel:
    def __init__(self, client: LocalModelClient, *, max_tokens: int = 256, temperature: float = 0,
                 seed: int = 0, model_revision: str | None = None, journal: str | Path | None = None):
        self.client, self.max_tokens, self.temperature, self.seed = client, max_tokens, temperature, seed
        self.model_revision = model_revision
        self.records: list[dict] = []
        self.journal = Path(journal) if journal else None
        self.task_evaluator = None

    def evaluate_example(self, instruction, example):
        if self.task_evaluator is not None:
            return self.task_evaluator(instruction, example)
        answer = self.call([{"role":"system","content":instruction},{"role":"user","content":example.prompt}],
                           role=example.split,example_id=example.id)
        reference={'_specrsi_example':example.to_dict()} if example.dataset=='planbench' else example.reference
        return answer,score_answer(example.dataset,answer,reference)

    def call(self, messages, *, role: str, example_id: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        started = time.monotonic()
        record = {"role": role, "example_id": example_id, "call_index": len(self.records),
                  "input_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}
        generation_limit = self.max_tokens if max_tokens is None else min(self.max_tokens, max_tokens)
        generation_temperature = self.temperature if temperature is None else temperature
        record.update(max_generation_tokens=generation_limit, temperature=generation_temperature)
        try:
            reply = self.client.generate(messages, max_tokens=generation_limit, temperature=generation_temperature,
                                         seed=self.seed + len(self.records), enable_thinking=False)
            record.update(asdict(reply), status="complete")
            if self.model_revision is None:
                self.model_revision = reply.model_revision
            if reply.model_revision != self.model_revision:
                raise RuntimeError("Model revision changed during frozen baseline evaluation")
            if reply.finish_reason == "timeout":
                raise RuntimeError("Incomplete model generation: timeout")
            return reply.text
        except Exception as exc:
            # Do not erase paid work on scoring or response-processing failures.
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            record.setdefault("elapsed_seconds", time.monotonic() - started)
            raise
        finally:
            self.records.append(record)
            if self.journal:
                self.journal.parent.mkdir(parents=True, exist_ok=True)
                with self.journal.open("a") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __call__(self, prompt):
        return self.call(prompt, role="reflection")

    def summary(self):
        totals = {}
        for role in {r["role"] for r in self.records}:
            records = [r for r in self.records if r["role"] == role]
            totals[role] = {"calls": len(records), "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in records),
                            "completion_tokens": sum(r.get("completion_tokens", 0) for r in records),
                            "model_seconds": sum(r.get("elapsed_seconds", 0) for r in records),
                            "allocation_gpu_seconds": sum(r.get("elapsed_seconds", 0)*r.get("gpu_count",1) for r in records),
                            "unmetered_failed_calls": sum("prompt_tokens" not in r for r in records)}
        return totals


class LocalGEPAAdapter:
    propose_new_texts = None

    def __init__(self, model: MeteredLocalModel):
        self.model = model

    def evaluate(self, batch: list[BenchmarkExample], candidate: Mapping[str, str], capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch
        if set(candidate) != {"system_prompt"}:
            raise ValueError("Local GEPA adapter requires one system_prompt component")
        if any(e.split not in {"development", "selection"} for e in batch):
            raise ValueError("Audit data cannot enter GEPA's optimization adapter")
        outputs, scores, trajectories = [], [], []
        for example in batch:
            answer, score = self.model.evaluate_example(candidate['system_prompt'], example)
            outputs.append({"answer": answer, "metrics": dict(score.metrics), "example_id": example.id})
            scores.append(score.score)
            if capture_traces:
                trajectories.append({"input": example.prompt, "output": answer, "score": score.score,
                                     "reference": optimization_reference(example), "metrics": dict(score.metrics), "split": example.split})
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if components_to_update != ["system_prompt"] or eval_batch.trajectories is None:
            raise ValueError("Expected captured trajectories for system_prompt")
        if any(t["split"] != "development" for t in eval_batch.trajectories):
            raise ValueError("Only development traces may enter reflection")
        return {"system_prompt": [{"Inputs": t["input"], "Generated Outputs": t["output"],
                                    "Feedback": json.dumps({"score": t["score"], "reference": t["reference"],
                                                              "metrics": t["metrics"]}, ensure_ascii=False)}
                                   for t in eval_batch.trajectories]}


def run_gepa(*, development: list[BenchmarkExample], selection: list[BenchmarkExample], audit: list[BenchmarkExample],
             model: MeteredLocalModel, output_dir: str | Path, seed_prompt: str, max_metric_calls: int = 10,
             seed: int = 0, use_merge: bool = False) -> dict:
    import gepa
    if not development or not selection or not audit:
        raise ValueError("Development, selection, and independent audit must be nonempty")
    if model.records:
        raise ValueError("Use a fresh MeteredLocalModel for each experiment")
    validate_disjoint_splits({"development": development, "selection": selection, "audit": audit})
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve run identity: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.journal = output / "model_calls.jsonl"
    manifest = {"schema_version": 1, "method": "GEPA+Merge" if use_merge else "GEPA",
                "protocol": "prospective_local_adaptation", "author_exact_replication": False,
                "reuse_matched": False, "budget_endpoint": "native_metric_calls",
                "stopping_semantics": "official GEPA checks threshold between iterations; final batch may exceed threshold",
                "max_metric_calls": max_metric_calls, "seed": seed,
                "dataset": development[0].dataset, "model_revision": model.model_revision,
                "max_generation_tokens": model.max_tokens,
                "splits": {k: [e.id for e in v] for k, v in {"development": development, "selection": selection, "audit": audit}.items()},
                "metric_variant": "GEPA_three_hop_all_supporting_documents" if development[0].dataset == "hover" else "prospective_local_PAPILLON_judge" if development[0].dataset=='pupa' else "task_native"}
    manifest.update(baseline_source_identity("gepa"))
    if getattr(model,'task_protocol',None):
        manifest['task_protocol']=model.task_protocol
        manifest['metric_variant']=model.task_protocol['metric_variant']
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    adapter = LocalGEPAAdapter(model)
    started = time.monotonic()
    try:
        result = gepa.optimize(seed_candidate={"system_prompt": seed_prompt}, trainset=development, valset=selection,
                               adapter=adapter, reflection_lm=model, max_metric_calls=max_metric_calls,
                               reflection_minibatch_size=min(2, len(development)), skip_perfect_score=False,
                               use_merge=use_merge, seed=seed, raise_on_exception=True,
                               run_dir=str(output / "gepa"), display_progress_bar=False)
        candidate = result.best_candidate
        # Persist the complete selected version before accessing audit examples.
        candidate_text = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        (output / "selected_candidate.json").write_text(candidate_text + "\n")
        result_dict = result.to_dict()
        (output / "optimization.json").write_text(json.dumps(result_dict, indent=2, default=str) + "\n")
        audit_records = []
        for example in audit:
            answer, score = model.evaluate_example(candidate['system_prompt'], example)
            audit_records.append({"id": example.id, "answer": answer, "score": score.score, "metrics": dict(score.metrics)})
        (output / "audit.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_records))
        summary = {"status": "complete", "method": manifest["method"], "protocol": manifest["protocol"],
                   "selected_candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
                   "selection_score": result.val_aggregate_scores[result.best_idx],
                   "audit_score": sum(r["score"] for r in audit_records) / len(audit_records),
                   "candidate_count": len(result.candidates), "native_metric_calls": result.total_metric_calls,
                   "model_revision": model.model_revision, "meter": model.summary(),
                   "wall_seconds": time.monotonic() - started}
    except Exception as exc:
        summary = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "meter": model.summary(),
                   "wall_seconds": time.monotonic() - started}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
