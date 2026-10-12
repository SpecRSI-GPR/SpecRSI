"""Original DSPy MIPROv2, with the explicit no-demonstration restriction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from specrsi.baselines.gepa import MeteredLocalModel, baseline_source_identity
from specrsi.benchmarks.data import validate_disjoint_splits
from specrsi.experiments import score_answer


def make_dspy_lm(model: MeteredLocalModel, role: str, example_lookup: dict | None = None):
    import dspy

    class LocalDSPyLM(dspy.BaseLM):
        """DSPy's supported BaseLM callable protocol, without remote providers."""
        def __init__(self):
            super().__init__(model="local/"+getattr(model.client,'model','Qwen3.5-35B-A3B'), max_tokens=model.max_tokens,
                             temperature=model.temperature, cache=False)

        def __call__(self, prompt=None, *, messages=None, **kwargs):
            values = {**self.kwargs, **kwargs}
            n = values.get("n", 1)
            if not isinstance(n, int) or not 1 <= n <= 16:
                raise ValueError("Unsupported DSPy completion count")
            request = messages if messages is not None else prompt
            call_role, example_id = role, None
            if example_lookup:
                user_text = next((m["content"] for m in reversed(messages or []) if m["role"] == "user"), str(prompt or ""))
                matches = sorted((p for p in example_lookup if p in user_text), key=len, reverse=True)
                if matches:
                    call_role, example_id = example_lookup[matches[0]]
            return [model.call(request, role=call_role, example_id=example_id, temperature=values.get("temperature"),
                               max_tokens=values.get("max_tokens")) for _ in range(n)]

        async def acall(self, prompt=None, *, messages=None, **kwargs):
            # This adapter deliberately uses one worker: the local GPU service
            # serializes execution and all calls share one physical ledger.
            return self(prompt, messages=messages, **kwargs)

        def __deepcopy__(self, memo):
            return self.copy()

    return LocalDSPyLM()


def run_mipro(*, development, selection, audit, model: MeteredLocalModel,
              output_dir: str | Path, seed_prompt: str, num_candidates: int = 2,
              num_trials: int = 2, seed: int = 0, data_aware: bool = True) -> dict:
    import dspy
    if not development or not selection or not audit:
        raise ValueError("All three prospective splits must be nonempty")
    if model.records:
        raise ValueError("Use a fresh MeteredLocalModel for each experiment")
    validate_disjoint_splits({"development": development, "selection": selection, "audit": audit})
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.journal = output / "model_calls.jsonl"
    dataset = development[0].dataset
    if any(e.dataset != dataset for e in development + selection + audit):
        raise ValueError("MIPRO experiment must have one registered dataset/metric")
    def convert(examples):
        from .gepa import optimization_reference
        return [dspy.Example(question=e.prompt, answer={'_specrsi_example':e.to_dict()} if dataset=='planbench' else optimization_reference(e), example_id=e.id).with_inputs("question") for e in examples]
    def metric(gold, prediction, trace=None):
        if model.task_evaluator is not None:
            return prediction.native_score
        return score_answer(dataset, prediction.answer, gold.answer).score
    signature = dspy.Signature("question -> answer", seed_prompt)
    student = dspy.Predict(signature)
    if model.task_evaluator is not None:
        examples_by_prompt={e.prompt:e for e in development+selection+audit}
        class PipelineProgram(dspy.Module):
            def __init__(self):
                super().__init__(); self.instruction=dspy.Predict(signature)
            def forward(self,question):
                example=examples_by_prompt[question]
                answer,score=model.evaluate_example(self.instruction.signature.instructions,example)
                return dspy.Prediction(answer=answer,native_score=score.score,native_metrics=dict(score.metrics))
        student=PipelineProgram()
    lookup = {e.prompt: (e.split, e.id) for e in development + selection}
    task_lm, prompt_lm = make_dspy_lm(model, "optimization", lookup), make_dspy_lm(model, "reflection")
    manifest = {"schema_version": 1, "method": "MIPROv2-no-demos", "protocol": "prospective_local_adaptation",
                "author_exact_replication": False, "reuse_matched": False, "dataset": dataset,
                "max_bootstrapped_demos": 0, "max_labeled_demos": 0, "num_candidates": num_candidates,
                "native_internal_development_bootstrap": True,
                "num_trials": num_trials, "data_aware_proposer": data_aware, "program_aware_proposer": False,
                "fewshot_aware_proposer": False, "seed": seed,
                "max_generation_tokens": model.max_tokens,
                "splits": {k: [e.id for e in v] for k,v in {"development": development, "selection": selection, "audit": audit}.items()}}
    manifest.update(baseline_source_identity("dspy"))
    if getattr(model,'task_protocol',None):
        manifest['task_protocol']=model.task_protocol
        manifest['metric_variant']=model.task_protocol['metric_variant']
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    try:
        optimizer = dspy.MIPROv2(metric=metric, prompt_model=prompt_lm, task_model=task_lm,
                                 max_bootstrapped_demos=0, max_labeled_demos=0, auto=None,
                                 num_candidates=num_candidates, num_threads=1, max_errors=1,
                                 seed=seed, log_dir=str(output / "mipro"))
        with dspy.context(lm=task_lm):
            selected = optimizer.compile(student, trainset=convert(development), valset=convert(selection),
                                         num_trials=num_trials, minibatch=False, max_bootstrapped_demos=0,
                                         max_labeled_demos=0, seed=seed, program_aware_proposer=False,
                                         data_aware_proposer=data_aware, fewshot_aware_proposer=False,
                                         view_data_batch_size=min(2, len(development)))
        if any(p.demos for p in selected.predictors()):
            raise ValueError("MIPRO violated the registered no-demonstration condition")
        selected_state = selected.dump_state()
        selected_text = json.dumps(selected_state, sort_keys=True, default=str, ensure_ascii=False)
        (output / "selected_candidate.json").write_text(selected_text + "\n")
        audit_lm = make_dspy_lm(model, "audit", {e.prompt: (e.split, e.id) for e in audit})
        audits = []
        with dspy.context(lm=audit_lm):
            selected.set_lm(audit_lm)
            for example in audit:
                prediction = selected(question=example.prompt)
                if model.task_evaluator is not None:
                    from specrsi.benchmarks.base import ScoreResult
                    score=ScoreResult(prediction.native_score,prediction.native_metrics)
                else:
                    score = score_answer(dataset, prediction.answer, {'_specrsi_example':example.to_dict()} if dataset=='planbench' else example.reference)
                audits.append({"id": example.id, "answer": prediction.answer, "score": score.score, "metrics": dict(score.metrics)})
        (output / "audit.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audits))
        summary = {"status": "complete", "method": "MIPROv2-no-demos", "protocol": "prospective_local_adaptation",
                   "selected_candidate_sha256": hashlib.sha256(selected_text.encode()).hexdigest(),
                   "audit_score": sum(r["score"] for r in audits) / len(audits), "model_revision": model.model_revision,
                   "demonstrations": 0, "meter": model.summary(), "wall_seconds": time.monotonic() - started}
    except Exception as exc:
        summary = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "meter": model.summary(),
                   "wall_seconds": time.monotonic() - started}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
