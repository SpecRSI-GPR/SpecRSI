"""Original DSPy MIPROv2 with demonstrations that enter the real native agent."""
from __future__ import annotations

from fractions import Fraction
import json
import time

from .gepa import baseline_source_identity
from .mipro import make_dspy_lm
from .secondary_common import finish_run, freeze_selection, prepare_run, record_failure


def labeled_answer(example):
    """Development-only public answer targets; sealed coding fixtures stay private."""
    if example.split != 'development':
        raise ValueError('Only development references can become demonstrations')
    reference, dataset = example.reference, example.dataset
    if dataset in {'swe_verified', 'swe_lite', 'polyglot', 'hover', 'ifbench'}:
        return None  # Use native-score-validated bootstrap outputs for these tasks.
    if dataset == 'aime25':
        return str(reference)
    if dataset == 'game24':
        def solve(values):
            if len(values) == 1:
                return values[0][1] if values[0][0] == 24 else None
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    a, sa = values[i]; b, sb = values[j]
                    rest = [v for k, v in enumerate(values) if k not in {i, j}]
                    choices = [(a+b, f'({sa}+{sb})'), (a-b, f'({sa}-{sb})'),
                               (b-a, f'({sb}-{sa})'), (a*b, f'({sa}*{sb})')]
                    if b: choices.append((a/b, f'({sa}/{sb})'))
                    if a: choices.append((b/a, f'({sb}/{sa})'))
                    for choice in choices:
                        answer = solve(rest + [choice])
                        if answer is not None:
                            return answer
            return None
        return solve([(Fraction(n), str(n)) for n in reference])
    keys = {'hotpotqa': 'answer', 'math': 'solution', 'mbpp': 'code',
            'planbench': 'ground_truth_plan', 'pupa': 'target_response', 'livebench_math': 'ground_truth'}
    value = reference.get(keys.get(dataset, 'answer')) if isinstance(reference, dict) else None
    return str(value) if value is not None else None


def render_demonstrations(instruction, demonstrations, development_prompts):
    rendered = []
    for demo in demonstrations:
        question, answer = demo.get('question'), demo.get('answer')
        if question not in development_prompts or not isinstance(answer, str) or not answer.strip():
            raise ValueError('MIPRO demonstration must contain a development question and actual answer text')
        rendered.append({'question': question, 'answer': answer})
    if not rendered:
        return instruction
    return instruction + '\n\nDevelopment demonstrations (examples only; solve the current new task):\n' + json.dumps(rendered, ensure_ascii=False)


def run_mipro_demos(config, splits, model):
    import dspy
    output, manifest = prepare_run(config, splits, model)
    started = time.monotonic()
    settings = config.get('algorithm', {})
    boot = int(settings.get('max_bootstrapped_demos', 4))
    labeled = int(settings.get('max_labeled_demos', 4))
    if boot < 0 or labeled < 0 or boot + labeled < 1:
        raise ValueError('MIPRO-with-demos must permit at least one demonstration')
    targets = {e.id: labeled_answer(e) for e in splits['development']}
    if any(value is None for value in targets.values()):
        labeled = 0
        if boot == 0:
            raise ValueError('This task has no public labeled answer; enable native-verified bootstrapped demonstrations')
    examples = {e.prompt: e for rows in splits.values() for e in rows}
    development_prompts = {e.prompt for e in splits['development']}
    access = {'audit_open': False}
    execution_records = []
    signature = dspy.Signature('question -> answer', config['seed_prompt'])

    class NativePredictor(dspy.Predict):
        def forward(self, question, **kwargs):
            example = examples[question]
            if example.split == 'audit' and not access['audit_open']:
                raise ValueError('Audit cannot enter MIPRO compilation or demonstration bootstrapping')
            instruction = render_demonstrations(self.signature.instructions, self.demos, development_prompts)
            answer, score = model.evaluate_example(instruction, example)
            prediction = dspy.Prediction(answer=answer, native_score=score.score, native_metrics=dict(score.metrics))
            if dspy.settings.trace is not None:
                dspy.settings.trace.append((self, {'question': question}, {'answer': answer}))
            execution_records.append({'example_id': example.id, 'split': example.split,
                'demonstration_count': len(self.demos), 'score': score.score,
                'demonstration_questions': [demo['question'] for demo in self.demos]})
            return prediction

    class NativeProgram(dspy.Module):
        def __init__(self):
            super().__init__(); self.predictor = NativePredictor(signature)
        def forward(self, question):
            return self.predictor(question=question)

    def convert(rows):
        # Non-development answers never enter demonstration candidates; native
        # scoring uses the separately sealed host example, not this label field.
        return [dspy.Example(question=e.prompt, answer=targets.get(e.id) or '', example_id=e.id).with_inputs('question') for e in rows]

    def metric(gold, prediction, trace=None):
        return prediction.native_score

    task_lm = make_dspy_lm(model, 'optimization')
    proposer = make_dspy_lm(model, 'mipro_instruction_proposal')
    manifest.update(baseline_source_identity('dspy'))
    manifest.update(max_bootstrapped_demos=boot, max_labeled_demos=labeled,
        labeled_policy='Public development answers; unavailable/sealed answer tasks use only native-verified bootstraps',
        native_execution='Every selected demo is rendered into the actual native task instruction',
        program_aware_proposer=False, fewshot_aware_proposer=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        optimizer = dspy.MIPROv2(metric=metric, prompt_model=proposer, task_model=task_lm,
            max_bootstrapped_demos=boot, max_labeled_demos=labeled,
            metric_threshold=float(settings.get('bootstrap_score_threshold', 1.)),
            auto=None, num_candidates=int(settings.get('num_candidates', 6)), num_threads=1,
            max_errors=1, seed=config.get('seed', 0), log_dir=str(output / 'mipro'))
        with dspy.context(lm=task_lm):
            selected = optimizer.compile(NativeProgram(), trainset=convert(splits['development']),
                valset=convert(splits['selection']), num_trials=int(settings.get('num_trials', 20)),
                minibatch=False, max_bootstrapped_demos=boot, max_labeled_demos=labeled,
                seed=config.get('seed', 0), program_aware_proposer=False,
                data_aware_proposer=bool(settings.get('data_aware_proposer', True)),
                fewshot_aware_proposer=True, view_data_batch_size=min(2, len(splits['development'])))
        artifact = selected.dump_state()
        freeze_selection(output, artifact, model)
        access['audit_open'] = True
        audits = []
        with dspy.context(lm=task_lm):
            for example in splits['audit']:
                prediction = selected(question=example.prompt)
                audits.append({'id': example.id, 'answer': prediction.answer, 'score': prediction.native_score,
                               'metrics': prediction.native_metrics})
        (output / 'native_program_executions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in execution_records))
        return finish_run(config, output, manifest, model, artifact, audits, started=started,
            demonstrations=sum(len(p.demos) for p in selected.predictors()),
            executions_with_demonstrations=sum(row['demonstration_count'] > 0 for row in execution_records),
            optimizer='original_DSPy_MIPROv2_instruction_and_demo_search')
    except Exception as error:
        (output / 'native_program_executions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in execution_records))
        record_failure(output, model, error)
        raise
