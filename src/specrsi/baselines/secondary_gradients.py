"""Original TextGrad and Trace/OptoPrime on one editable native-task instruction."""
from __future__ import annotations

import json
import random
import time
from types import SimpleNamespace

from .gepa import optimization_reference
from .secondary_common import (evaluate_prompt, finish_run, freeze_selection, pinned_import,
                               prepare_run, record_failure)


def textgrad_optimizer(model, seed_prompt, settings):
    tg, source = pinned_import('textgrad')
    from textgrad.autograd.llm_ops import LLMCall
    from textgrad.autograd.function import BackwardContext

    class LocalEngine(tg.EngineLM):
        model_string = 'specrsi-pinned-local-model'
        def __init__(self, role, example=None):
            self.role, self.example, self.last_score = role, example, None
        def generate(self, prompt, system_prompt=None, **kwargs):
            if self.example is not None:
                if prompt != self.example.prompt:
                    raise ValueError('TextGrad task input changed outside the editable instruction')
                answer, self.last_score = model.evaluate_example(system_prompt, self.example)
                return answer
            messages = ([{'role': 'system', 'content': system_prompt}] if system_prompt else [])
            messages.append({'role': 'user', 'content': prompt})
            return model.call(messages, role=self.role)
        def __call__(self, *args, **kwargs):
            return self.generate(*args, **kwargs)

    parameter = tg.Variable(seed_prompt, role_description='system instruction of the fixed native task agent', requires_grad=True)
    optimizer = tg.TGD([parameter], engine=LocalEngine('textgrad_update'),
        constraints=['Only change the system instruction. Preserve task format and native grader boundaries.'],
        gradient_memory=int(settings.get('gradient_memory', 0)))
    backward_engine = LocalEngine('textgrad_backward')

    def update(examples):
        optimizer.zero_grad()
        records = []
        for example in examples:
            forward_engine = LocalEngine('development', example)
            answer = LLMCall(forward_engine, system_prompt=parameter)(
                tg.Variable(example.prompt, role_description='immutable public task input', requires_grad=False))
            score = forward_engine.last_score
            feedback = json.dumps({'native_score': score.score, 'metrics': dict(score.metrics),
                'development_reference': optimization_reference(example),
                'objective': 'Improve the general instruction to maximize the native score. Do not edit the grader.'}, ensure_ascii=False)
            # A native loss is not differentiable numerically. Seed its measured
            # feedback in TextGrad's graph, then use ORIGINAL LLMCall.backward
            # to obtain the textual gradient of the editable system instruction.
            loss = tg.Variable(feedback, role_description='measured native scoring feedback', predecessors=[answer])
            def propagate(response, feedback, backward_engine):
                gradient = tg.Variable(feedback, role_description='upstream native feedback', requires_grad=False)
                response.gradients.add(gradient)
                response.gradients_context[gradient] = None
            loss.set_grad_fn(BackwardContext(backward_fn=propagate, response=answer, feedback=feedback))
            loss.backward(engine=backward_engine)
            records.append({'example_id': example.id, 'answer': answer.value, 'score': score.score,
                            'native_feedback': feedback})
        gradients = [gradient.value for gradient in parameter.gradients]
        if not gradients:
            raise ValueError('TextGrad produced no textual gradient for the editable prompt')
        optimizer.step()
        return parameter.value, {'native_records': records, 'textual_gradients': gradients,
                                 'optimizer': 'original_textgrad_TGD'}
    return update, source


def trace_optimizer(model, seed_prompt, settings):
    trace, source = pinned_import('trace')
    from opto.optimizers import OptoPrime

    class LocalTraceEngine:
        def __call__(self, messages, **kwargs):
            # The upstream optimizer requests JSON through its own prompt. The
            # local serving API uses the same metered text protocol for all calls.
            answer = model.call(messages, role='optoprime_update', max_tokens=kwargs.get('max_tokens'))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])

    parameter = trace.node(seed_prompt, name='system_prompt', trainable=True,
        constraint='Only this strategy instruction may change. No task, grader or source code changes.')
    optimizer = OptoPrime([parameter], llm=LocalTraceEngine(), ignore_extraction_error=False,
        memory_size=int(settings.get('memory_size', 0)), max_tokens=model.max_tokens)
    current_examples = {}

    @trace.bundle(description='[native_execution] Execute the fixed local native task agent with the supplied instruction and public question; return its actual answer and native grade.',
                  allow_external_dependencies=True, catch_execution_error=False)
    def native_execution(instruction, question):
        example = current_examples[question]
        answer, score = model.evaluate_example(instruction, example)
        return {'answer': answer, 'score': score.score, 'metrics': dict(score.metrics),
                'development_reference': optimization_reference(example)}

    @trace.bundle(description='[native_quality] Extract the complete measured native task score. The native scorer is immutable.', catch_execution_error=False)
    def native_quality(result):
        return result['score']

    def update(examples):
        optimizer.zero_feedback()
        current_examples.clear()
        current_examples.update({e.prompt: e for e in examples})
        records = []
        for example in examples:
            execution = native_execution(parameter, trace.node(example.prompt, name='public_task'))
            quality = native_quality(execution)
            optimizer.backward(quality, 'Improve the general system instruction to maximize the native task score. '
                'Use the actual forward answer, measured score and development feedback in the graph; do not edit the native grader.')
            records.append({'example_id': example.id, **execution.data})
        optimizer.step()
        return parameter.data, {'native_records': records, 'optimizer': 'original_Trace_OptoPrime',
                                'original_optimizer_log': optimizer.log[-1] if optimizer.log else None}
    return update, source


def run_gradients(config, splits, model):
    output, manifest = prepare_run(config, splits, model)
    started = time.monotonic()
    settings = config.get('algorithm', {})
    steps, batch_size = int(settings.get('steps', 10)), int(settings.get('batch_size', 2))
    if steps < 1 or batch_size < 1:
        raise ValueError('Gradient baselines require positive update steps and development batch size')
    try:
        factory = textgrad_optimizer if config['method'] == 'textgrad' else trace_optimizer
        update, source = factory(model, config['seed_prompt'], settings)
        manifest.update(original_source=source, program_space='one system instruction; fixed native execution and grader',
            optimization_data='development feedback only; selection only chooses a completed saved prompt',
            adaptation='Native task pipeline is a single opaque forward operator; its complete real output and score enter the original optimizer.')
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        best = config['seed_prompt']
        best_score, initial_rows = evaluate_prompt(model, best, splits['selection'])
        history = [{'step': -1, 'system_prompt': best, 'selection_score': best_score, 'selection': initial_rows}]
        rng = random.Random(config.get('seed', 0))
        for step in range(steps):
            batch = rng.sample(list(splits['development']), min(batch_size, len(splits['development'])))
            candidate, trace = update(batch)
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError('Original optimizer produced an empty/non-text instruction')
            value, rows = evaluate_prompt(model, candidate, splits['selection'])
            if value > best_score:
                best, best_score = candidate, value
            history.append({'step': step, 'system_prompt': candidate, 'selection_score': value,
                            'selection': rows, 'development_update': trace})
            (output / 'optimization.json').write_text(json.dumps(history, indent=2, default=str) + '\n')
        selected = {'system_prompt': best}
        freeze_selection(output, selected, model)
        _, audit = evaluate_prompt(model, best, splits['audit'])
        return finish_run(config, output, manifest, model, selected, audit, started=started,
                          selection_score=best_score, optimizer_steps=steps, candidate_count=len(history))
    except Exception as error:
        record_failure(output, model, error)
        raise
