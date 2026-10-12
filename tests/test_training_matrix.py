"""Independent PDF replicas must actually run the registered two-stage method."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_matrix', ROOT / 'scripts/run_training_matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def write_config(tmp_path, **overrides):
    config = yaml.safe_load((ROOT / 'configs/training/paper_pdf_reference.yaml').read_text())
    config.update(overrides)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return path


def test_pdf_replicas_keep_architecture_and_seed_every_stage(tmp_path):
    path = write_config(tmp_path, warmup={'training': {'seed': 123}}, gvo={'training': {'seed': 456}})
    rows = matrix.plan(path, ['gvo'], [11, 12], tmp_path / 'matrix')
    from specrsi.learning.pipeline import stage_configs
    for row in rows:
        assert row['kind'] == 'two_stage'
        cfg = row['config']
        assert cfg['model']['option_dim'] == 12
        assert cfg['model']['seed'] == cfg['training']['seed'] == row['seed']
        stages = stage_configs(cfg, cfg['output'])
        assert all(stage['training']['seed'] == row['seed'] for stage in stages)
        assert all(stage['collection']['algorithm'] == 'vats_pdf' for stage in stages)
        assert stages[1]['resume'] == str(Path(stages[0]['output']) / 'checkpoint')
    assert rows[0]['config']['output'] != rows[1]['config']['output']


def test_pdf_defaults_and_pair_ablation_preserve_paid_rollouts(tmp_path):
    rows = matrix.plan(write_config(tmp_path), None, [1], tmp_path / 'matrix')
    assert [row['method'] for row in rows] == ['sft', 'gvo', 'no_pair']
    sft, gvo, no_pair = (row['config'] for row in rows)
    assert sft['stage'] == 'warmup' and sft['collection']['demonstration_policy'] == 'complete_then_draft'
    assert no_pair['collection']['forced_repetitions'] == gvo['collection']['forced_repetitions'] == 2
    assert no_pair['gvo']['training']['pair_weight'] == 0
    assert gvo['training']['pair_weight'] == 1
    for method in ('rloo', 'grpo', 'no_p', 'additive'):
        with pytest.raises(ValueError, match='supported'):
            matrix.plan(write_config(tmp_path), [method], [1], tmp_path)


@pytest.mark.parametrize('seeds', [[], [1, 1], [True], [-1]])
def test_replicas_cannot_duplicate_training_seeds(tmp_path, seeds):
    with pytest.raises(ValueError, match='distinct'):
        matrix.plan(write_config(tmp_path), ['gvo'], seeds, tmp_path)


def test_pdf_replicas_cannot_share_resumed_initialization(tmp_path):
    for overrides in ({'resume': 'existing/checkpoint'}, {'warmup': {'resume': 'existing/checkpoint'}}):
        with pytest.raises(ValueError, match='checkpoint'):
            matrix.plan(write_config(tmp_path, **overrides), ['gvo'], [1], tmp_path)


def test_launcher_leaves_fresh_output_to_trainer_and_verifies_both_stages(tmp_path, monkeypatch):
    row = matrix.plan(write_config(tmp_path), ['gvo'], [1], tmp_path / 'matrix')[0]
    folder = Path(row['config']['output'])
    calls = []

    def train(command, **kwargs):
        assert not folder.exists()
        calls.append(command)
        folder.mkdir()
        for stage in ('warmup', 'gvo'):
            directory = folder / stage
            directory.mkdir()
            (directory / 'metrics.json').write_text(json.dumps({'history': [{'steps': 1}],
                'initial_optimizer_steps': 0, 'optimizer_updates': 1}))
            (directory / 'checkpoint').mkdir()
            (directory / 'checkpoint' / 'weights').write_text('actual-test-fixture')
        (folder / 'checkpoint').symlink_to('gvo/checkpoint', target_is_directory=True)
        (folder / 'metrics.json').write_text(json.dumps({'complete': True}))

    monkeypatch.setattr(matrix.subprocess, 'run', train)
    matrix.execute(row)
    assert len(calls) == 1 and (folder / 'completed.json').is_file()
    matrix.execute(row)
    assert len(calls) == 1  # Resume only a hash-verified complete replica.
    (folder / 'checkpoint').unlink()
    (folder / 'checkpoint').symlink_to('warmup/checkpoint', target_is_directory=True)
    with pytest.raises(ValueError, match='changed'):
        matrix.execute(row)
    (folder / 'checkpoint').unlink()
    (folder / 'checkpoint').symlink_to('gvo/checkpoint', target_is_directory=True)
    (folder / 'warmup' / 'checkpoint' / 'weights').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        matrix.execute(row)


def test_launcher_does_not_certify_an_untrained_gvo_stage(tmp_path, monkeypatch):
    row = matrix.plan(write_config(tmp_path), ['gvo'], [1], tmp_path / 'matrix')[0]
    folder = Path(row['config']['output'])

    def train(*args, **kwargs):
        for stage in ('warmup', 'gvo'):
            directory = folder / stage
            directory.mkdir(parents=True)
            (directory / 'checkpoint').mkdir()
            (directory / 'metrics.json').write_text(json.dumps({'history': [{'steps': 3}],
                'initial_optimizer_steps': 0 if stage == 'warmup' else 3,
                'optimizer_updates': 3 if stage == 'warmup' else 0}))
        (folder / 'checkpoint').symlink_to('gvo/checkpoint', target_is_directory=True)
        (folder / 'metrics.json').write_text(json.dumps({'complete': True}))

    monkeypatch.setattr(matrix.subprocess, 'run', train)
    with pytest.raises(ValueError, match='optimizer updates'):
        matrix.execute(row)
    assert not (folder / 'completed.json').exists()


@pytest.mark.parametrize('missing', [None, 'signal', 'manifest'])
def test_v3_launcher_preserves_legitimate_zero_signal_without_actor_claim(tmp_path, monkeypatch, missing):
    from test_learning_update_evidence import model, decision, trajectory
    from specrsi.learning.trainer import SpecRSITrainer, TrainingConfig
    net = model()
    trainer = SpecRSITrainer(net, TrainingConfig(weight_decay=0.))
    item = decision(net)
    warm = trainer.warmup_epoch(plans=[item], trajectories=[trajectory([item], 1)])
    zero = trainer.gvo_epoch([[trajectory([], 1), trajectory([], 2)]])
    base = yaml.safe_load((ROOT / 'configs/training/paper_pdf_reference.yaml').read_text())
    base['model']['architecture_version'] = 3
    path = tmp_path / 'v3.yaml'
    path.write_text(yaml.safe_dump(base))
    row = matrix.plan(path, ['gvo'], [1], tmp_path / 'matrix')[0]
    folder = Path(row['config']['output'])

    def train(*args, **kwargs):
        for stage, record in [('warmup', warm), ('gvo', zero)]:
            directory = folder / stage
            checkpoint = directory / 'checkpoint'
            checkpoint.mkdir(parents=True)
            evidence = dict(record)
            if missing == 'signal' and stage == 'gvo':
                evidence.pop('learning_signal')
            if not (missing == 'manifest' and stage == 'gvo'):
                (checkpoint / 'training_evidence.json').write_text(json.dumps({'schema': 'role_training_evidence_v1'}))
            (directory / 'metrics.json').write_text(json.dumps({'history': [evidence],
                'initial_optimizer_steps': 0 if stage == 'warmup' else warm['steps'],
                'optimizer_updates': warm['steps'] if stage == 'warmup' else 0, 'checkpoint': str(checkpoint)}))
        (folder / 'checkpoint').symlink_to('gvo/checkpoint', target_is_directory=True)
        (folder / 'metrics.json').write_text(json.dumps({'complete': True}))

    monkeypatch.setattr(matrix.subprocess, 'run', train)
    if missing:
        with pytest.raises(ValueError, match='evidence'):
            matrix.execute(row)
        assert not (folder / 'completed.json').exists()
    else:
        matrix.execute(row)
        completed = json.loads((folder / 'completed.json').read_text())
        signal = completed['stage_learning_signals'][-1]['learning_signal']
        assert signal['status'] == 'no_parameter_change'
        assert signal['optimizer_steps'] == 0 and not signal['all_actor_roles_updated']
        matrix.execute(row)  # Hash-verified idempotent completion still works.
