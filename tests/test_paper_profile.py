import json
from dataclasses import asdict
from pathlib import Path

import pytest

from specrsi.paper_profile import algorithm_name, paper_provenance, require_role_checkpoint, vats_config
from specrsi.core.search import SearchConfig


@pytest.mark.parametrize('settings', [
    {'algorithm': 'invented'},
    {'algorithm': 'vats_pdf', 'eta': .005},
    {'algorithm': 'vats_pdf', 'all_u': False},
    {'algorithm': 'vats_pdf', 'controller': 'completion_cost_rule'},
    {'algorithm': 'vats_pdf', 'batch_mode': 'additive'},
])
def test_invalid_pdf_profile_fails_before_external_work(settings):
    with pytest.raises(ValueError):
        algorithm_name(settings)


def test_pdf_identity_is_checked_and_old_profiles_are_explicit(tmp_path):
    assert algorithm_name({}) == 'legacy_qv'
    assert algorithm_name({'algorithm': 'vats_pdf'}) == 'vats_pdf'
    paper = tmp_path / 'paper.pdf'
    paper.write_bytes(b'a different paper')
    with pytest.raises(ValueError, match='SHA-256'):
        paper_provenance({'paper_path': str(paper), 'paper_sha256': 'wrong'})


def test_current_paper_registration_cannot_silently_run_a_missing_or_old_pdf(tmp_path):
    import hashlib
    paper = tmp_path / 'paper.pdf'
    config = {'paper_path': str(paper), 'require_current_paper': True}
    with pytest.raises(ValueError, match='exact SpecRSI'):
        paper_provenance(config)
    paper.write_bytes(b'an older manuscript with its own correct registration')
    config['paper_sha256'] = hashlib.sha256(paper.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='exact SpecRSI'):
        paper_provenance(config)
    # Backward-compatible explicit replay records that it is not the current
    # paper; matching its own SHA does not make an arbitrary PDF version 33.
    config['require_current_paper'] = False
    receipt = paper_provenance(config)
    assert not receipt['matches_implementation_reference']
    assert receipt['manuscript_version'] == 'unrecognized'


def test_default_paper_identity_is_the_supplied_latest_manuscript():
    from specrsi.paper_profile import PDF_PATH, PDF_SHA256
    root = Path(__file__).resolve().parents[1]
    if not (root / PDF_PATH).is_file():
        pytest.skip('Provide the registered manuscript PDF to run this file-integration check')
    receipt = paper_provenance({'paper_path': str(root / PDF_PATH), 'paper_sha256': PDF_SHA256,
                                'require_current_paper': True})
    assert receipt['matches_implementation_reference']
    assert receipt['manuscript_version'] == '33'


def test_missing_checkpoint_is_rejected_before_model_evaluation(tmp_path):
    with pytest.raises(ValueError, match='Missing trained'):
        require_role_checkpoint({'controller': 'learned', 'role_checkpoint': str(tmp_path)})
    (tmp_path / 'model_config.json').write_text(json.dumps({'backend': 'tiny'}))
    with pytest.raises(ValueError, match='HF role'):
        require_role_checkpoint({'controller': 'learned', 'role_checkpoint': str(tmp_path)})


def test_vats_profile_preserves_execution_but_removes_legacy_acceptance():
    base = SearchConfig(execution_cap=37, draft_cap=11, margin=.2, reserve_completion=True)
    config = vats_config(base, {'algorithm':'vats_pdf'}, learned_query_cap=16)
    assert config.execution_cap == 37 and config.draft_cap == 11
    assert config.all_u and config.max_sources == 0
    assert config.margin == 0 and not config.reserve_completion
    assert config.scorer_query_cap == 16 and config.policy_call_cost == 16


def test_scientific_pdf_gain_keeps_native_units_and_can_change_ranking():
    from specrsi.core import Candidate, EvaluationScope, Outcome, Task
    from specrsi.science.executor import make_science_ledger
    tasks = (Task('a'), Task('b'))
    scope = EvaluationScope('native-test', tasks, 'test-v1', 'native-v1',
        scored_fields=('native_score',), aggregation_id='science_native_weighted_mean_v1')
    ledger = make_science_ledger()
    root, child = Candidate('root'), Candidate('child')
    for task, root_native, child_native in zip(tasks, (0., 10.), (4., 4.)):
        # The normalized scores would prefer child, but the native means prefer root.
        import math
        for candidate, native in ((root, root_native), (child, child_native)):
            ledger.admit_direct(candidate, task, scope,
                Outcome(.5 + math.atan(native)/math.pi, {'native_score': native}), {'test': True})
    assert ledger.score(root, scope) == 5.
    assert ledger.score(child, scope) == 4.


def test_search_return_includes_all_paid_phase_time_but_not_host_bookkeeping():
    from types import SimpleNamespace
    from specrsi.paper_profile import realized_search_return
    result = SimpleNamespace(initial_f=1., final_f=3., total_cost=7,
        elapsed_seconds=1000., transitions=[SimpleNamespace(elapsed_seconds=x) for x in (2., 3., 5., 0.)])
    assert realized_search_return(result, {'cost_lambda': .1, 'time_mu': .05}) == pytest.approx(.8)


def test_search_return_uses_the_same_compensated_clock_as_result_reporting():
    from types import SimpleNamespace
    from specrsi.core.search import SearchResult
    from specrsi.paper_profile import realized_search_return
    # Small measured phases must not disappear merely because a much longer
    # phase was encountered first. This is a numeric clock test, not a runtime.
    result = SimpleNamespace(initial_f=0., final_f=0., total_cost=0,
        transitions=[SimpleNamespace(elapsed_seconds=x) for x in (1.e16, 1., 1.)])
    reported = SearchResult.objective_elapsed_seconds.fget(result)
    assert reported == 10000000000000002.
    assert realized_search_return(result, {'time_mu': 1.}) == -reported


@pytest.mark.parametrize('name,value', [('time_mu', -1), ('cost_lambda', float('nan')),
                                      ('time_mu', float('inf')), ('cost_lambda', True)])
def test_invalid_objective_coefficients_are_rejected(name, value):
    from specrsi.paper_profile import objective_coefficients
    with pytest.raises(ValueError, match='finite nonnegative'):
        objective_coefficients({name: value})


@pytest.fixture
def reference_backbone_config(tmp_path, monkeypatch):
    import hashlib
    import specrsi.paper_profile as profile
    path = tmp_path / 'Qwen3.5-35B-A3B'
    path.mkdir()
    # A small stand-in for trusted publisher metadata exercises local preflight
    # without model weights/network access. Its trusted blob is fixture-scoped.
    raw = json.dumps({'model_type': 'qwen3_5_moe', 'text_config': {'dtype': 'bfloat16'}}).encode()
    (path / 'config.json').write_bytes(raw)
    monkeypatch.setattr(profile, 'REFERENCE_BACKBONE_CONFIG_BLOB_SHA1',
                        hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest())
    return {'require_reference_backbone': True, 'model': {
        'backend': 'hf', 'model_path': str(path), 'revision': profile.REFERENCE_BACKBONE_REVISION,
        'dtype': 'bfloat16', 'architecture_version': 3, 'head_hidden_dim': 256,
        'lora_rank': 16, 'feature_dim': 8, 'option_dim': 12,
        'language_model_only': True, 'require_unquantized_backbone': True,
        'lora_target_profile': 'qwen35_hybrid_attention', 'experts_implementation': 'eager'}}


def test_explicit_reference_metadata_receipt_and_demo_compatibility(reference_backbone_config):
    from specrsi.paper_profile import validate_reference_backbone
    assert validate_reference_backbone({'model': {'backend': 'tiny'}}) is None
    receipt = validate_reference_backbone(reference_backbone_config)
    assert receipt['repo_id'] == 'Qwen/Qwen3.5-35B-A3B'
    assert receipt['relation_attention_heads'] == 4
    assert receipt['dtype'] == 'bfloat16'
    assert receipt['lora_target_profile'] == 'qwen35_hybrid_attention'
    assert receipt['experts_implementation'] == 'eager'
    assert len(receipt['publisher_config_sha256']) == 64


@pytest.mark.parametrize('field,value', [
    ('backend', 'tiny'), ('revision', 'main'), ('dtype', 'float16'),
    ('architecture_version', 2), ('head_hidden_dim', 128), ('lora_rank', 4),
    ('feature_dim', 7), ('option_dim', 8), ('language_model_only', False),
    ('require_unquantized_backbone', False),
    ('lora_target_profile', 'configured'), ('experts_implementation', 'grouped_mm'),
    ('model_path', 'models/Qwen3-8B'), ('model_path', 'models/Qwen3.5-35B-A3B-FP8'),
])
def test_reference_rejects_mismatched_backbone_or_role_contract(reference_backbone_config, field, value):
    from specrsi.paper_profile import validate_reference_backbone
    reference_backbone_config['model'][field] = value
    with pytest.raises(ValueError, match='Reference backbone'):
        validate_reference_backbone(reference_backbone_config)


@pytest.mark.parametrize('field', ['backend', 'revision', 'dtype', 'architecture_version',
    'head_hidden_dim', 'lora_rank', 'model_path', 'language_model_only', 'require_unquantized_backbone',
    'lora_target_profile', 'experts_implementation'])
def test_reference_requires_explicit_model_fields(reference_backbone_config, field):
    from specrsi.paper_profile import validate_reference_backbone
    del reference_backbone_config['model'][field]
    with pytest.raises(ValueError, match='Reference backbone'):
        validate_reference_backbone(reference_backbone_config)


@pytest.mark.parametrize('nested', [False, True])
def test_reference_rejects_quantization_even_with_requested_bfloat16(reference_backbone_config, nested):
    from specrsi.paper_profile import validate_reference_backbone
    path = Path(reference_backbone_config['model']['model_path']) / 'config.json'
    raw = json.loads(path.read_text())
    (raw['text_config'] if nested else raw)['quantization_config'] = {'quant_method': 'fp8'}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='FP8 dequantization'):
        validate_reference_backbone(reference_backbone_config)


def test_reference_rejects_changed_publisher_metadata(reference_backbone_config):
    from specrsi.paper_profile import validate_reference_backbone
    path = Path(reference_backbone_config['model']['model_path']) / 'config.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='Git blob SHA-1'):
        validate_reference_backbone(reference_backbone_config)
