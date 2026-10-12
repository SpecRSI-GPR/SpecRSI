from specrsi.benchmarks.swe_native import public_problem,registered_tracks
from specrsi.coding_agent import parse_action,safe_path
from pathlib import Path
import pytest


def test_actor_input_excludes_evaluation_oracles():
    row={k:'hidden' for k in ('patch','test_patch','FAIL_TO_PASS','PASS_TO_PASS','hints_text')}
    row.update(instance_id='x',repo='a/b',base_commit='abc',problem_statement='fix this')
    assert set(public_problem(row))=={'instance_id','repo','base_commit','problem_statement'}


def test_actions_and_file_boundary():
    assert parse_action('```json\n{"action":"finish"}\n```')=={'action':'finish'}
    assert safe_path('requests/sessions.py')=='requests/sessions.py'
    for path in ('../foo','/etc/passwd','.git/config','foo/../../bar'):
        with pytest.raises(ValueError):safe_path(path)
    with pytest.raises(ValueError):parse_action('{"action":"invented"}')


def test_registered_tracks_are_frozen_and_audit_disjoint():
    if not all(Path(path).is_file() for path in (
        'data/raw/swe_verified/test-00000-of-00001.parquet',
        'data/raw/swe_lite/test-00000-of-00001.parquet',
        'data/raw/swe_lite/dev-00000-of-00001.parquet',
    )):
        pytest.skip('Download pinned SWE-bench inputs with scripts/prepare_data.py')
    tracks=registered_tracks()
    assert len(tracks['selection60'])==60
    assert len(tracks['verified500'])==500
    assert not set(tracks['selection60'])&set(tracks['lite_audit_filtered'])
    assert not set(tracks['calibration_exposed'])&set(tracks['selection60'])
    assert not set(tracks['calibration_exposed'])&set(tracks['lite_audit_filtered'])
    assert len(tracks['lite_audit_disjoint_verified500'])==207
    assert not set(tracks['verified500'])&set(tracks['lite_audit_disjoint_verified500'])
    assert tracks==registered_tracks()


def test_native_image_accepts_only_exact_or_single_empty_publisher_setup_commit():
    from types import SimpleNamespace
    from specrsi.benchmarks.swe_native import verified_image_base
    class Container:
        def __init__(self,head='setup',parent='base',tree='tree',subject='SWE-bench'):
            self.head,self.parent,self.tree,self.subject=head,parent,tree,subject
        def exec_run(self,argv):
            if argv[-1]=='HEAD':value=self.head
            elif argv[-1]=='--format=%s':value=self.subject
            else:value='\n'.join([self.parent,self.tree,'tree'])
            return SimpleNamespace(exit_code=0,output=value.encode())
    assert not verified_image_base(Container(head='base'),'base')['empty_setup_commit']
    assert verified_image_base(Container(),'base')['empty_setup_commit']
    for container in (Container(parent='future'),Container(tree='changed'),Container(subject='unknown')):
        with pytest.raises(ValueError,match='mismatch'):verified_image_base(container,'base')
