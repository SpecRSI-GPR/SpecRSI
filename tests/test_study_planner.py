"""Planner controls: no model calls, paired recipes, and evidence-based resume."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("specrsi_study_launcher",ROOT/"scripts/run_studies.py")
planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)


@pytest.fixture
def cheap_sources(monkeypatch):
    monkeypatch.setattr(planner,"source_fingerprint",lambda root=ROOT:{"sha256":"frozen-test-source","files":{}})


def test_full_registry_preserves_reported_pair_counts_and_explicit_limits(cheap_sources):
    plan=planner.build_plan()
    registry=planner.read(ROOT/'configs/experiments/prospective_studies.json')
    expected=0
    for record in plan['study_records']:
        study=record['paper'];family='polyglot' if study.get('dataset')=='polyglot' else study['family']
        expected+=80 if family=='mechanisms' else study['reported_n']*len(study.get('datasets',[study.get('dataset')]))*len(registry['method_families'][family])
    assert plan["job_count"]==expected
    assert not plan["unavailable_requested_tracks"]
    assert {x["paper"]["id"] for x in plan["study_records"]}=={f"S{i:02d}" for i in range(1,14)}
    assert all(set(x["default_methods"])<=set(x["supported_methods"]) for x in plan["study_records"])
    reactor=next(x for x in plan["study_records"] if x["paper"]["id"]=="S04")
    assert reactor["split_contracts"][0]["scenario_sets"]==["development","selection","audit"]
    by_method={}
    for job in plan["jobs"]:
        by_method.setdefault((job["study"],job["task"],job["method"]),[]).append(job)
        assert not job["config"]["study_provenance"]["author_exact"]
        assert not job["config"]["study_provenance"]["original_seed_known"]
    assert all(len(jobs)==jobs[0]["reported_n"] for jobs in by_method.values())
    assert len(by_method[("S13","mechanisms","fixed-pool")])==50
    assert len(by_method[("S13","mechanisms","online")])==30
    assert any(x["id"]=="training-ablations" for x in plan["supplemental_protocols"])


def test_pairing_and_actual_workflow_models(cheap_sources):
    plan=planner.build_plan(studies=["S11","S12"],replicates=[0,1])
    paired={}
    for job in plan["jobs"]:
        paired.setdefault(job["pair_id"],set()).add(job["seed"])
        config=job["config"]
        workflow=config["search"]["initial_workflow"] if job["runner"]=="specrsi" else config["initial_workflow"]
        assert len(workflow["nodes"])>=2
        assert any(node["dependencies"] for node in workflow["nodes"])
        assert ("kimi" in config["model"]["lock"].lower())==(job["study"]=="S11")
    assert all(len(seeds)==1 for seeds in paired.values())


def test_science_training_dependency_is_shared_with_all_u(cheap_sources):
    jobs=planner.build_plan(studies=["S04"],methods=["specrsi","specrsi-all-u"],replicates=[1])["jobs"]
    assert len(jobs)==2 and jobs[0]["seed"]==jobs[1]["seed"]
    assert jobs[0]["config"]["role_checkpoint"]==jobs[1]["config"]["role_checkpoint"]
    assert all(j["gpu_required"] and j["config"]["controller"]=="learned" for j in jobs)
    artifact=jobs[0]["required_artifacts"][0]
    assert artifact["preparation_command"][-1]=="--execute"
    assert "configs/training/prospective/science_reactor.yaml" in artifact["preparation_command"]
    assert "replica-1" in artifact["path"]
    control=planner.build_plan(studies=["S04"],methods=["specrsi-heuristic"],replicates=[1])["jobs"][0]
    assert not control["gpu_required"] and not control["required_artifacts"]


def test_prompt_and_workflow_formal_specrsi_have_actual_training_recipes(cheap_sources):
    jobs=planner.build_plan(studies=['S07','S10','S11','S12'],methods=['specrsi'],replicates=[0])['jobs']
    assert len(jobs)==15
    for job in jobs:
        assert job['config']['search']['controller']=='learned'
        assert job['required_artifacts']
        assert (ROOT/job['config']['training_config']).is_file()


def test_secondary_native_prompt_methods_and_named_historical_boundaries(cheap_sources):
    plan=planner.build_plan(studies=['S10'],methods=['textgrad','trace-optoprime','mipro-demos'],replicates=[0])
    assert len(plan['jobs'])==18 and not plan['unavailable_requested_tracks']
    for job in plan['jobs']:
        assert job['runner']=='secondary-baseline'
        assert job['config']['method']==job['method']
        assert job['config']['data']['audit']>0
        assert job['command'][1].endswith('/scripts/run_secondary_baselines.py')
    missing=planner.build_plan(studies=['S10'],methods=['gpt-4.1-mini','gepa+r'],replicates=[0])
    assert not missing['jobs'] and len(missing['unavailable_requested_tracks'])==2
    assert all(row['reason'] for row in missing['unavailable_requested_tracks'])


def test_verified500_audit_track_cannot_overlap_selection(cheap_sources):
    job=planner.build_plan(studies=['S08'],methods=['specrsi-heuristic'],replicates=[0])['jobs'][0]
    assert job['config']['selection_track']=='verified500'
    assert job['config']['audit_track']=='lite_audit_disjoint_verified500'
    assert 'selection_ids' not in job['config'] and 'audit_ids' not in job['config']


def test_requested_unsupported_native_track_is_explicit_and_not_substituted(cheap_sources):
    plan=planner.build_plan(studies=["S06"],methods=["author-exact"],replicates=[0])
    assert not plan["jobs"]
    assert plan["unavailable_requested_tracks"][0]["status"]=="unavailable"
    with pytest.raises(ValueError,match="unavailable tracks"):
        planner.execute_plan(plan)
    with pytest.raises(ValueError,match="Unknown study"):
        planner.build_plan(studies=["S99"])
    with pytest.raises(ValueError,match="outside reported count"):
        planner.build_plan(studies=["S04"],replicates=[10])


def test_registered_split_sizes_clip_without_inventing_examples(cheap_sources):
    jobs=planner.build_plan(studies=["S10"],methods=["gepa"],replicates=[0])["jobs"]
    aime=next(j for j in jobs if j["task"]=="aime25")
    assert aime["config"]["sizes"]=={"development":13,"selection":11,"audit":6}
    assert "--dataset" in aime["command"] and "--model-lock" in aime["command"]
    assert aime["command"][aime["command"].index("--seed-prompt")+1]==aime["config"]["seed_prompt"]


def test_default_main_only_materializes_plan(monkeypatch,tmp_path,cheap_sources):
    def forbid(*args,**kwargs):raise AssertionError("Planning invoked execution")
    monkeypatch.setattr(planner.subprocess,"run",forbid)
    assert planner.main(["--study","S10","--method","gepa","--replicate","0",
                         "--plan-dir",str(tmp_path/"plans"),"--output-root",str(tmp_path/"results")])==0
    plan=json.loads((tmp_path/"plans/plan.json").read_text())
    assert len(plan["jobs"])==6
    assert not (tmp_path/"results").exists()
    assert all(Path(j["config_path"]).exists() for j in plan["jobs"])


def small_native_plan(tmp_path):
    return planner.build_plan(studies=["S04"],methods=["native"],replicates=[0],
        plan_root=str(tmp_path/"plan"),output_root=str(tmp_path/"results"))


def create_native_outputs(job):
    folder=Path(job["output"]);folder.mkdir(parents=True)
    rows=[]
    for role in job["config"]["scenario_sets"]:
        for case in job["config"]["case_ids"]:
            filename=f"{role}_{case}.json"
            (folder/filename).write_text(json.dumps({"complete":True,"valid":True,"score":0.5}))
            rows.append({"role":role,"case_id":case,"file":filename,"complete":True,"valid":True})
    (folder/"summary.json").write_text(json.dumps({"all_complete":True,"results":rows}))


def test_launcher_new_output_directory_resume_and_hash_tamper(monkeypatch,tmp_path,cheap_sources):
    plan=small_native_plan(tmp_path);job=plan["jobs"][0];calls=[]
    def fake_run(command,**kwargs):
        assert not Path(job["output"]).exists()  # GEPA/MIPRO's fresh-output constraint.
        assert kwargs.get("shell",False) is False
        create_native_outputs(job);calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(planner.subprocess,"run",fake_run)
    assert planner.execute_plan(plan)[0]["status"]=="completed"
    assert planner.valid_completion(job)
    assert planner.execute_plan(plan)[0]["status"]=="resumed_complete"
    assert len(calls)==1
    (Path(job["output"])/"selection_0.json").write_text("tampered")
    assert not planner.valid_completion(job)
    assert planner.execute_plan(plan)[0]["status"]=="completed"
    assert len(calls)==2
    assert list(Path(job["output"]).parent.glob("*.incomplete-*"))
    altered=copy.deepcopy(job);altered["config"]["seed"]+=1
    assert not planner.valid_completion(altered)


@pytest.mark.parametrize("failure",["exit","missing_output","config_tamper"])
def test_launcher_never_marks_failure_complete(monkeypatch,tmp_path,cheap_sources,failure):
    plan=small_native_plan(tmp_path);job=plan["jobs"][0]
    def fake_run(command,**kwargs):
        if failure!="missing_output":create_native_outputs(job)
        if failure=="config_tamper":Path(job["config_path"]).write_text('{}')
        return SimpleNamespace(returncode=1 if failure=="exit" else 0)
    monkeypatch.setattr(planner.subprocess,"run",fake_run)
    assert planner.execute_plan(plan)[0]["status"]=="failed"
    assert not (Path(job["output"])/"completed.json").exists()
    assert not planner.valid_completion(job)


def test_fingerprint_tracks_source_and_locked_configuration(tmp_path):
    (tmp_path/"src/specrsi").mkdir(parents=True);(tmp_path/"configs").mkdir()
    source=tmp_path/"src/specrsi/example.py";source.write_text("x=1\n")
    lock=tmp_path/"configs/model.lock.json";lock.write_text('{"revision":"a"}')
    first=planner.source_fingerprint(tmp_path)
    lock.write_text('{"revision":"b"}')
    second=planner.source_fingerprint(tmp_path)
    source.write_text("x=2\n")
    assert len({first["sha256"],second["sha256"],planner.source_fingerprint(tmp_path)["sha256"]})==3


def test_baseline_completion_requires_matching_frozen_candidate_and_audit_ids(tmp_path):
    selected='{"instruction": "fixture"}'
    files={"summary.json":{"status":"complete","selected_candidate_sha256":hashlib.sha256(selected.encode()).hexdigest()},
           "manifest.json":{"splits":{"audit":["test-1"]}}}
    for name,value in files.items():(tmp_path/name).write_text(json.dumps(value))
    (tmp_path/"selected_candidate.json").write_text(selected+'\n')
    (tmp_path/"model_calls.jsonl").write_text('{"test_fixture":true}\n')
    (tmp_path/"audit.jsonl").write_text('{"id":"test-1","score":0.0}\n')
    job={"runner":"prompt-baseline","output":str(tmp_path),"config":{"sizes":{"audit":1}}}
    assert "model_calls.jsonl" in planner.completion_outputs(job)
    (tmp_path/"audit.jsonl").write_text('{"id":"wrong-task","score":0.0}\n')
    with pytest.raises(ValueError,match="example IDs"):planner.completion_outputs(job)


def test_actual_training_prerequisite_graph_is_deduplicated_and_rebound(monkeypatch,tmp_path,cheap_sources):
    state={'ready':False};calls=[]
    def artifacts(config,kind,root=ROOT):
        if kind!='science-specrsi' or config.get('controller')!='learned':return []
        return [{'kind':'test_training_fixture','path':str(tmp_path/'roles/gvo/replica-0/learned/checkpoint'),
            'status':'present' if state['ready'] else 'requires_preparation',
            'sha256':{'weights':'actual-fixture-hash'} if state['ready'] else {},
            'preparation_command':['training-test-fixture']}]
    monkeypatch.setattr(planner,'required_artifacts',artifacts)
    def fake_run(command,**kwargs):
        calls.append(command)
        if command==('training-test-fixture',):
            state['ready']=True
            return SimpleNamespace(returncode=0)
        config=planner.read(command[command.index('--worker-config')+1])
        folder=Path(config['output']);folder.mkdir(parents=True)
        selected={'test_fixture':'frozen-parameter-candidate'}
        search={'acceptance':{'selected_id':'fixture','complete_candidates':['fixture'],'selected':selected},
                'source_changed_during_run':False}
        audit=[{'status':'complete','score':0.0} for _ in config['case_ids']]
        for name,value in [('search.json',search),('selected.json',selected),('audit.json',audit)]:
            (folder/name).write_text(json.dumps(value))
        (folder/'physical_calls.jsonl').write_text('{"test_fixture":true}\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(planner.subprocess,'run',fake_run)
    plan=planner.build_plan(studies=['S04'],methods=['specrsi','specrsi-all-u'],replicates=[0],
        plan_root=str(tmp_path/'plan'),output_root=str(tmp_path/'results'))
    old_fingerprints={j['fingerprint'] for j in plan['jobs']}
    results=planner.execute_plan(plan)
    refreshed=json.loads((tmp_path/'plan/plan.json').read_text())
    assert [r['status'] for r in results]==['completed','completed']
    assert calls.count(('training-test-fixture',))==1 and len(calls)==3
    assert not old_fingerprints & {j['fingerprint'] for j in refreshed['jobs']}
    assert all(planner.valid_completion(j) for j in refreshed['jobs'])
    assert all(r['status']=='resumed_complete' for r in planner.execute_plan(refreshed))
    assert len(calls)==3


def test_native_coding_optimizer_commands_preserve_whole_registered_tracks(cheap_sources):
    jobs=planner.build_plan(studies=['S08'],methods=['hgm','gepa','mipro-no-demos'],replicates=[0])['jobs']
    assert len(jobs)==3
    for job in jobs:
        c=job['config'];sizes=c['data'] if job['runner']=='extended' else c['sizes']
        assert sizes['development']==23 and sizes['selection']==500 and sizes['audit']==207
        if job['runner']=='extended':
            assert c['backend_factory']=='specrsi.baselines.native_backend:make_backend'
            assert c['audit_track']=='lite_audit_disjoint_verified500'
        else:
            assert '--native-execution-cap' in job['command']
            assert job['command'][job['command'].index('--audit-track')+1]=='lite_audit_disjoint_verified500'


def test_inference_masks_and_retrained_masks_are_separate_recipes(cheap_sources):
    jobs=planner.build_plan(studies=['S12'],methods=['specrsi','specrsi-outer-blind','specrsi-trained-outer-blind','specrsi-no-exploration'],replicates=[0])['jobs']
    jobs={j['method']:j for j in jobs if j['task']=='math'}
    base=jobs['specrsi']['config'];masked=jobs['specrsi-outer-blind']['config'];trained=jobs['specrsi-trained-outer-blind']['config']
    assert base['search']['role_checkpoint']==masked['search']['role_checkpoint']
    assert trained['search']['role_checkpoint']!=base['search']['role_checkpoint']
    assert masked['search']['observation_mode']==trained['search']['observation_mode']=='outer_blind'
    command=jobs['specrsi-trained-outer-blind']['required_artifacts'][0]['preparation_command']
    assert command[command.index('--methods')+1]=='outer_blind'
    assert base['search']['draft_exploration']==.1
    assert jobs['specrsi-no-exploration']['config']['search']['draft_exploration']==0.


def test_standalone_s09_executes_s06_dependency_and_never_calls_audit_model(monkeypatch,tmp_path,cheap_sources):
    plan=planner.build_plan(studies=['S09'],methods=['hgm'],replicates=[0],
        plan_root=str(tmp_path/'plan'),output_root=str(tmp_path/'results'))
    calls=[]
    def fake_run(command,**kwargs):
        calls.append(command)
        if '--worker' in command:
            c=planner.read(command[command.index('--worker-config')+1])
            planner.run_worker('audit-artifact',c)
        else:
            c=planner.read(command[command.index('--config')+1]);folder=Path(c['output']);folder.mkdir(parents=True)
            selected={'id':'frozen-test-policy','payload':{'system_prompt':'fixture'}}
            audit={'items':[{'item_id':str(i),'complete':True,'score':0.0} for i in range(c['data']['audit'])]}
            for name,value in [('result.json',{'selected':selected,'audit':audit}),('selected.json',selected),
                ('physical_calls.json',[]),('resolved_config.json',c),('native_receipts.json',[])]:
                (folder/name).write_text(json.dumps(value))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(planner.subprocess,'run',fake_run)
    assert planner.execute_plan(plan)[0]['status']=='completed'
    assert len(calls)==2 and 'run_extended_baselines.py' in calls[0][1]
    assert planner.valid_completion(plan['jobs'][0])
    report=planner.read(Path(plan['jobs'][0]['output'])/'audit_summary.json')
    assert report['new_model_calls']==0 and report['shared_audit_receipt_not_independent_run']
    assert planner.execute_plan(plan)[0]['status']=='resumed_complete' and len(calls)==2
    source=Path(report['source_output']);(source/'result.json').write_text('{}')
    assert not planner.valid_completion(plan['jobs'][0])
