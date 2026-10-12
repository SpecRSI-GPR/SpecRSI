#!/usr/bin/env python3
"""Plan/execute independent role replicas and explicitly registered training controls."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
LEGACY_METHODS=('sft','gvo','rloo','grpo','no_p','item_reward','outer_blind','inner_blind','evidence_blind','additive','blind_additive')
PDF_METHODS=('sft','gvo','no_pair')
METHODS=(*LEGACY_METHODS, 'no_pair')


def plan(config,methods,seeds,output):
    base=yaml.safe_load(Path(config).read_text())
    output=Path(output).resolve()
    pdf=base.get('collection',{}).get('algorithm')=='vats_pdf'
    allowed=PDF_METHODS if pdf else LEGACY_METHODS
    methods=list(allowed if methods is None else methods)
    seeds=list(seeds)
    if not methods or len(set(methods))!=len(methods) or set(methods)-set(allowed):
        raise ValueError(f'Choose distinct methods supported by this algorithm: {allowed}')
    if not seeds or any(type(seed) is not int or seed<0 for seed in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Independent replicas require distinct nonnegative integer seeds')
    if pdf:
        if base.get('stage')!='two_stage' or base.get('resume'):
            raise ValueError('PDF replicas require a fresh two_stage configuration, not a shared resumed checkpoint')
        from specrsi.learning.pipeline import stage_configs
        rows=[]
        for seed in seeds:
            for method in methods:
                folder=Path(output)/method/f'replica-{seed}'
                c=copy.deepcopy(base)
                c['output']=str(folder)
                c.setdefault('training',{})['seed']=seed
                if c.get('model',{}).get('backend','tiny')=='tiny':
                    c.setdefault('model',{})['seed']=seed
                for stage in ('warmup','gvo'):
                    overrides=c.setdefault(stage,{})
                    if overrides.get('resume'):
                        raise ValueError('Replica stage cannot resume a shared checkpoint')
                    overrides.setdefault('training',{})['seed']=seed
                    if overrides.get('model',c.get('model',{})).get('backend','tiny')=='tiny' and 'model' in overrides:
                        overrides['model']['seed']=seed
                if method=='no_pair':
                    # Preserve the same acquired forced arms and mean-Q targets;
                    # only remove Eq.23's pairwise loss, as in the paper ablation.
                    c['gvo']['training']['pair_weight']=0.
                stages=stage_configs(c,folder)
                if method=='sft':
                    c=stages[0]
                    c['output']=str(folder)
                    c['lineage_registry']=str(folder/'lineage_splits.json')
                rows.append({'kind':c['stage'],'method':method,'seed':seed,'config':c})
        return rows
    warm=yaml.safe_load((ROOT/'configs/training/qwen_native_warmup.yaml').read_text())
    rows=[]
    for seed in seeds:
        for method in methods:
            folder=Path(output)/method/f'replica-{seed}'
            w=copy.deepcopy(warm);w['training']['seed']=seed;w['output']=str(folder/'warmup')
            if method=='no_p':w['training']['surface_weight']=0
            rows.append({'kind':'warmup','method':method,'seed':seed,'config':w})
            if method=='sft':continue
            c=copy.deepcopy(base);c['training']['seed']=seed;c['resume']=w['output']+'/checkpoint';c['output']=str(folder/'learned')
            if method in {'gvo','rloo','grpo'}:c['training']['estimator']=method
            if method=='no_p':c['training']['surface_weight']=0
            if method=='item_reward':c['training']['reward_mode']='completed_items'
            if method in {'outer_blind','inner_blind','evidence_blind'}:c['collection']['observation_mode']=method
            if method in {'additive','blind_additive'}:c['collection']['batch_mode']='additive'
            if method=='blind_additive':c['collection']['observation_mode']='outer_blind'
            rows.append({'kind':'gvo','method':method,'seed':seed,'config':c})
    return rows


def output_evidence(folder):
    """Include link identities and the entire produced file set when resuming."""
    files,links={},{}
    for path in folder.rglob('*'):
        if path.name=='completed.json' and path.parent==folder:
            continue
        if path.is_symlink():
            links[str(path)]=str(path.readlink())
        if path.is_file():
            files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return files,links


def execute(row):
    cfg=copy.deepcopy(row['config']);folder=Path(cfg['output']).resolve();cfg['output']=str(folder)
    def input_hash(path):
        path=Path(path)
        return hashlib.sha256((path if path.is_absolute() else ROOT/path).read_bytes()).hexdigest()
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/specrsi').rglob('*.py')}
    for p in (ROOT/'scripts/train.py', Path(__file__).resolve()):
        sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    sections=[cfg]
    if cfg.get('stage')=='two_stage':
        from specrsi.learning.pipeline import stage_configs
        sections.extend(stage_configs(cfg,folder))
    # Freeze effective stage inputs, including an overridden warmup curriculum.
    for section in sections:
        if section.get('data'):
            p=section['data'];sources[p]=input_hash(p)
        donors=section.get('collection',{}).get('observation_donors')
        if isinstance(donors,str):sources[donors]=input_hash(donors)
        collection=section.get('collection',{})
        if collection.get('domain'):
            for p in (ROOT/'configs/science').rglob('*.json'):
                sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        elif collection.get('dataset'):
            p=Path(collection.get('data_root','data'))/'manifests'/(collection['dataset']+'.json')
            sources[str(p)]=input_hash(p)
        # Freeze the configured backbone instead of an unrelated Qwen lock.
        p=section.get('collection',{}).get('executor',{}).get('lock')
        if p:
            sources[p]=input_hash(p)
    if cfg.get('paper_path'):
        p=Path(cfg['paper_path']);sources[str(p)]=input_hash(p)
    if cfg.get('resume'):
        resumed=Path(cfg['resume'])
        for p in (resumed if resumed.is_absolute() else ROOT/resumed).rglob('*'):
            if p.is_file():sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    identity=hashlib.sha256(json.dumps({'config':cfg,'sources':sources},sort_keys=True).encode()).hexdigest()
    marker=folder/'completed.json'
    if marker.exists():
        m=json.loads(marker.read_text())
        files,links=output_evidence(folder)
        if m['identity']==identity and files==m['outputs'] and links==m.get('output_links',{}):return
        raise ValueError(f'Existing replica inputs or outputs changed: {folder}; choose a new output root')
    if folder.exists():raise ValueError(f'Unmarked training output exists; preserve it and use a new root: {folder}')
    # PDF trainers create their own output directory exclusively. Launcher files
    # belong beside it; creating folder here previously made every PDF run fail.
    launch=folder.with_name(folder.name+'.launch')
    launch.mkdir(parents=True,exist_ok=False)
    frozen=launch/'run.yaml';frozen.write_text(yaml.safe_dump(cfg,sort_keys=False))
    with (launch/'run.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--config',str(frozen)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics=json.loads((folder/'metrics.json').read_text())
    stage_metrics=[metrics]
    if cfg.get('stage')=='two_stage':
        if not metrics.get('complete'):raise ValueError('Incomplete two-stage replica')
        stage_metrics=[json.loads((folder/stage/'metrics.json').read_text()) for stage in ('warmup','gvo')]
        if not (folder/'checkpoint').is_dir() or (folder/'checkpoint').resolve()!=(folder/'gvo/checkpoint').resolve():
            raise ValueError('The published checkpoint must point to the completed GVO stage')
    stage_names = ['warmup', 'gvo'] if cfg.get('stage') == 'two_stage' else [cfg.get('stage')]
    signals = []
    from specrsi.learning.trainer import merge_learning_signals
    for stage, m in zip(stage_names, stage_metrics):
        model_config = cfg.get(stage, {}).get('model', cfg.get('model', {})) if stage else cfg.get('model', {})
        v3 = model_config.get('architecture_version', 2) == 3
        minimum_updates = 0 if v3 and stage == 'gvo' else 1
        if (not m.get('history') or m.get('source_changed_during_run')
                or type(m.get('optimizer_updates')) is not int or m['optimizer_updates'] < minimum_updates
                or m['history'][-1].get('steps',0)-m.get('initial_optimizer_steps',0)!=m['optimizer_updates']):
            raise ValueError('Every training stage requires actual optimizer updates and unchanged sources')
        signal = merge_learning_signals(item.get('learning_signal') for item in m['history'])
        if v3 and not signal['diagnostics_complete']:
            raise ValueError('v3 replicas require explicit role update evidence, including zero-signal GVO')
        if v3 and signal['optimizer_steps'] != m['optimizer_updates']:
            raise ValueError('Observed v3 role evidence disagrees with the actual optimizer update count')
        if v3 and not (Path(m.get('checkpoint', folder / 'checkpoint')) / 'training_evidence.json').is_file():
            raise ValueError('v3 replicas require the checkpoint training evidence manifest')
        signals.append({'stage': stage, 'learning_signal': signal})
    for name,h in sources.items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Training input changed during replica: '+name)
    files,links=output_evidence(folder)
    marker.write_text(json.dumps({'identity':identity,'sources':sources,'outputs':files,'output_links':links,
        'stage_learning_signals': signals,
        'completion_scope': 'Execution and immutable outputs verified; role learning is described by stage evidence, not by global optimizer counts.'},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/training/qwen35_paper33_science.yaml')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=None)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
    parser.add_argument('--output',default='artifacts/training/qwen35_paper33_20261010/replicas')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();rows=plan(args.config,args.methods,args.seeds,args.output)
    if not args.execute:
        print(json.dumps({'prospective':True,'jobs':rows,'execution_requested':False},indent=2));return
    for row in rows:execute(row)
if __name__=='__main__':main()
