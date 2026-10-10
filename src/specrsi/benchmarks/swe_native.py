"""Pinned, unmodified SWE-bench harness integration with local OCI image receipts.

Model inputs are deliberately separate from sealed gold/test patches. Native
evaluation runs on a fresh container after the submitted patch is frozen.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

MIRROR = 'ghcr.io/epoch-research/swe-bench.eval.x86_64.'
DOCKER_HOST = 'unix:///tmp/specrsi-podman-api/podman.sock'


class _FixtureContainer:
    def __init__(self,underlying):self.underlying=underlying
    def __getattr__(self,name):return getattr(self.underlying,name)
    def start(self,*args,**kwargs):
        self.underlying.start(*args,**kwargs)
        # Direct HTTPAdapter tests bypass REQUESTS_CA_BUNDLE. Register the local
        # fixture CA in existing CA stores, preserving every original CA. This
        # changes trust configuration only; test scripts/assertions are intact.
        command='for f in /testbed/requests/cacert.pem /opt/miniconda3/envs/testbed/lib/python*/site-packages/certifi/cacert.pem /opt/miniconda3/envs/testbed/ssl/cert.pem; do if [ -f "$f" ]; then cat /specrsi-httpbin-ca.crt >> "$f"; fi; done'
        result=self.underlying.exec_run(['bash','-lc',command])
        if result.exit_code:raise RuntimeError('Could not install the local HTTPBin fixture CA')


class _FixtureContainers:
    def __init__(self, collection, url, certificate):
        self.collection,self.url,self.certificate=collection,url,certificate
    def __getattr__(self,name):
        return getattr(self.collection,name)
    def create(self,*args,**kwargs):
        kwargs['network_mode']='host'
        kwargs['environment']={**kwargs.get('environment',{}),'HTTPBIN_URL':self.url,
            'HTTP_PROXY':'','HTTPS_PROXY':'','ALL_PROXY':'','NO_PROXY':'*',
            'REQUESTS_CA_BUNDLE':'/specrsi-httpbin-ca.crt'}
        kwargs['extra_hosts']={'httpbin.org':'127.0.0.1','www.httpbin.org':'127.0.0.1','www.google.co.uk':'127.0.0.1'}
        kwargs['volumes']={self.certificate:{'bind':'/specrsi-httpbin-ca.crt','mode':'ro'}}
        return _FixtureContainer(self.collection.create(*args,**kwargs))


class _FixtureClient:
    """Only supply the test suite's documented HTTPBIN_URL runtime fixture."""
    def __init__(self, underlying,url,certificate):
        self.underlying=underlying;self.containers=_FixtureContainers(underlying.containers,url,certificate)
    def __getattr__(self,name):
        return getattr(self.underlying,name)


def load_rows(root='data/raw'):
    import pyarrow.parquet as pq
    rows = {}
    for dataset in ('swe_lite', 'swe_verified'):
        for path in sorted((Path(root) / dataset).glob('*.parquet')):
            for row in pq.read_table(path).to_pylist():
                rows[row['instance_id']] = row
    return rows


def registered_tracks(root, *, seed: int, selection_count: int, excluded_ids=()):
    """Build disjoint selection and audit tracks from caller-provided local rows."""
    import pyarrow.parquet as pq
    if type(seed) is not int or type(selection_count) is not int or selection_count < 0:
        raise ValueError("An integer seed and nonnegative selection_count are required")
    def read_rows(dataset, split):
        return [row for path in sorted((Path(root) / dataset).glob(f"{split}-*.parquet"))
                for row in pq.read_table(path).to_pylist()]
    verified = read_rows("swe_verified", "test")
    lite = read_rows("swe_lite", "test")
    development = read_rows("swe_lite", "dev")
    excluded = set(excluded_ids)
    verified_ids = {row["instance_id"] for row in verified}
    ordered = sorted(verified_ids - excluded,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    if selection_count > len(ordered):
        raise ValueError("selection_count exceeds the supplied eligible rows")
    selected = ordered[:selection_count]
    return {"seed": seed, "selection_count": selection_count,
            "excluded_ids": sorted(excluded), "selection": selected,
            "development": sorted({row["instance_id"] for row in development} - excluded),
            "verified": sorted(verified_ids - excluded),
            "audit": sorted({row["instance_id"] for row in lite} - set(selected) - excluded),
            "audit_disjoint_verified": sorted({row["instance_id"] for row in lite} - verified_ids - excluded)}


def public_problem(row):
    return {key: row[key] for key in ('instance_id', 'repo', 'base_commit', 'problem_statement')}


def client(host=DOCKER_HOST):
    import docker
    return docker.DockerClient(base_url=host, timeout=1800)


def make_spec(row):
    from swebench.harness.test_spec.test_spec import make_test_spec
    # An explicit local alias of the downloaded immutable digest. Remote=True
    # makes the official harness use the existing image without rebuilding it.
    return make_test_spec(row, namespace='localhost/specrsi-swe')


def official_development_image(row):
    """Pin primary DockerHub manifest, fetch identical bytes via public mirror.

    Resolve the primary publisher's manifest and use its immutable digest for
    mirror retrieval. The OCI engine validates every content-addressed layer.
"""
    import requests
    repository='swebench/sweb.eval.x86_64.'+row['instance_id'].replace('__','_1776_').lower()
    session=requests.Session()
    response=session.get('https://auth.docker.io/token',params={'service':'registry.docker.io',
        'scope':'repository:'+repository+':pull'},timeout=60)
    response.raise_for_status();token=response.json()['token']
    response=session.get('https://registry-1.docker.io/v2/'+repository+'/manifests/latest',
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.docker.distribution.manifest.v2+json'},timeout=60)
    response.raise_for_status()
    digest='sha256:'+hashlib.sha256(response.content).hexdigest()
    if response.headers.get('Docker-Content-Digest')!=digest:
        raise ValueError('Primary DockerHub manifest content digest mismatch')
    return 'docker.1ms.run/'+repository+'@'+digest


def verified_image_base(container, expected):
    """Accept exact base or the publisher's one empty setup commit only."""
    result=container.exec_run(['git','-C','/testbed','rev-parse','HEAD'])
    head=result.output.decode().strip()
    if result.exit_code:raise ValueError('Cannot read native image Git HEAD')
    if head==expected:return {'image_head':head,'base_commit':expected,'empty_setup_commit':False}
    result=container.exec_run(['git','-C','/testbed','rev-parse','HEAD^','HEAD^{tree}',expected+'^{tree}'])
    values=result.output.decode().splitlines()
    subject=container.exec_run(['git','-C','/testbed','log','-1','--format=%s'])
    if result.exit_code or len(values)!=3 or values[0]!=expected or values[1]!=values[2] or subject.output.decode().strip()!='SWE-bench':
        raise ValueError(f'Image base commit mismatch: {head} != {expected}')
    return {'image_head':head,'base_commit':expected,'empty_setup_commit':True,'verified_source_tree':values[1]}


def ensure_image(row, docker_client=None, *, pull=True):
    import docker.errors
    c = docker_client or client()
    source = MIRROR + row['instance_id'].lower() + ':latest'
    lock_path=Path('configs/coding/swe_images.lock.json')
    locked=json.loads(lock_path.read_text()).get('images',{}).get(row['instance_id']) if lock_path.exists() else None
    if locked:
        if locked['base_commit']!=row['base_commit']:raise ValueError('Frozen image lock does not match task base commit')
        source=locked['source']
    elif row['instance_id'] in json.loads(Path('configs/coding/swe_tracks.json').read_text()).get('development',[]):
        source=official_development_image(row)
    try:
        image = c.images.get(source)
    except docker.errors.ImageNotFound:
        # OCI imports preserve the config/image digest but may not attach the
        # remote manifest RepoDigest. The immutable image lock is sufficient.
        try:
            image=c.images.get(locked['image_id']) if locked else None
        except docker.errors.ImageNotFound:
            image=None
        if image is None:
            if not pull:raise
            image = c.images.pull(source)
    if locked and image.id!=locked['image_id']:
        raise ValueError('Pulled image differs from the frozen image identity')
    labels = image.attrs.get('Config', {}).get('Labels', {}) or {}
    # Official SWE images may omit epoch's identity label. Their primary digest
    # and actual repository base commit jointly identify the requested runtime.
    declared=labels.get('swe-bench.instance_id')
    official='swebench/sweb.eval.x86_64.' in source and '@sha256:' in source
    if declared != row['instance_id'] and not (declared is None and official):
        raise ValueError('Mirror image identity does not match instance')
    spec = make_spec(row)
    repository, tag = spec.instance_image_key.rsplit(':', 1)
    image.tag(repository, tag=tag)
    # Base commit is checked in the actual runtime, not inferred from tag names.
    probe = c.containers.run(image.id, command=['tail','-f','/dev/null'],
                             detach=True, network_mode='none')
    try:
        base=verified_image_base(probe,row['base_commit'])
    finally:
        probe.remove(force=True)
    return {'instance_id': row['instance_id'], 'source': source, 'image_id': image.id,
            'identity_label':declared,'primary_digest_verified':official,
            'digests': image.attrs.get('RepoDigests', []), 'alias': spec.instance_image_key,
            'size_bytes': image.attrs.get('Size'), **base,
            'verified_unix': time.time()}


def evaluate_patch(row, patch, *, run_id, model_name='specrsi-local', timeout=1800,
                   docker_client=None, image_receipt=None):
    from swebench.harness.run_evaluation import run_instance
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in run_id):
        raise ValueError('run_id must be a safe explicit run identifier')
    c = docker_client or client()
    receipt = image_receipt or ensure_image(row, c)
    fixture=None
    if row['repo']=='psf/requests':
        import urllib.request
        url='http://httpbin.org'
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1/get',timeout=5).close()
        certificate=Path('artifacts/coding/httpbin_tls/fixture.crt').resolve()
        c=_FixtureClient(c,url,str(certificate))
        fixture={'HTTPBIN_URL':url,'network_mode':'host','httpbin_version':'0.10.4',
                 'dns_override':{'httpbin.org':'127.0.0.1','www.httpbin.org':'127.0.0.1','www.google.co.uk':'127.0.0.1'},
                 'trust_configuration':'append_fixture_ca_to_existing_CA_stores_v1',
                 'ca_sha256':hashlib.sha256(certificate.read_bytes()).hexdigest(),
                 'policy':'native_suite_supported_environment_variable_no_test_or_assertion_changes'}
    prediction = {'instance_id': row['instance_id'], 'model_name_or_path': model_name,
                  'model_patch': patch}
    patch_hash = hashlib.sha256(patch.encode()).hexdigest()
    # Include patch identity so the upstream report cache cannot alias two patches.
    protocol={'fixture':fixture,'image_id':receipt['image_id'],
              'harness_version':importlib.metadata.version('swebench'),
              'integration_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'eval_script_sha256':hashlib.sha256(make_spec(row).eval_script.encode()).hexdigest()}
    fixture_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:8]
    native_run = f'{run_id}-{patch_hash[:12]}-{fixture_hash}'
    started = time.monotonic()
    result = run_instance(make_spec(row), prediction, False, False, c, native_run, timeout)
    log = Path('logs/run_evaluation')/native_run/model_name.replace('/', '__')/row['instance_id']
    report = json.loads((log/'report.json').read_text()) if (log/'report.json').exists() else None
    return {'instance_id': row['instance_id'], **result, 'score': float(result['resolved']) if result['completed'] else None,
            'official_report': report, 'log_dir': str(log), 'patch_sha256': patch_hash,
            'wall_seconds': time.monotonic()-started, 'harness': 'swebench',
            'harness_version': importlib.metadata.version('swebench'), 'image': receipt,
            'runtime_fixture': fixture,
            'runtime_protocol': protocol,
            'test_protocol': 'official_FAIL_TO_PASS_and_PASS_TO_PASS'}
