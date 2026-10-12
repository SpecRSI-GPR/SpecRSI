#!/usr/bin/env python3
"""Verify official DockerHub manifests/blobs and import native Lite dev images.

This transport fallback uses an accessible public byte mirror. SHA256 identities
come from official DockerHub; no image files are extracted by this script.
"""
import argparse,concurrent.futures,fcntl,hashlib,io,json,os,re,subprocess,sys,tarfile,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import requests
from specrsi.benchmarks.swe_native import load_rows,client,make_spec,official_development_image,verified_image_base


def authenticated(session,base,repository):
    url=base+'/v2/'+repository+'/manifests/latest'
    reply=session.get(url,timeout=60)
    if reply.status_code==401:
        values=dict(re.findall(r'(\w+)="([^"]+)"',reply.headers['WWW-Authenticate']))
        realm=values.pop('realm');token=session.get(realm,params=values,timeout=60);token.raise_for_status()
        data=token.json();session.headers['Authorization']='Bearer '+(data.get('token') or data['access_token'])
        reply=session.get(url,headers={'Accept':'application/vnd.docker.distribution.manifest.v2+json'},timeout=60)
    reply.raise_for_status();return reply.content


def import_one(row,cache,state):
    source=official_development_image(row)
    repository=source.split('/',1)[1].split('@')[0];expected=source.split('@')[1]
    session=requests.Session();base='https://docker.1ms.run'
    manifest_bytes=authenticated(session,base,repository)
    if 'sha256:'+hashlib.sha256(manifest_bytes).hexdigest()!=expected:
        raise ValueError('Mirror latest differs from the official primary manifest')
    manifest=json.loads(manifest_bytes)
    descriptors=[manifest['config'],*manifest['layers']]
    blobs=cache/'blobs/sha256';blobs.mkdir(parents=True,exist_ok=True)
    for descriptor in descriptors:
        key=descriptor['digest'].split(':')[1];target=blobs/key
        with (blobs/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if target.exists() and target.stat().st_size==descriptor['size']:
                # Cache files are only published after complete SHA256 checks.
                continue
            for attempt in range(3):
                try:
                    response=session.get(base+'/v2/'+repository+'/blobs/'+descriptor['digest'],stream=True,timeout=(30,180))
                    response.raise_for_status();hasher=hashlib.sha256();count=0
                    with target.with_suffix('.part').open('wb') as handle:
                        for chunk in response.iter_content(8*1024*1024):
                            if chunk:handle.write(chunk);hasher.update(chunk);count+=len(chunk)
                    if hasher.hexdigest()!=key or count!=descriptor['size']:
                        raise ValueError('Blob digest/size mismatch')
                    target.with_suffix('.part').replace(target);break
                except Exception:
                    if attempt==2:raise
            print('BLOB',row['instance_id'],key[:12],descriptor['size'],flush=True)
    # A standard OCI layout can reference verified Docker layer/config bytes.
    oci={**manifest,'mediaType':'application/vnd.oci.image.manifest.v1+json',
         'config':{**manifest['config'],'mediaType':'application/vnd.oci.image.config.v1+json'},
         'layers':[{**layer,'mediaType':'application/vnd.oci.image.layer.v1.tar+gzip'} for layer in manifest['layers']]}
    encoded=json.dumps(oci,separators=(',',':')).encode();key=hashlib.sha256(encoded).hexdigest()
    alias=make_spec(row).instance_image_key
    index={'schemaVersion':2,'manifests':[{'mediaType':'application/vnd.oci.image.manifest.v1+json',
        'digest':'sha256:'+key,'size':len(encoded),'annotations':{'org.opencontainers.image.ref.name':alias}}]}
    with tempfile.TemporaryDirectory(prefix='specrsi-dev-oci-',dir=cache) as directory:
        archive=Path(directory)/'image.tar'
        with tarfile.open(archive,'w') as tar:
            for name,data in [('oci-layout',b'{"imageLayoutVersion":"1.0.0"}'),('index.json',json.dumps(index).encode()),('blobs/sha256/'+key,encoded)]:
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
            for digest in dict.fromkeys(d['digest'].split(':')[1] for d in descriptors):
                tar.add(blobs/digest,arcname='blobs/sha256/'+digest,recursive=False)
        command=state['command'][:state['command'].index('system')]+['load','--input',str(archive)]
        result=subprocess.run(command,env={**os.environ,'CONTAINERS_CONF':state['containers_conf'],
            'PODMAN_IGNORE_CGROUPSV1_WARNING':'1'},capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr[-3000:])
    c=client(state['docker_host']);image=c.images.get(manifest['config']['digest'])
    probe=c.containers.run(image.id,command=['tail','-f','/dev/null'],detach=True,network_mode='none')
    try:
        base_identity=verified_image_base(probe,row['base_commit'])
    finally:probe.remove(force=True)
    name,tag=alias.rsplit(':',1);image.tag(name,tag=tag)
    return {'instance_id':row['instance_id'],'source':source,'image_id':image.id,'alias':alias,
        **base_identity,'digests':image.attrs.get('RepoDigests',[]),'primary_manifest_sha256':expected,
        'transport':'primary_manifest_verified_public_mirror_blobs_OCI_import',
        'all_blobs_sha256_verified':True,'verified_unix':time.time()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ids',nargs='+')
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--cache',default='/tmp/specrsi-swe-official-oci')
    parser.add_argument('--output',default='reports/swe_development_images.json');args=parser.parse_args()
    registry=json.loads(Path('configs/coding/prospective_swe_tracks.json').read_text());ids=args.ids or registry['development23']
    if not set(ids)<=set(registry['development23']):raise ValueError('Importer is restricted to registered development IDs')
    output=Path(args.output);report=json.loads(output.read_text()) if output.exists() else {'images':{},'failures':{}}
    report.update(requested_ids=ids,complete=False,track='development23')
    cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    state=json.loads(Path('reports/swe_container_service.json').read_text());rows=load_rows()
    lock_path=Path('configs/coding/swe_images.lock.json');lock=json.loads(lock_path.read_text())
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        pending={pool.submit(import_one,rows[key],cache,state):key for key in ids if key not in report['images']}
        for future in concurrent.futures.as_completed(pending):
            key=pending[future]
            try:
                receipt=future.result();report['images'][key]=receipt;report['failures'].pop(key,None)
                lock['images'][key]={field:receipt[field] for field in ('source','image_id','base_commit')}
                temporary=lock_path.with_suffix('.tmp');temporary.write_text(json.dumps(lock,indent=2));temporary.replace(lock_path)
                print('READY',key,flush=True)
            except Exception as exc:
                report['failures'][key]={'error':str(exc),'type':type(exc).__name__};print('FAILED',key,str(exc),flush=True)
            report['updated_unix']=time.time();temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(output)
    report['complete']=all(key in report['images'] for key in ids);output.write_text(json.dumps(report,indent=2)+'\n')
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
