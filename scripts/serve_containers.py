#!/usr/bin/env python3
"""Start a persistent local Podman Docker-compatible API for native SWE-bench."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import signal
import ssl


def start_httpbin(python):
    url='http://127.0.0.1:18089'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    state=Path('reports/swe_httpbin_service.json')
    previous=json.loads(state.read_text()) if state.exists() else {}
    if previous.get('runtime_version')==2:
        try:
            opener.open(url+'/get',timeout=2).close()
            opener.open('http://127.0.0.1/get',timeout=2).close()
            https=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(
                context=ssl.create_default_context(cafile=previous['certificate'])))
            https.open('https://127.0.0.1/get',timeout=2).close()
            return
        except Exception:pass
    if not Path(python).exists():
        raise RuntimeError('Prepare the HTTPBin fixture first: python scripts/prepare_swe.py --bootstrap-fixtures')
    certificate=Path('artifacts/coding/httpbin_tls/fixture.crt').resolve()
    key=certificate.with_suffix('.key')
    if not certificate.exists():
        script='''import datetime,ipaddress,sys,pathlib
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import rsa
p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
k=rsa.generate_private_key(public_exponent=65537,key_size=2048)
name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'SpecRSI local HTTPBin fixture')])
now=datetime.datetime.now(datetime.timezone.utc)
c=x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(days=1)).not_valid_after(now+datetime.timedelta(days=3650)).add_extension(x509.BasicConstraints(ca=True,path_length=None),critical=True).add_extension(x509.SubjectAlternativeName([x509.DNSName('httpbin.org'),x509.DNSName('www.httpbin.org'),x509.DNSName('localhost'),x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False).sign(k,hashes.SHA256())
p.write_bytes(c.public_bytes(serialization.Encoding.PEM))
q=p.with_suffix('.key');q.write_bytes(k.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.TraditionalOpenSSL,serialization.NoEncryption()));q.chmod(0o600)
'''
        subprocess.run([python,'-c',script,str(certificate)],check=True)
    for pid in previous.get('pids',[previous.get('pid')]):
        if pid:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    commands=[[python,'-m','gunicorn','--bind','127.0.0.1:18089','--bind','127.0.0.1:80','--workers','4','httpbin:app'],
              [python,'-m','gunicorn','--bind','127.0.0.1:443','--workers','4','--certfile',str(certificate),'--keyfile',str(key),'httpbin:app']]
    procs=[]
    with Path('logs/swe_httpbin_service.log').open('ab') as log:
        for command in commands:
            procs.append(subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True))
    for _ in range(80):
        try:
            opener.open(url+'/get',timeout=1).close()
            opener.open('http://127.0.0.1/get',timeout=1).close()
            https=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=str(certificate))))
            https.open('https://127.0.0.1/get',timeout=1).close()
            state.write_text(json.dumps(dict(pids=[p.pid for p in procs],url=url,runtime_version=2,
                httpbin_version='0.10.4',gunicorn_version='23.0.0',commands=commands,
                certificate=str(certificate)),indent=2)+'\n');return
        except Exception:time.sleep(.1)
    raise RuntimeError('HTTPBin fixture did not become healthy; see logs/swe_httpbin_service.log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/tmp/specrsi-podman-overlay-root')
    parser.add_argument('--runroot', default='/tmp/specrsi-podman-overlay-run')
    parser.add_argument('--storage-driver', choices=['overlay','vfs'], default='overlay')
    parser.add_argument('--socket', default='/tmp/specrsi-podman-api/podman.sock')
    parser.add_argument('--state', default='reports/swe_container_service.json')
    parser.add_argument('--httpbin-python',default='/tmp/specrsi-httpbin-venv/bin/python')
    args = parser.parse_args()
    Path('logs').mkdir(exist_ok=True);Path('reports').mkdir(exist_ok=True)
    start_httpbin(args.httpbin_python)
    socket = Path(args.socket)
    socket.parent.mkdir(parents=True, exist_ok=True)
    conf = socket.parent / 'containers.conf'
    conf.write_text('[containers]\ncgroups = "disabled"\nnetns = "none"\n[engine]\ncgroup_manager = "cgroupfs"\nevents_logger = "file"\n')
    command = ['podman', '--root', args.root, '--runroot', args.runroot,
               '--storage-driver', args.storage_driver]
    if args.storage_driver == 'overlay':
        command.extend(['--storage-opt','overlay.mount_program=/usr/bin/fuse-overlayfs'])
    command.extend(['system', 'service', '--time=0', 'unix://' + str(socket)])
    env = {**os.environ, 'CONTAINERS_CONF': str(conf), 'PODMAN_IGNORE_CGROUPSV1_WARNING': '1'}
    ping = ['curl', '--silent', '--fail', '--unix-socket', str(socket), 'http://localhost/_ping']
    if subprocess.run(ping, capture_output=True).returncode == 0:
        print('Container API already healthy: unix://' + str(socket))
        return
    log_path = Path('logs/swe_container_service.log'); log_path.parent.mkdir(exist_ok=True)
    with log_path.open('ab') as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, start_new_session=True)
    for _ in range(100):
        if subprocess.run(ping, capture_output=True).returncode == 0:
            value = dict(pid=proc.pid, docker_host='unix://' + str(socket), command=command,
                         containers_conf=str(conf), log=str(log_path), started_unix=time.time())
            Path(args.state).parent.mkdir(parents=True, exist_ok=True)
            Path(args.state).write_text(json.dumps(value, indent=2) + '\n')
            print(json.dumps(value, indent=2)); return
        if proc.poll() is not None:
            raise RuntimeError(f'Podman service exited {proc.returncode}; see {log_path}')
        time.sleep(.1)
    raise RuntimeError('Container API did not become healthy')


if __name__ == '__main__':
    main()
