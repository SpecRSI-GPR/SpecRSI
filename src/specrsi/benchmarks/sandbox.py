"""Bounded offline process execution for generated solutions and parsers."""
from __future__ import annotations
from pathlib import Path
import math, os, resource, shutil, signal, subprocess, sys, tempfile, time, stat
from .base import MissingArtifactError

ROOT=Path(__file__).resolve().parents[3]

def run_isolated(command: list[str], work: Path, *, timeout: float=180,
                 memory_mb: int=8192, mounts: dict[str,str]|None=None,
                 environment: dict[str,str]|None=None) -> dict:
    if sys.platform != 'linux':
        raise MissingArtifactError('Native sandbox execution requires Linux with bubblewrap namespaces')
    runtime=shutil.which('bwrap')
    if runtime is None:
        raise MissingArtifactError('Native sandbox execution requires bubblewrap; install it before evaluation')
    work=work.resolve()
    null=os.stat('/dev/null')
    if not stat.S_ISCHR(null.st_mode) or null.st_rdev!=os.makedev(1,3):
        raise MissingArtifactError('Host /dev/null is not the correct character device; repair runtime before evaluation')
    argv=[runtime,'--unshare-pid','--unshare-net','--unshare-ipc','--unshare-uts',
          '--die-with-parent','--new-session','--cap-drop','ALL']
    for directory in ['/usr','/bin','/lib','/lib64']:
        if Path(directory).exists(): argv.extend(['--ro-bind',directory,directory])
    argv+=['--dir','/etc']
    for name in ['ld.so.cache','java','fonts','alternatives','pki','ssl']:
        p=Path('/etc')/name
        if p.exists(): argv+=['--ro-bind',str(p),str(p)]
    for source,dest in (mounts or {}).items():
        if not Path(source).exists(): raise MissingArtifactError(f'Missing sandbox runtime: {source}')
        argv+=['--ro-bind',str(Path(source).resolve()),dest]
    argv+=['--proc','/proc','--dev','/dev','--tmpfs','/tmp','--bind',str(work),'/work',
           '--chdir','/work','--clearenv']
    env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
         'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',**(environment or {})}
    for key,value in env.items(): argv+=['--setenv',key,value]
    argv+=command
    def limits():
        memory=memory_mb*1024**2
        resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
        cpu=max(1,math.ceil(timeout)); resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE,(2048,2048))
        resource.setrlimit(resource.RLIMIT_NPROC,(256,256))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    with tempfile.TemporaryDirectory(prefix='specrsi-native-log-') as logs:
        started=time.monotonic(); timed_out=False
        with open(Path(logs)/'stdout','w+') as out, open(Path(logs)/'stderr','w+') as err:
            process=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True,preexec_fn=limits)
            try: process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True; os.killpg(process.pid,signal.SIGKILL); process.wait()
            out.seek(0); err.seek(0); stdout=out.read(4*1024**2); stderr=err.read(4*1024**2)
    if stderr.startswith('bwrap:'): raise MissingArtifactError(stderr[-2000:])
    return {'exit_code':process.returncode,'timeout':timed_out,'stdout':stdout,'stderr':stderr,
            'wall_seconds':time.monotonic()-started,'isolation':'bubblewrap_offline_namespaces'}
