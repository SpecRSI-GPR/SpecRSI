"""A dead API parent must not strand a GPU worker or transfer process ownership."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process identity; covered by Linux process checks")


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/deploy_qwen35_bf16.py'
    spec = importlib.util.spec_from_file_location('qwen35_deployment_test', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_birth_mismatch_never_confers_group_ownership():
    deploy = module()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        record = {'pid': child.pid, 'birth_ticks': deploy.birth(child.pid),
                  'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        assert deploy.alive(record) and deploy.group_alive(record)
        assert not deploy.alive({**record, 'birth_ticks': '-1'})
        assert not deploy.group_alive({**record, 'birth_ticks': '-1'})
        assert not deploy.group_alive({**record, 'boot_id': 'different-host-boot'})
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_worker_group_remains_owned_after_parent_exit():
    deploy = module()
    code = ('import subprocess,sys,time; '
            'subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"], '
            'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
            'print("ready",flush=True); time.sleep(.3)')
    parent = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                              stdout=subprocess.PIPE, text=True)
    record = {'pid': parent.pid, 'birth_ticks': deploy.birth(parent.pid),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    try:
        assert parent.stdout.readline().strip() == 'ready'
        parent.wait(timeout=10)
        assert not deploy.alive(record)
        assert deploy.group_alive(record)
    finally:
        # The test created this private session; no external process is targeted.
        try:
            os.killpg(parent.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        parent.wait(timeout=10)
