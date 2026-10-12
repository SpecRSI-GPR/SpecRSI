"""Native workers retain campaign ownership without inheriting host settings."""
import json

import pytest

from specrsi.science import executor


@pytest.mark.parametrize("supervised", [False, True])
def test_native_worker_environment_is_closed_except_for_ownership_tags(monkeypatch, supervised):
    scope=executor.make_scope("irp","development",(0,))
    task=scope.tasks[0]
    candidate=executor.make_candidate("irp")
    tags={"SPECRSI_JOB_DIR":"/tmp/test-campaign/jobs/example/attempt-0001",
          "SPECRSI_JOB_TOKEN":"fixture-ownership-token"}
    for key,value in tags.items():
        if supervised:
            monkeypatch.setenv(key,value)
        else:
            monkeypatch.delenv(key,raising=False)
    # Deliberately unrelated host settings must not cross the native boundary.
    for key,value in {"OPENAI_API_KEY":"fixture-only-not-a-credential",
                      "SPECRSI_JOB_UNREGISTERED":"fixture-unregistered",
                      "CUDA_VISIBLE_DEVICES":"7", "PYTHONPATH":"/tmp/untrusted",
                      "OMP_NUM_THREADS":"99"}.items():
        monkeypatch.setenv(key,value)
    captured={}

    class FakeWorker:
        returncode=0

        def communicate(self, request, timeout):
            captured["request"]=json.loads(request)
            return json.dumps({"evaluator_identity":task.metadata["evaluator_identity"],
                "complete":True,"native_score":0.,"valid":True,"specrsi_score":.5,
                "cpu_seconds":0.}),""

    def popen(command, **kwargs):
        captured.update(command=command,**kwargs)
        return FakeWorker()

    monkeypatch.setattr(executor.subprocess,"Popen",popen)
    result=executor.ScienceExecutor().run(candidate,task,scope,60000)
    assert result.status=="complete"
    env=captured["env"]
    expected={"PATH","HOME","PYTHONPATH","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
              "MKL_NUM_THREADS","NUMBA_NUM_THREADS","PYBAMM_DISABLE_TELEMETRY",
              "CUDA_VISIBLE_DEVICES","NUMBA_CACHE_DIR"}
    assert set(env)==expected | (set(tags) if supervised else set())
    assert all(env[key]==value for key,value in tags.items()) if supervised else not set(tags)&env.keys()
    assert env["CUDA_VISIBLE_DEVICES"]==""
    assert env["PYTHONPATH"]==str(executor.ROOT/"src")
    assert env["OMP_NUM_THREADS"]=="1"
    assert captured["start_new_session"] is True
    assert set(captured["request"])=={"domain","scenario_set","case_id","candidate"}
    assert not set(tags)&result.outcome.fields.keys()
