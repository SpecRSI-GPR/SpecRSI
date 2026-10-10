"""Strict interfaces for original simulator and repository-test harnesses."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

from .base import BenchmarkExample, MissingArtifactError, ScoreResult, canonical_json
from .data import sha256_file


@dataclasses.dataclass(frozen=True)
class NativeHarness:
    """Native protocol: one request JSON on stdin; one result JSON on stdout.

    The executable is trusted evaluator code. Candidate code must be isolated
    inside that evaluator's declared environment; do not run arbitrary candidates
    through this wrapper itself. Scientific units and transforms must be supplied.
    """
    command: tuple[str, ...]
    artifact_hashes: Mapping[str, str]
    evaluator_identity: str
    native_unit: str
    timeout_seconds: float = 600.0
    score_min: float | None = None
    score_max: float | None = None

    def validate(self):
        if not self.command or not self.evaluator_identity or not self.native_unit or not self.artifact_hashes:
            raise MissingArtifactError("Native harness needs command, identity, units, and pinned artifact hashes")
        for filename, expected in self.artifact_hashes.items():
            if not Path(filename).is_file():
                raise MissingArtifactError(f"Native evaluator artifact not supplied: {filename}")
            if sha256_file(filename) != expected:
                raise ValueError(f"Native evaluator artifact changed: {filename}")

    def score(self, example: BenchmarkExample, candidate: Mapping) -> ScoreResult:
        self.validate()
        request = {"schema_version": 1, "example": example.to_dict(), "candidate": candidate,
                   "evaluator_identity": self.evaluator_identity}
        completed = subprocess.run(self.command, input=canonical_json(request), text=True, capture_output=True,
                                   timeout=self.timeout_seconds, check=True)
        response = json.loads(completed.stdout)
        if response.get("example_id") != example.id or response.get("evaluator_identity") != self.evaluator_identity:
            raise ValueError("Native evaluator result identity mismatch")
        if response.get("native_unit") != self.native_unit or response.get("complete") is not True:
            raise ValueError("Native evaluator must return a complete scenario in its registered units")
        score = float(response["score"])
        if not math.isfinite(score) or (self.score_min is not None and score < self.score_min) or (self.score_max is not None and score > self.score_max):
            raise ValueError("Native evaluator returned an out-of-domain score")
        return ScoreResult(score, {**response.get("metrics", {}), "native_unit": self.native_unit,
                                   "evaluator_identity": self.evaluator_identity})


SCIENCE_REQUIRED_ARTIFACTS = {
    "pybamm": ["scenario inputs and seeds", "PyBaMM version and parameter set", "controller API", "native scoring transform and constraints"],
    "aqm": ["network topology and traffic traces", "packet simulator version", "AQM controller API", "native scoring transform and constraints"],
    "fdtd": ["geometry and material parameters", "source/boundary/grid/convergence settings", "FDTD solver version", "native scoring transform and constraints"],
    "reactor": ["kinetics and initial/boundary conditions", "simulator version", "control horizon and controller API", "native scoring transform and constraints"],
    "irp": ["inventory/routing instances and seeds", "demand and cost models", "feasibility checker", "native scoring transform and constraints"],
}


def require_scientific_harness(domain: str, config_path: str | Path) -> NativeHarness:
    if domain not in SCIENCE_REQUIRED_ARTIFACTS:
        raise ValueError(f"Unknown science domain: {domain}")
    path = Path(config_path)
    if not path.is_file():
        raise MissingArtifactError(f"{domain} requires original artifacts: {', '.join(SCIENCE_REQUIRED_ARTIFACTS[domain])}")
    config = json.loads(path.read_text())
    harness = NativeHarness(**config)
    harness.validate()
    return harness


def mbpp_score(code: str, reference: Mapping, *, image: str, timeout_seconds: int = 15, runtime: str = "docker") -> ScoreResult:
    """Execute all MBPP tests in an explicitly pinned, networkless container.

    Infrastructure failures raise; test failures score zero. The benchmark code
    and tests share the container because MBPP tests call generated functions.
    This is resource isolation, not protection against a hostile container escape.
    """
    if "@sha256:" not in image:
        raise ValueError("Use a digest-pinned Python container image")
    if shutil.which(runtime) is None:
        raise MissingArtifactError(f"MBPP requires the {runtime} container runtime")
    tests = reference.get("test_list", [])
    if not tests:
        raise ValueError("MBPP reference has no test_list")
    import uuid
    container_name = "specrsi-mbpp-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="specrsi-mbpp-") as directory:
        source = "\n".join([reference.get("test_setup_code", ""), code, *tests])
        Path(directory, "test.py").write_text(source)
        command = [runtime, "run", "--rm", "--name", container_name, "--network=none", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=256m", "--cpus=1",
                   "--user=65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "-v", f"{directory}:/work:ro",
                   "-w", "/work", image, "python", "-I", "/work/test.py"]
        # TemporaryDirectory is 0700; the unprivileged container needs read access.
        Path(directory).chmod(0o755)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            subprocess.run([runtime, "rm", "-f", container_name], capture_output=True, timeout=15)
            return ScoreResult(0.0, {"passed": False, "timeout": True, "tests": len(tests)})
    if result.returncode in {125, 126, 127}:
        raise MissingArtifactError(f"MBPP container failed to start: {result.stderr[-2000:]}")
    return ScoreResult(float(result.returncode == 0), {"passed": result.returncode == 0, "tests": len(tests),
                                                       "exit_code": result.returncode, "stderr": result.stderr[-2000:]})


def mbpp_bubblewrap_score(code: str, reference: Mapping, *, timeout_seconds: float = 10,
                         memory_mb: int = 512, python_executable: str = "/usr/local/bin/python") -> ScoreResult:
    """Run MBPP assertions inside Linux user/mount/PID/network namespaces.

    Only read-only interpreter/system directories and the test file are mounted;
    workspace, credentials, and home directories are absent. Bubblewrap startup
    errors are infrastructure failures, while assertion/runtime failures score 0.
    """
    import os
    import resource
    import signal
    import uuid
    runtime = shutil.which("bwrap")
    if runtime is None or not Path(python_executable).is_file():
        raise MissingArtifactError("MBPP needs bubblewrap and an interpreter under /usr or /bin")
    resolved_python = Path(python_executable).resolve()
    if not any(resolved_python.is_relative_to(p) for p in [Path("/usr"), Path("/bin")]):
        raise ValueError("Sandbox interpreter must resolve under /usr or /bin")
    tests = reference.get("test_list", [])
    if not tests or timeout_seconds <= 0 or memory_mb < 64:
        raise ValueError("Supply nonempty MBPP tests and valid resource limits")
    completion = "SPECRSI_TESTS_COMPLETE_" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="specrsi-mbpp-") as directory:
        file = Path(directory, "test.py")
        # A completion record prevents os._exit(0) and early process exits from
        # counting as a pass. The sandbox is for containment, not a proof that
        # intentionally malicious code cannot introspect a Python test harness.
        file.write_text("\n".join([reference.get("test_setup_code", ""), code, *tests, f"print({completion!r})"]))
        command = [runtime, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
        for directory_name in ["/usr", "/bin", "/lib", "/lib64"]:
            if Path(directory_name).exists():
                command.extend(["--ro-bind", directory_name, directory_name])
        command.extend(["--dir", "/etc"])
        if Path("/etc/ld.so.cache").exists():
            command.extend(["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"])
        command.extend(["--dir", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/work",
                        "--ro-bind", str(file), "/work/test.py", "--chdir", "/work", "--clearenv",
                        "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin", "--setenv", "HOME", "/tmp",
                        "--setenv", "PYTHONDONTWRITEBYTECODE", "1", python_executable, "-I", "/work/test.py"])
        def limits():
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024**2, memory_mb * 1024**2))
            cpu = max(1, math.ceil(timeout_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Bound disk output and read it after process exit, avoiding unbounded
        # capture_output memory on generated print loops.
        with Path(directory, "stdout").open("w+") as stdout, Path(directory, "stderr").open("w+") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True, preexec_fn=limits)
            timed_out = False
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stdout.seek(0)
            stderr.seek(0)
            output, error = stdout.read(1 << 20), stderr.read(1 << 20)
    if error.startswith("bwrap:"):
        raise MissingArtifactError(f"Bubblewrap startup failed: {error[-2000:]}")
    passed = not timed_out and process.returncode == 0 and completion in output.splitlines()
    return ScoreResult(float(passed), {"passed": passed, "timeout": timed_out, "tests": len(tests),
                                      "exit_code": process.returncode, "stderr": error[-2000:],
                                      "isolation": "bubblewrap_user_mount_pid_network_namespaces"})


def swebench_command(predictions: str | Path, *, dataset_name: str = "princeton-nlp/SWE-bench_Verified",
                     run_id: str, workers: int = 4, instance_ids: Sequence[str] = ()) -> list[str]:
    """Construct the official harness invocation; predictions are real patches."""
    if not Path(predictions).is_file():
        raise MissingArtifactError(f"Missing SWE-bench prediction patches: {predictions}")
    command = [sys.executable, "-m", "swebench.harness.run_evaluation", "--dataset_name", dataset_name,
               "--predictions_path", str(predictions), "--max_workers", str(workers), "--run_id", run_id]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    return command


def swebench_score(report: Mapping, instance_id: str) -> ScoreResult:
    instance = report.get(instance_id)
    if not isinstance(instance, Mapping) or type(instance.get("resolved")) is not bool:
        raise ValueError("Expected the official per-instance report with a resolved boolean")
    return ScoreResult(float(instance["resolved"]), {"resolved": instance["resolved"], "instance_id": instance_id})
